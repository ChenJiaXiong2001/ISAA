"""Bounded STG adaptation; the learned node count has no fixed budget."""
import math
import torch
from isaa.models.input_joint_selection import InputJointSelectionFusion, JointStochasticGates

INPUT_SELECTION_V2_VARIANT = 'best-input-joint-selection-v2'

class BoundedJointGates(JointStochasticGates):
    def __init__(self, sigma=.5, initial_mu=.5, mu_min=-.5, mu_max=1.):
        super().__init__(sigma, initial_mu)
        if not all(math.isfinite(v) for v in (mu_min, mu_max)) or not mu_min < initial_mu < mu_max:
            raise ValueError('Gate initialization must be strictly inside finite projection bounds')
        if not 0. < mu_max <= 1. or mu_min >= 0.:
            raise ValueError('Gate bounds must straddle zero with upper bound at most 1')
        self.mu_min, self.mu_max = float(mu_min), float(mu_max)
        self.initial_mu = float(initial_mu)
        self.register_buffer('open_mix', torch.ones(()))
        self.register_buffer('hard_eval', torch.zeros((), dtype=torch.bool))
        self.register_buffer('threshold', torch.tensor(.5))
        self.noise_scale.zero_()

    def forward(self):
        noise = torch.randn_like(self.mu) * self.sigma * self.noise_scale if self.training else 0.
        soft = torch.lerp((self.mu + noise).clamp(0., 1.), torch.ones_like(self.mu), self.open_mix)
        if self.training:
            return soft
        hard = (self.deterministic() > self.threshold).to(soft.dtype)
        return torch.where(self.hard_eval, hard, soft)

    def deterministic(self):
        return torch.lerp(self.mu.detach().clamp(0., 1.), torch.ones_like(self.mu), self.open_mix)

    @torch.no_grad()
    def project_(self, optimizer=None):
        self.mu.clamp_(self.mu_min, self.mu_max)
        if optimizer is not None:
            moment = optimizer.state.get(self.mu, {}).get('exp_avg')
            if moment is not None:
                outward = ((self.mu >= self.mu_max) & (moment < 0)) | ((self.mu <= self.mu_min) & (moment > 0))
                moment.masked_fill_(outward, 0.)

    @torch.no_grad()
    def begin_epoch(self, epoch, warmup_epochs, noise_ramp_epochs, gate_optimizer=None):
        active = epoch > warmup_epochs
        if epoch == warmup_epochs + 1:
            self.mu.fill_(self.initial_mu)
            if gate_optimizer is not None:
                gate_optimizer.state.clear()
        progress = min(1., max(0., (epoch - warmup_epochs) / noise_ramp_epochs))
        self.open_mix.fill_(1. - progress)
        self.noise_scale.fill_(progress)
        self.hard_eval.zero_()
        return active

    def sparse_gradient(self, weight):
        # Exact derivative of weight * sum_j cost_j * Phi(mu_j/sigma).
        return weight * self.cost * torch.exp(-.5 * (self.mu.detach() / self.sigma).square()) / (self.sigma * math.sqrt(2.*math.pi))


class InputJointSelectionV2Fusion(InputJointSelectionFusion):
    ARCHITECTURE = 'rtmw_original_pool_bounded_stg_input_subset_v2'

    def __init__(self, num_classes=60, *, stage='select', subset=None, gate_sigma=.5,
                 gate_initial_mu=.5, gate_mu_min=-.5, gate_mu_max=1.):
        super().__init__(num_classes, stage=stage, subset=subset, gate_sigma=gate_sigma, gate_initial_mu=gate_initial_mu)
        if stage == 'select':
            self.gates = BoundedJointGates(gate_sigma, gate_initial_mu, gate_mu_min, gate_mu_max)
            self.gates.initial_mu = float(gate_initial_mu)

    def forward(self, x, valid_frame_mask=None, *, return_selection=False):
        logits, normalized_penalty = super().forward(x, valid_frame_mask, return_selection=True)
        # v2 lambda is cost per expected raw node, NOT v1's normalized penalty.
        penalty = normalized_penalty * self.gates.cost.sum() if self.stage == 'select' else normalized_penalty
        return (logits, penalty) if return_selection else logits
