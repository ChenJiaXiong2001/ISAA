"""Conditional binary gates with matched costs and persistent exploration."""
import math
import torch
from torch import nn
from isaa.models.input_joint_selection_v4 import InputJointSelectionV4Fusion

INPUT_SELECTION_V5_VARIANT = 'best-input-joint-selection-v5-dynamic'


class ConditionalJointGates(nn.Module):
    def __init__(self, *, temperature=1., exploration=.05, logit_bound=4.):
        super().__init__()
        if not all(math.isfinite(v) for v in (temperature,exploration,logit_bound)):
            raise ValueError('Nonfinite gate settings')
        if temperature<=0 or not 0<exploration<.5 or not 0<logit_bound<=8:
            raise ValueError('Invalid temperature, exploration, or logit bound')
        self.temperature=float(temperature)
        self.exploration=float(exploration)
        self.logit_bound=float(logit_bound)
        self.descriptor_norm=nn.LayerNorm(8)
        self.context_norm=nn.LayerNorm(8)
        self.node_embedding=nn.Parameter(torch.empty(40,8))
        nn.init.normal_(self.node_embedding,std=.02)
        self.router=nn.Sequential(nn.Linear(24,32),nn.ReLU(),nn.Linear(32,1))
        nn.init.normal_(self.router[-1].weight,std=.01)
        nn.init.constant_(self.router[-1].bias,2.)
        self.register_buffer('cost',torch.tensor([1.]*14+[2.]*20+[1.]*6))
        self.register_buffer('open_mix',torch.ones(()))
        self.register_buffer('noise_scale',torch.zeros(()))
        self.register_buffer('hard_train',torch.zeros((),dtype=torch.bool))
        self.register_buffer('hard_eval',torch.zeros((),dtype=torch.bool))
        self.register_buffer('threshold',torch.tensor(.5))
        # Diagnostic only: 0=learned, 1=all open, 2=all optional closed.
        self.register_buffer('forced_mode',torch.zeros((),dtype=torch.long))

    def begin_epoch(self,epoch,warmup_epochs,learning_epochs=None):
        active=epoch>warmup_epochs
        self.open_mix.fill_(0. if active else 1.)
        self.noise_scale.fill_(1. if active else 0.)
        self.hard_train.fill_(active)
        self.hard_eval.zero_()
        self.forced_mode.zero_()

    def forward(self,descriptors,global_context):
        descriptors=self.descriptor_norm(descriptors)
        context=self.context_norm(global_context)[:,None,:].expand_as(descriptors)
        embedding=self.node_embedding[None].expand(descriptors.shape[0],-1,-1)
        raw=self.router(torch.cat((descriptors,context,embedding),-1)).squeeze(-1)
        # Bounded forward with nonzero surrogate derivative, unlike tanh or
        # an ordinary clamp whose saturated backward cannot recover.
        clipped=raw.clamp(-self.logit_bound,self.logit_bound)
        logits=raw+(clipped-raw).detach()
        probabilities=self.exploration+(1.-2.*self.exploration)*torch.sigmoid(logits/self.temperature)
        frozen=self.open_mix>0
        probabilities=torch.where(frozen,torch.ones_like(probabilities),probabilities)
        probabilities=torch.where(self.forced_mode==1,torch.ones_like(probabilities),probabilities)
        probabilities=torch.where(self.forced_mode==2,torch.zeros_like(probabilities),probabilities)
        if self.training:
            # Bernoulli(p) forward and direct probability surrogate backward.
            # The exact same p is charged in the expected cost below.
            sampled=(torch.rand_like(probabilities)<probabilities).to(raw.dtype)
            gates=sampled-probabilities.detach()+probabilities
        else:
            gates=torch.where(self.hard_eval,(probabilities>self.threshold).to(raw.dtype),probabilities)
        expected=(probabilities*self.cost).sum(-1)
        overflow=(raw-clipped.detach()).square().mean()
        overflow=torch.where(frozen,overflow*0.,overflow)
        return gates,expected.mean()/self.cost.sum(),dict(
            gate_values=probabilities,sampled_gate_values=gates,
            expected_optional_counts=expected,gate_logits=logits,router_raw_logits=raw,
            logit_guard_loss=overflow)


class InputJointSelectionV5Fusion(InputJointSelectionV4Fusion):
    ARCHITECTURE='rtmw_original_pool_conditional_binary_input_subset_v5'

    def __init__(self,num_classes=60,*,gate_temperature=1.,gate_exploration=.05,gate_logit_bound=4.):
        super().__init__(num_classes,gate_temperature=gate_temperature)
        self.gates=ConditionalJointGates(temperature=gate_temperature,
            exploration=gate_exploration,logit_bound=gate_logit_bound)

    @staticmethod
    def _pool_gated_nodes(x,observed,gates):
        # Fixed observed denominator: lowering a soft gate must not be
        # cancelled by simultaneously lowering the pooling denominator.
        weights=observed.to(x.dtype)
        count=weights.sum(-1).squeeze(1)
        features=(x*weights).sum(-1).transpose(1,2)
        return features/count.unsqueeze(-1).clamp_min(1.),count>0
