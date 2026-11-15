"""Vectorized HF/BF inference fusion and calibration-only scalar helpers.

``weights[S, C]`` is a classwise HF *prior*, not a final sample gate. For
strategy s, sample n and class c, the two head contributions are proportional
to ``w[s,c] * reliability_hf[s,n] ** beta[s]`` and
``(1-w[s,c]) * reliability_bf[s,n] ** beta[s]``. Normalize these two
contributions first, form their arithmetic mixture of temperature-scaled head
probabilities, then normalize the resulting scores globally across classes.
The implementation stays in log space and returns normalized log probability.

Only calibration logits/labels may be passed to the temperature/count helpers.
They do not train a network or inspect validation labels. ``LogitFusion`` has
no parameters or stored routing state; callers can compile it once with
``torch.compile(..., backend="inductor", fullgraph=True)`` and pass batches of
strategies as tensor inputs. Compilation failures must propagate to the caller.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


CONFIDENCE_UNIFORM = 0
CONFIDENCE_MAXPROB = 1
CONFIDENCE_MARGIN = 2
CONFIDENCE_NORMALIZED_ENTROPY = 3
CONFIDENCE_NAMES = {
    CONFIDENCE_UNIFORM: "uniform",
    CONFIDENCE_MAXPROB: "maxprob",
    CONFIDENCE_MARGIN: "margin",
    CONFIDENCE_NORMALIZED_ENTROPY: "normalized_entropy",
}


class LogitFusion(nn.Module):
    """Fuse S strategies over N samples without a Python loop per strategy.

    Forward arguments, all on the same device:

    * ``hf``, ``bf``: finite floating-point logits, each ``[N, C]``;
    * ``weights``: HF class priors ``[S, C]`` in ``[0, 1]``;
    * ``temperatures``: finite positive temperatures ``[S, 2]`` (HF, BF);
    * ``confidence_beta``: finite nonnegative exponents ``[S]``;
    * ``confidence_mode``: integer mode IDs ``[S]`` from ``CONFIDENCE_NAMES``.

    Returns float32 log probabilities ``[S, N, C]``. Mode 0 or beta 0
    reproduces static class-prior fusion. Exact hard priors remain hard under
    dynamic reliability. Call ``validate_strategy_tensors`` before compiling;
    value validation is deliberately outside the captured forward graph.
    """

    @staticmethod
    def _log_reliability(log_probability, mode):
        probability = log_probability.exp()
        if log_probability.shape[-1] > 1:
            top = probability.topk(2, dim=-1).values
            margin = top[..., 0] - top[..., 1]
            # Treat 0 * log(0) as zero even if an extreme float32 logit
            # difference cannot be represented and log_softmax returns -inf.
            entropy_terms = torch.where(probability > 0,
                                        probability * log_probability,
                                        torch.zeros_like(probability))
            entropy_confidence = 1.0 + entropy_terms.sum(-1) / math.log(log_probability.shape[-1])
        else:
            margin = probability[..., 0]
            entropy_confidence = torch.ones_like(margin)
        maxprob = probability.amax(-1)
        selected = torch.where(mode[:, None] == CONFIDENCE_MAXPROB, maxprob,
                               torch.where(mode[:, None] == CONFIDENCE_MARGIN, margin,
                                           torch.where(mode[:, None] == CONFIDENCE_NORMALIZED_ENTROPY,
                                                       entropy_confidence, torch.ones_like(maxprob))))
        return selected.clamp(min=1e-6, max=1.0).log()

    def forward(self, hf, bf, weights, temperatures, confidence_beta, confidence_mode):
        hf_log_probability = F.log_softmax(hf.float()[None] / temperatures[:, 0, None, None].float(), dim=-1)
        bf_log_probability = F.log_softmax(bf.float()[None] / temperatures[:, 1, None, None].float(), dim=-1)
        log_hf_reliability = self._log_reliability(hf_log_probability, confidence_mode)
        log_bf_reliability = self._log_reliability(bf_log_probability, confidence_mode)
        prior = weights.float()[:, None]
        exponent = confidence_beta.float()[:, None, None]
        hf_log_contribution = prior.log() + exponent * log_hf_reliability[..., None]
        bf_log_contribution = torch.log1p(-prior) + exponent * log_bf_reliability[..., None]
        contribution_normalizer = torch.logaddexp(hf_log_contribution, bf_log_contribution)
        score = torch.logaddexp(hf_log_probability + hf_log_contribution - contribution_normalizer,
                               bf_log_probability + bf_log_contribution - contribution_normalizer)
        score = score - score.amax(-1, keepdim=True)
        return score - torch.logsumexp(score, dim=-1, keepdim=True)


def validate_strategy_tensors(weights, temperatures, confidence_beta, confidence_mode):
    """Validate candidate tensors outside compiled inference; return None."""
    if weights.ndim != 2 or not weights.shape[0] or not weights.shape[1]:
        raise ValueError("weights must have nonempty S x C shape")
    strategies = weights.shape[0]
    if temperatures.shape != (strategies, 2):
        raise ValueError("temperatures must have S x 2 shape")
    if confidence_beta.shape != (strategies,) or confidence_mode.shape != (strategies,):
        raise ValueError("confidence_beta and confidence_mode must each have S shape")
    if len({tensor.device for tensor in (weights, temperatures, confidence_beta, confidence_mode)}) != 1:
        raise ValueError("All strategy tensors must use the same device")
    if not bool(torch.isfinite(weights).all()) or bool(((weights < 0) | (weights > 1)).any()):
        raise ValueError("Class weights must be finite and in [0, 1]")
    if not bool(torch.isfinite(temperatures).all()) or bool((temperatures <= 0).any()):
        raise ValueError("Head temperatures must be finite and positive")
    if not bool(torch.isfinite(confidence_beta).all()) or bool((confidence_beta < 0).any()):
        raise ValueError("Confidence exponents must be finite and nonnegative")
    if confidence_mode.is_floating_point() or confidence_mode.is_complex() or confidence_mode.dtype == torch.bool:
        raise ValueError("Confidence modes must be integer IDs")
    if bool(((confidence_mode < CONFIDENCE_UNIFORM) | (confidence_mode > CONFIDENCE_NORMALIZED_ENTROPY)).any()):
        raise ValueError("Unknown confidence mode")


def class_weights_from_counts(hf_correct, bf_correct, class_counts, gammas):
    """Return CPU float32 ``[S, C]`` priors from calibration accuracy powers.

    For gamma > 0, ``w = accuracy_hf**gamma / (accuracy_hf**gamma +
    accuracy_bf**gamma)``. The common per-class support cancels in this ratio.
    Gamma 0 gives equal priors for every class; gamma +inf gives hard accuracy
    selection, with ties averaging. Both-zero and unseen classes always use .5.
    Log odds avoid underflow when large finite powers would become zero.
    """
    counts = [torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64)
              for value in (hf_correct, bf_correct, class_counts)]
    hf, bf, total = counts
    if hf.ndim != 1 or not hf.numel() or bf.shape != hf.shape or total.shape != hf.shape:
        raise ValueError("Correct counts and class counts must have the same nonempty C shape")
    for value in counts:
        if (not bool(torch.isfinite(value).all()) or bool((value < 0).any())
                or bool((value != value.floor()).any())):
            raise ValueError("Calibration counts must be finite nonnegative integers")
    if bool((hf > total).any()) or bool((bf > total).any()):
        raise ValueError("Correct counts cannot exceed class support")
    gamma = torch.as_tensor(gammas).detach().to(device="cpu", dtype=torch.float64)
    if gamma.ndim > 1 or not gamma.numel():
        raise ValueError("gammas must be a nonempty scalar or S vector")
    gamma = gamma.reshape(-1, 1)
    if bool(torch.isnan(gamma).any()) or bool((gamma < 0).any()):
        raise ValueError("Gamma must be nonnegative, allowing +inf for hard selection")
    finite_gamma = torch.where(torch.isinf(gamma), torch.zeros_like(gamma), gamma)
    log_odds = hf.clamp_min(1).log() - bf.clamp_min(1).log()
    weight = torch.sigmoid(finite_gamma * log_odds[None])
    weight = torch.where(hf[None] == 0, torch.zeros_like(weight), weight)
    weight = torch.where(bf[None] == 0, torch.ones_like(weight), weight)
    hard = torch.where(hf > bf, 1., torch.where(hf < bf, 0., .5))
    weight = torch.where(torch.isinf(gamma), hard[None], weight)
    weight = torch.where((gamma == 0) | ((hf + bf)[None] == 0), .5, weight)
    return weight.float()


def _prepare_calibration(logits, labels):
    values = torch.as_tensor(logits).detach().to(device="cpu", dtype=torch.float64).contiguous()
    targets = torch.as_tensor(labels).detach().to(device="cpu")
    if (values.ndim != 2 or not values.shape[0] or not values.shape[1]
            or targets.shape != (values.shape[0],)):
        raise ValueError("Expected nonempty calibration logits N x C and labels N")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Calibration logits must be finite")
    if targets.is_complex() or targets.dtype == torch.bool or not bool(torch.isfinite(targets).all()):
        raise ValueError("Calibration labels must be finite integer class indices")
    if targets.is_floating_point() and bool((targets != targets.floor()).any()):
        raise ValueError("Calibration labels must be integer class indices")
    targets = targets.long().contiguous()
    if bool((targets < 0).any()) or bool((targets >= values.shape[1]).any()):
        raise ValueError("Calibration labels are outside the class range")
    return values, targets


def _temperature_nll(values, targets, temperature):
    return float(F.cross_entropy(values / temperature, targets))


def temperature_nll(logits, labels, temperature):
    """Evaluate mean calibration NLL on CPU float64 at one positive T."""
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    values, targets = _prepare_calibration(logits, labels)
    return _temperature_nll(values, targets, temperature)


def fit_temperature(logits, labels, *, minimum=.1, maximum=10., grid_size=81, iterations=64):
    """Deterministically minimize calibration NLL over one head's scalar T.

    Search a uniform grid in log(T), then refine the best adjacent interval
    with gradient-free golden-section search, including both boundaries and
    T=1 in the final candidates when allowed. All arithmetic is CPU float64.
    Returns a float, so callers can separately report bounds and NLL before /
    after using ``temperature_nll``. No validation labels may be used here.
    """
    minimum, maximum = float(minimum), float(maximum)
    if (not math.isfinite(minimum) or not math.isfinite(maximum)
            or not 0 < minimum < maximum or grid_size < 3 or iterations < 1):
        raise ValueError("Require 0 < minimum < maximum, grid_size >= 3 and iterations >= 1")
    values, targets = _prepare_calibration(logits, labels)
    log_minimum, log_maximum = math.log(minimum), math.log(maximum)
    grid = [log_minimum + (log_maximum - log_minimum) * i / (grid_size - 1)
            for i in range(grid_size)]
    candidates = []

    def evaluate(log_temperature):
        temperature = min(maximum, max(minimum, math.exp(log_temperature)))
        nll = _temperature_nll(values, targets, temperature)
        candidates.append((nll, abs(log_temperature), temperature))
        return nll

    grid_values = [evaluate(point) for point in grid]
    best = min(range(grid_size), key=lambda index: (grid_values[index], abs(grid[index])))
    left, right = grid[max(0, best - 1)], grid[min(grid_size - 1, best + 1)]
    ratio = (math.sqrt(5.) - 1.) / 2.
    lower, upper = right - ratio * (right - left), left + ratio * (right - left)
    lower_value, upper_value = evaluate(lower), evaluate(upper)
    for _ in range(iterations):
        if lower_value <= upper_value:
            right, upper, upper_value = upper, lower, lower_value
            lower = right - ratio * (right - left)
            lower_value = evaluate(lower)
        else:
            left, lower, lower_value = lower, upper, upper_value
            upper = left + ratio * (right - left)
            upper_value = evaluate(upper)
    if minimum <= 1. <= maximum:
        evaluate(0.)
    # Exact endpoints make boundary-hit reporting independent of exp(log(T))
    # roundoff. The tie-break prefers a temperature closest to the baseline 1.
    for temperature in (minimum, maximum):
        candidates.append((_temperature_nll(values, targets, temperature), abs(math.log(temperature)), temperature))
    return min(candidates)[2]
