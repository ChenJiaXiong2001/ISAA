"""Scratch entry point for the pre-registered temporal HF interaction screen.

The command deliberately reuses the audited face-pair training loop while
substituting the isolated interaction subclass.  The public variant remains
``best-face-pair`` so the shared parser and data protocol cannot drift.
"""
from __future__ import annotations

import sys

from train_best_face_pair_100 import training_arguments as _base_arguments
from isaa import face_pair_training
from experiments.hand_face_interaction_20261107.model import HandFaceTemporalInteractionFusion


def training_arguments(overrides):
    supplied = {value.split("=", 1)[0] for value in overrides if value.startswith("--")}
    if "--no-compile" in supplied:
        raise ValueError("hand-face temporal interaction requires torch.compile; --no-compile is not allowed")
    if supplied.intersection({"--init-checkpoint", "--resume-checkpoint", "--model-variant", "--npy-dir"}):
        raise ValueError("hand-face temporal interaction is a scratch-only candidate")
    arguments = _base_arguments(list(overrides))
    # The audited loop dispatches this string to the monkey-patched class.
    arguments[arguments.index("--model-variant") + 1] = "best-face-pair"
    if "--save-dir" not in supplied:
        arguments[arguments.index("--save-dir") + 1] = (
            "outputs/hand_face_temporal_interaction_20261107")
    return arguments


def main():
    # run_face_pair_training resolves FacePairFusion from its module globals;
    # this substitution is confined to this process and leaves all historical
    # entry points untouched.
    face_pair_training.FacePairFusion = HandFaceTemporalInteractionFusion
    original_compile_scope = face_pair_training.compile_scope

    def interaction_compile_scope(args):
        scope = original_compile_scope(args)
        scope["scope"] += "; temporal hand-face projections, elementwise interaction, masked pooling and HF residual readout"
        scope["interaction"] = "new residual path is inside torch.compile/Inductor scope"
        return scope

    face_pair_training.compile_scope = interaction_compile_scope
    sys.argv[1:] = training_arguments(sys.argv[1:])
    from isaa.train import main as train_main
    train_main()


if __name__ == "__main__":
    main()

