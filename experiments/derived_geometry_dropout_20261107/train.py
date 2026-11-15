"""Scratch entry point for the derived-geometry dropout screen."""
from __future__ import annotations

import sys

from train_best_face_pair_100 import training_arguments as _base_arguments
from isaa import face_pair_training
from experiments.derived_geometry_dropout_20261107.model import DerivedGeometryDropoutFacePairFusion


def training_arguments(overrides):
    supplied = {value.split("=", 1)[0] for value in overrides if value.startswith("--")}
    if "--no-compile" in supplied:
        raise ValueError("derived-geometry dropout requires torch.compile; --no-compile is not allowed")
    if supplied.intersection({"--init-checkpoint", "--resume-checkpoint", "--model-variant", "--npy-dir"}):
        raise ValueError("derived-geometry dropout is a scratch-only candidate")
    arguments = _base_arguments(list(overrides))
    arguments[arguments.index("--model-variant") + 1] = "best-face-pair"
    if "--save-dir" not in supplied:
        arguments[arguments.index("--save-dir") + 1] = "outputs/derived_geometry_dropout_20261107"
    return arguments


def main():
    face_pair_training.FacePairFusion = DerivedGeometryDropoutFacePairFusion
    original_compile_scope = face_pair_training.compile_scope

    def dropout_compile_scope(args):
        scope = original_compile_scope(args)
        scope["scope"] += "; training-only derived distance/direction dropout in hand path"
        scope["derived_geometry_dropout"] = {
            "probability": DerivedGeometryDropoutFacePairFusion.DERIVED_GEOMETRY_DROPOUT_P,
            "training_only": True,
        }
        return scope

    face_pair_training.compile_scope = dropout_compile_scope
    sys.argv[1:] = training_arguments(sys.argv[1:])
    from isaa.train import main as train_main
    train_main()


if __name__ == "__main__":
    main()
