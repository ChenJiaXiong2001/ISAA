"""Scratch-only experimental entry; the original project entry is unchanged."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_best_face_pair_100 import training_arguments
import isaa.train as main_training
import isaa.face_pair_training as pair_training
from model import BodyFaceInteractionFusion

VARIANT = "best-face-pair-body-face-interaction"


def main():
    flags = list(sys.argv[1:])
    if not any(arg == "--save-dir" or arg.startswith("--save-dir=") for arg in flags):
        flags += ["--save-dir", "outputs/face_pair_body_face_interaction_scratch"]
    sys.argv[1:] = training_arguments(flags)
    index = sys.argv.index("--model-variant")
    sys.argv[index + 1] = VARIANT
    main_training.FACE_PAIR_VARIANT = VARIANT
    main_training.FacePairFusion = BodyFaceInteractionFusion
    pair_training.FACE_PAIR_VARIANT = VARIANT
    pair_training.FacePairFusion = BodyFaceInteractionFusion
    pair_training.COMPILE_SCOPE = dict(
        pair_training.COMPILE_SCOPE,
        scope=pair_training.COMPILE_SCOPE["scope"] + "; masked temporal body-face residual interaction",
    )
    main_training.main()


if __name__ == "__main__":
    main()
