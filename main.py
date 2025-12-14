import argparse
import sys
from utils.constants import *



def build_parser():
    parser = argparse.ArgumentParser(
        description="Unified entry point"
    )

    subparsers = parser.add_subparsers(
        dest="mode",
        required=True,
        help="Run mode",
    )

    # TRAIN

    train_parser = subparsers.add_parser(
        "train",
        help="Train a model",
    )

    train_parser.add_argument(
        "--dataset",
        type=str,
        default=CONSTANTS.DATASET_DIR,
        help="Dataset root directory",
    )

    train_parser.add_argument(
        "--classes",
        nargs="+",
        default=CONSTANTS.CLASS_LABELS,
        help="Class names",
    )

    train_parser.add_argument(
        "--epochs",
        type=int,
        default=CONSTANTS.EPOCHS,
    )

    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=CONSTANTS.BATCH_SIZE,
    )

    train_parser.add_argument(
        "--lr",
        type=float,
        default=CONSTANTS.LEARNING_RATE,
    )

    train_parser.add_argument(
        "--img-size",
        type=int,
        default=CONSTANTS.IMAGE_SIZE,
    )

    # detect
    test_parser = subparsers.add_parser(
        "detect",
        help="Run Prediction",
    )

    test_parser.add_argument(
        "--dataset",
        type=str,
        default=CONSTANTS.DATASET_DIR,
    )

    test_parser.add_argument(
        "--weights",
        type=str,
        default=CONSTANTS.CHECKPOINT_FILE,
        help="Path to trained weights",
    )

    test_parser.add_argument(
        "--batch-size",
        type=int,
        default=CONSTANTS.BATCH_SIZE,
    )

    test_parser.add_argument(
        "--img-size",
        type=int,
        default=CONSTANTS.IMAGE_SIZE,
    )



    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    CONSTANTS.recalculate_dataset_paths(args.dataset)

    if args.mode == "train":
        import modes.train
        modes.train.main()

    elif args.mode == "detect":
        import modes.detect
        modes.detect.main()

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
