import argparse

from . import infer, train


def main():
    parser = argparse.ArgumentParser(description="Illuminant estimation with NNX")
    subparser = parser.add_subparsers(dest="command", help="command", required=True)

    train_parser = subparser.add_parser("train", help="train model")
    train_parser.add_argument("--config", help="path to config file")
    train_parser.add_argument(
        "--resume",
        help="path to checkpoint directory to resume from",
    )

    infer_parser = subparser.add_parser("infer", help="<path_to_img> <checkpoint>")

    infer_parser.add_argument("image", help="path to input image")
    infer_parser.add_argument(
        "--checkpoint",
        default="checkpoints/checkpoint_010",
        help="path to checkpoint directory",
    )

    args = parser.parse_args()
    if args.command == "train":
        train.main(args)
    elif args.command == "infer":
        infer.main(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
