import argparse

from . import infer, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--infer", action="store_true", help="Run inference")
    args = parser.parse_args()
    if args.train:
        train.main()
    elif args.infer:
        infer.main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
