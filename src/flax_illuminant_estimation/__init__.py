import argparse
import os

import jax

from . import infer, train

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_autotune_level=2 "
    "--xla_gpu_enable_async_all_reduce=true "
    "--xla_gpu_deterministic_reductions=false "
    "--xla_gpu_enable_async_all_gather=true "
    "--xla_gpu_max_kernel_unroll=32"
)


def main():
    parser = argparse.ArgumentParser(description="Illuminant estimation with NNX")
    subparser = parser.add_subparsers(dest="command", help="command", required=True)

    train_parser = subparser.add_parser("train", help="train model")
    train_parser.add_argument("--config", help="path to config file")

    infer_parser = subparser.add_parser("infer", help="run inference")

    infer_parser.add_argument("image", help="path to input image")
    infer_parser.add_argument("--checkpoint", help="path to checkpoint (defaults to latest)")
    infer_parser.add_argument("--config", help="path to config file")

    args = parser.parse_args()

    print(f"Found devices: {jax.local_devices()[0].platform}")

    if args.command == "train":
        train.main(args)
    elif args.command == "infer":
        infer.main(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
