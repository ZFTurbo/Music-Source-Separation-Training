"""
    Script to list layers of model for possible freeze
"""

import argparse
from utils.settings import get_model_from_config


def parse_args() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, type=str, default='mdx23c',
                        help="One of mdx23c, htdemucs, segm_models, mel_band_roformer, bs_roformer, swin_upernet, bandit, etc")
    parser.add_argument("--config_path", required=True, type=str, help="path to config file")
    parser.add_argument("--output_file", type=str, default="layers.txt", help="path to results file with list of model layers")
    parser.add_argument("--layer_mask", nargs="+", type=str, help="mask to print layer names containing mask. Can be several masks")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model, config = get_model_from_config(args.model_type, args.config_path)
    for name, module in model.named_modules():
        if args.layer_mask is not None:
            for mask in args.layer_mask:
                if mask in name:
                    print(name)
        else:
            print(name)


if __name__ == '__main__':
    main()