# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import os
import torch
import argparse


def clean_weights(args):
    weights = torch.load(args.checkpoint, map_location='cpu')
    print('Keys: {}'.format(list(weights.keys())))
    if 'model_state_dict' in list(weights.keys()):
        weights = weights['model_state_dict']
    if args.float16:
        for el in weights:
            weights[el] = weights[el].to(torch.float16)
    torch.save(weights, args.output_file)


def parse_args(dict_args):
    parser = argparse.ArgumentParser(
        description="Clean all except weights from checkpoint file. Optionally converts to float16.")
    parser.add_argument('--checkpoint', type=str, help="Input checkpoint to clean")
    parser.add_argument('--output_file', type=str, help="File to save cleaned checkpoint")
    parser.add_argument('--float16', action='store_true',
                        help="Convert weights to float16 instead of float32. Reduce weights size two times.")

    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
        print(args)
    else:
        args = parser.parse_args()
    return args


def main(dict_args):
    args = parse_args(dict_args)
    clean_weights(args)


if __name__ == "__main__":

    name_stem = 'percussion'

    dict_args = {
        'checkpoint': rf'E:\trash\weights\{name_stem}\model_bs_roformer_ep_7_sdr_6.2084.ckpt',
        'output_file': fr'checkpoints\model_bs_roformer_{name_stem}_sdr_6.2084.ckpt',
        'float16': True,
    }

    main(dict_args)
