import glob
import os
from typing import Tuple

import numpy as np
import torchaudio as ta
from tqdm.contrib.concurrent import process_map


def process_one(inputs: Tuple[str, str, int]) -> None:
    infile, outfile, target_fs = inputs

    dir = os.path.dirname(outfile)
    os.makedirs(dir, exist_ok=True)

    data, fs = ta.load(infile)

    if fs != target_fs:
        data = ta.functional.resample(data, fs, target_fs, resampling_method="sinc_interp_kaiser")
        fs = target_fs

    data = data.numpy()
    data = data.astype(np.float32)

    if os.path.exists(outfile):
        data_ = np.load(outfile)
        if np.allclose(data, data_):
            return

    np.save(outfile, data)


def preprocess(
        data_path: str,
        output_path: str,
        fs: int
) -> None:
    files = glob.glob(os.path.join(data_path, "**", "*.wav"), recursive=True)
    print(files)
    outfiles = [
            f.replace(data_path, output_path).replace(".wav", ".npy") for f in
            files
    ]

    os.makedirs(output_path, exist_ok=True)
    inputs = list(zip(files, outfiles, [fs] * len(files)))

    process_map(process_one, inputs, chunksize=32)


if __name__ == "__main__":
    import fire

    fire.Fire()
