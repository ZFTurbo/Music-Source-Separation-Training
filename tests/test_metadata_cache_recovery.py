from utils.dataset import MSSDataset


def test_empty_metadata_cache_is_rebuilt(tmp_path):
    cache = tmp_path / "metadata.pkl"
    cache.touch()
    dataset = MSSDataset.__new__(MSSDataset)
    dataset.metadata_path = str(cache)
    dataset.verbose = False
    paths = ["track"]
    remaining, metadata = dataset.read_from_metadata_cache(paths)
    assert remaining == paths
    assert metadata == []


def test_concurrent_reader_never_sees_partial_pickle(tmp_path, monkeypatch):
    import pickle
    import threading

    import numpy as np
    import soundfile as sf
    from ml_collections import ConfigDict
    from utils import dataset as dataset_module

    track = tmp_path / "data" / "song"
    track.mkdir(parents=True)
    for name in ("vocals", "other"):
        sf.write(track / f"{name}.wav", np.zeros((512, 2), dtype="float32"), 8000)
    config = ConfigDict({
        "training": {"instruments": ["vocals", "other"], "batch_size": 1,
                     "num_steps": 1, "read_metadata_procs": 1},
        "audio": {"chunk_size": 256, "min_mean_abs": 0},
        "augmentations": {"enable": False},
    })
    cache = tmp_path / "metadata.pkl"
    previous = [(str(track), 512)]
    cache.write_bytes(pickle.dumps(previous))
    first_half_written = threading.Event()
    finish_writing = threading.Event()

    def partial_dump(obj, file):
        payload = pickle.dumps(obj)
        middle = len(payload) // 2
        file.write(payload[:middle])
        file.flush()
        first_half_written.set()
        assert finish_writing.wait(timeout=10)
        file.write(payload[middle:])

    monkeypatch.setattr(dataset_module.pickle, "dump", partial_dump)
    errors = []

    def build_metadata():
        try:
            MSSDataset(config, [str(track.parent)], str(cache), dataset_type=1, verbose=False)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=build_metadata)
    worker.start()
    try:
        assert first_half_written.wait(timeout=10)
        assert pickle.loads(cache.read_bytes()) == previous
    finally:
        finish_writing.set()
        worker.join(timeout=10)
    assert not worker.is_alive()
    assert not errors
    assert [tuple(row) for row in pickle.loads(cache.read_bytes())] == previous
