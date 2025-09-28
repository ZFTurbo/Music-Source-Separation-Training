### Dataset types for training

* **Type 1 (MUSDB)**: different folders. Each folder contains all needed stems in format _< stem name >.wav_. The same as in MUSDBHQ18 dataset. In latest code releases it's possible to use `flac` instead of `wav`. 

Example:
```
--- Song 1:
------ vocals.wav  
------ bass.wav 
------ drums.wav
------ other.wav
--- Song 2:
------ vocals.wav  
------ bass.wav 
------ drums.wav
------ other.wav
--- Song 3:
...........
```

* **Type 2 (Stems)**: each folder is "stem name". Folder contains wav files which consists only of required stem.
```
--- vocals:
------ vocals_1.wav
------ vocals_2.wav
------ vocals_3.wav
------ vocals_4.wav
------ ...
--- bass:
------ bass_1.wav
------ bass_2.wav
------ bass_3.wav
------ bass_4.wav
------ ...
...........
```

* **Type 3 (CSV file)**:

You can provide CSV-file (or list of CSV-files) with following structure:
```
instrum,path
vocals,/path/to/dataset/vocals_1.wav
vocals,/path/to/dataset2/vocals_v2.wav
vocals,/path/to/dataset3/vocals_some.wav
...
drums,/path/to/dataset/drums_good.wav
...
```

* **Type 4 (MUSDB Aligned)**:

The same as Type 1, but during training all instruments will be from the same position of song. 

* **Type 5 (Precomputed Chunks)**:

The same structure as Type 1, but all tracks are pre-split into chunks with 50% overlap (each second appears in two chunks except edges). This ensures that during one epoch the model sees every part of every track exactly once. Recommended for small datasets or when you want deterministic epoch boundaries.

Key features:

* Structure: Same as Type 1 (folder per song with stem files)

* Chunking: Automatic splitting into chunks of config.audio.chunk_size

* Overlap: 50% overlap between consecutive chunks

* Epoch completeness: Each epoch covers 100% of available audio data

* No random sampling: Deterministic access pattern

* Memory efficient: Chunks are computed on-the-fly, not stored in memory

Usage recommendations:

* Use when you want reproducible training cycles

* Good for small to medium datasets

* Ensures no data is missed during training


### Dataset for validation

* The validation dataset must be the same structure as type 1 datasets (regardless of what type of dataset you're using for training), but also each folder must include `mixture.wav` for each song. `mixture.wav` - is the sum of all stems for song.

Example:
```
--- Song 1:
------ vocals.wav  
------ bass.wav 
------ drums.wav
------ other.wav
------ mixture.wav
--- Song 2:
------ vocals.wav  
------ bass.wav 
------ drums.wav
------ other.wav
------ mixture.wav
--- Song 3:
...........
```
