### Dataset types for training

* **Type 1 (MUSDB)**: different folders. Each folder contains all needed stems in format _< stem name >.wav_. The same as in MUSDBHQ18 dataset. 

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

### Dataset for validation

* Validation dataset must be the same as Type 1 for training, but also each folder must include `mixture.wav` for each song. `mixture.wav` - it's sum of all stems for song.

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