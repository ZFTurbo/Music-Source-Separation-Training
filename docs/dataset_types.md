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
....
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
...
```

* **Type 3 (CSV file)**:

You can provide CSV-file (or list of CSV-files) with following structure:
```
type,path
vocals,/path/to/dataset/vocals_1.wav
vocals,/path/to/dataset2/vocals_v2.wav
vocals,/path/to/dataset3/vocals_some.wav
...
drums,/path/to/dataset/drums_good.wav
...
```

### Dataset for validation

* Validation dataset must be the same as Type 1 for training.