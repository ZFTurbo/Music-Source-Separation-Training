 Dataset types for training

### Type 1 (MUSDB)

Different folders. Each folder contains all needed stems in format  
`<stem_name>.wav` (or `flac` in latest releases).

The structure is the same as in MUSDBHQ18.

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

### Type 2 (Stems)

Each folder represents a single stem name.  
The folder contains audio files consisting only of that stem.

Example:
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

### Type 3 (CSV file)

You can provide a CSV file (or a list of CSV files) with the following structure:

```
instrum,path
vocals,/path/to/dataset/vocals_1.wav
vocals,/path/to/dataset2/vocals_v2.wav
vocals,/path/to/dataset3/vocals_some.wav
...
drums,/path/to/dataset/drums_good.wav
...
```

### Type 4 (MUSDB Aligned)

The same structure as Type 1, but during training all instruments are loaded from the same position of the song.

### Type 5 (Precomputed Chunks)

The same structure as Type 1, but all tracks are pre-split into chunks with 50% overlap.

### Type 6 (MUSDB Aligned + Explicit Mixture)

An extension of Type 4, designed for scenarios where the **mixture is treated as a separate signal**, rather than always being reconstructed as `sum(stems)`.

Structure is the same as Type 1 / Type 4, but it is recommended that each song folder contains `mixture.wav`.

Example:
```
--- Song 1:
------ vocals.wav
------ bass.wav
------ drums.wav
------ other.wav
------ mixture.wav
```

Key properties:
- All stems are loaded aligned from the same song position
- `mixture.wav` is loaded explicitly if present
- If `mixture.wav` is missing, mixture is computed as sum of stems
- Supports precomputed random chunks (same logic as Type 4)

Typical use cases:
- Teacher–student or distillation training
- Consistency losses
- Training with real mixes not equal to sum of stems


### Type 7 (Class-Balanced Aligned Dataset)

A class-balanced aligned dataset designed to reduce class frequency bias and improve learning of rare instruments.

Structure is the same as Type 6:
```
--- Song 1:
------ flute.wav
------ violin.wav
------ mixture.wav
```

How it works:
1. A random instrument (class) is selected
2. A random track containing this instrument is chosen
3. An aligned chunk is loaded from that track
4. The dataset returns which stems are actually present

Class frequency filtering:
- For each instrument, the ratio of tracks where it appears is computed
- Instruments appearing in more than `max_class_presence_ratio`
  (default: 0.4) of tracks are excluded
- Prevents dominant classes (e.g. vocals) from overwhelming training

Returned values:
- Stems tensor
- Mixture tensor (from `mixture.wav` if available, otherwise sum of stems)
- `active_stem_ids`: indices of instruments present in the current sample

Typical use cases:
- Training on sparse multi-instrument datasets
- Improving performance on rare instruments
- Conditional or multi-head source separation models

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
