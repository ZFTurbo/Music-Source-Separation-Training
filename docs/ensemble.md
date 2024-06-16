### Ensemble usage

Repository contains `ensemble.py` script which can be used to ensemble results of different algorithms.

Arguments:
* `--files` - Path to all audio-files to ensemble
* `--type` - Method to do ensemble. One of avg_wave, median_wave, min_wave, max_wave, avg_fft, median_fft, min_fft, max_fft. Default: avg_wave.
* `--weights` - Weights to create ensemble. Number of weights must be equal to number of files
* `--output` - Path to wav file where ensemble result will be stored (Default: res.wav)

Example:
```
ensemble.py --files ./results_tracks/vocals1.wav ./results_tracks/vocals2.wav --weights 2 1 --type max_fft --output out.wav
```

### Ensemble types:

* `avg_wave` - ensemble on 1D variant, find average for every sample of waveform independently
* `median_wave` - ensemble on 1D variant, find median value for every sample of waveform independently
* `min_wave` - ensemble on 1D variant, find minimum absolute value for every sample of waveform independently
* `max_wave` - ensemble on 1D variant, find maximum absolute value for every sample of waveform independently
* `avg_fft` - ensemble on spectrogram (Short-time Fourier transform (STFT), 2D variant), find average for every pixel of spectrogram independently. After averaging use inverse STFT to obtain original 1D-waveform back.
* `median_fft` - the same as avg_fft but use median instead of mean (only useful for ensembling of 3 or more sources).
* `min_fft` - the same as avg_fft but use minimum function instead of mean (reduce aggressiveness).
* `max_fft` - the same as avg_fft but use maximum function instead of mean (the most aggressive).

### Notes
* `min_fft` can be used to do more conservative ensemble - it will reduce influence of more aggressive models.
* It's better to ensemble models which are of equal quality - in this case it will give gain. If one of model is bad - it will reduce overall quality.
* In my experiments `avg_wave` was always better or equal in SDR score comparing with other methods.
