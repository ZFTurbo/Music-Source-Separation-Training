# Automatic single-GPU batch calibration

Add `--do_max_batch` to the normal training command. Calibration performs real optimizer steps and validation, restores weights, EMA and random state, then trains with the selected physical batch. The report is `RESULTS/auto_batch.json`; the source YAML is unchanged.

Defaults: `--max_batch_limit 64`, `--max_batch_memory_fraction 0.90`, `--max_batch_probe_steps 3`. The search stops at the first rejected batch; reaching the cap does not establish the hardware maximum. The probe uses the longest validation track and, when needed, additional tracks to cover every requested metric and target. It is not an exhaustive guarantee for every example or future memory pressure. With variable-length chunk augmentation, the training probe uses the longest permitted chunk. It requires two successful optimizer updates, retrying transient AMP overflow up to a bounded limit; EMA is updated only after successful steps.

Optional `--max_batch_chunk_reduction 0.02` allows up to 2% shorter windows for BS/Mel-Band RoFormer. Mel windows without explicit output-length matching stay aligned to the STFT hop. Reduction is disabled by default. DDP, DataParallel and CPU calibration are unsupported and fail explicitly.

For Slurm, request one GPU and run the normal command with `--device_ids 0 --do_max_batch`. Before full training, run the GPU tests and a short real-data training/validation smoke with the exact configuration, checkpoint, optimizer and resume flags. Keep the JSON report with each run. Do not modify an already running experiment to retrofit calibration.

Validation on 2026-11-06: Adam, AdamW, AMP, EMA/RNG restoration, validation OOM fallback, BS/Mel-Band RoFormer chunk reduction, and training through checkpoint saving. Compact fixtures are used; full production configuration compatibility requires its own smoke.
