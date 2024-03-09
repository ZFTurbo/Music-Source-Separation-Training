### Changes 

#### v1.0.2

* Added multi GPU validation (earlier validation was performed on single GPU)
* `training.batch_size` in config now must be set for single GPU (if you use multiple GPUs it will be automatically multiplied by number of GPUs)