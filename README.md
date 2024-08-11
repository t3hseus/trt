# TRT: Track Reconstruction Transformer

This repo contains the code of transformer architecture for solving particle track reconstruction problem in high energy physics. The main model is based on [Point Cloud Transformer (PCT)](https://link.springer.com/content/pdf/10.1007/s41095-021-0229-5.pdf) modified to be able to predict tracks parameters directly from input data. 


## Setup Environment

To setup virtual environment you need to download and install Python 3.9, then to create environment run the following `make` command:

```bash
make venv
```

To activate created venv use the following command:

```bash
source .venv/bin/activate
```

If you are using VSCode, you can select the newly installed venv with the help of command palette:
1. `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type `Python: Select Interpreter`
3. Choose the option with `.venv/bin/python` path

### Configuration
1. Paste your paths to `/config/user_settings/user_setting.yaml`
2. If you want to change run configuration, change settings in `train.yaml` or yaml files in corresponding directories

Hydra configuration is very powerful.
It allows to change configuration with distinct directories for each block. 
Also hydra allows links in configuration using `${config field}`.
For example, if you want to change model, go to the "model" directory and add new file or change the existing one.
To change the experiment, the easiest way is to change setting in default section of main config.
Example: 
```yaml
defaults:
  - model: base
  - criterion: naive
  - user_settings: user_settings
  - dataset: base
```
This config uses settings for 
model from file `model/base.yaml` and criterion from `criterion/naive.yaml`

You can also control process with experiments and overrides. More information in 
[documentation for override](https://hydra.cc/docs/advanced/override_grammar/basic/) and
[configuring experiments](https://hydra.cc/docs/patterns/configuring_experiments/)

The best part of hydra, IMHO, is instantiate from configs. All you need is to set up 
`_target_: YourClass` and `__init__` arguments in config and run
`instantiate(cfg.target_config)` in your code:
```yaml
_target_: src.model.TRT
dropout: 0.1
n_points: ${dataset.truncation_length}
num_candidates: 20
num_out_params: 6
```
More info in [instantiation documentation](https://hydra.cc/docs/advanced/instantiate_objects/overview/)

Hydra saves current config, log and main script copy into `hydra.run` directory. Hence, it's very easy to set setup
meaningful directories for tensorboard traces and other files. 


### Model training
```bash
make train CUDA_VISIBLE_DEVICES=DEVICE_ID
```

To check Tensorboard logs:
```bash
tensorboard --logdir experiment_logs
```

