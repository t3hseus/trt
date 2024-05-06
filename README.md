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

### Model training
```bash
make train CUDA_VISIBLE_DEVICES=DEVICE_ID
```

To check Tensorboard logs:
```bash
tensorboard --logdir experiment_logs
```

