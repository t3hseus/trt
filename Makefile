# Execution options
.SILENT:

CUDA_VISIBLE_DEVICES = 
PYTHON_VERSION = 3.9
VENV := .venv
REQPATH := requirements.txt


#################### Environment functions ####################

${VENV}:
	rm -rf ${VENV} || true
	python${PYTHON_VERSION} -m venv ${VENV}
	${VENV}/bin/pip install pip -U


.PHONY: venv
venv: ${VENV}
	${VENV}/bin/pip install -r ${REQPATH}


.PHONY: train
train:
	CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python train.py