# Execution options
.SILENT:

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