PYTHON ?= python

.PHONY: test demo
test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -B -m unittest discover -s tests

demo:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -B examples/feature_scaling/evaluate.py
