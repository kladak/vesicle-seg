.PHONY: install test train train-ci demo generate infer serve

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest -q

generate:
	python -m vesicle_seg.cli generate --config configs/ci.yaml

train:
	python -m vesicle_seg.cli train --config configs/default.yaml

train-ci:
	python -m vesicle_seg.cli train --config configs/ci.yaml

demo:
	python -m vesicle_seg.cli demo --config configs/ci.yaml

infer:
	python -m vesicle_seg.cli infer --config configs/ci.yaml --volume-id syn_000

serve:
	python -m vesicle_seg.cli serve --port 8765
