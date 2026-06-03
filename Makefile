.PHONY: install run test

install:
	pip install -r requirements.txt

run:
	python src/property_intel/train.py

test:
	pytest tests/ -v