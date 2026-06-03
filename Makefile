.PHONY: install run test

install:
	pip install -r requirements.txt

run:
	cd src && python -m property_intel.train

test:
	pytest tests/ -v