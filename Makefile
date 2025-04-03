.PHONY: install lint format check notebook clean

install:
	poetry install

lint:
	poetry run ruff check .

format:
	poetry run ruff format .

check:
	poetry run mypy .

notebook:
	poetry run jupyter-lab

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete 