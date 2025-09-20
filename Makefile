install:
	python -m venv .venv || true
	source .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

lint:
	source .venv/bin/activate && ruff check .

test:
	source .venv/bin/activate && pytest -q

train:
	source .venv/bin/activate && python src/train.py

ui:
	source .venv/bin/activate && mlflow ui
