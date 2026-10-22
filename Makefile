.PHONY: install install-all bootstrap api ui demo test test-all lint eval docker-up airflow-up airflow-down

install:
	python -m pip install -r requirements.txt

install-all:
	python -m pip install --pre -c constraints.txt -e ".[dev,ui,observability]"

bootstrap:
	python -m scripts.bootstrap

api:
	uvicorn app.api.main:app --reload

ui:
	streamlit run app/ui/streamlit_app.py

demo:
	python -m scripts.demo

# Fast: unit + smoke only (no Ollama required, ~seconds)
test:
	python -m scripts.bootstrap --rows 350 --max-iter 1000
	pytest -q

# Full deterministic path: integration + golden-dataset evaluation, no Ollama required.
test-all:
	python -m scripts.bootstrap --rows 350 --max-iter 1000
	AGENT_PROVIDER=deterministic RETRIEVAL_BACKEND=lexical pytest -q -m ""
	AGENT_PROVIDER=deterministic RETRIEVAL_BACKEND=lexical python -m scripts.run_evaluation

lint:
	ruff check .

eval:
	python -m scripts.run_evaluation

docker-up:
	docker compose up --build

airflow-up:
	docker compose -f docker-compose.airflow.yml up --build

airflow-down:
	docker compose -f docker-compose.airflow.yml down
