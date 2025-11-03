# Setting up a Development Environment

This document details how to set up a local development environment that will
allow you to contribute changes to the project.

Acquire sources and create virtualenv.
```shell
git clone https://github.com/langchain-ai/langchain-postgres
cd langchain-postgres
uv venv --python=3.13
source .venv/bin/activate
```

Install package in editable mode.
```shell
uv sync --group test
```

Start PostgreSQL/PGVector.
```shell
docker run --rm -it --name pgvector-container \
  -e POSTGRES_USER=langchain \
  -e POSTGRES_PASSWORD=langchain \
  -e POSTGRES_DB=langchain_test \
  -p 6024:5432 pgvector/pgvector:pg16 \
  postgres -c log_statement=all
```

Invoke test cases.
```shell
export POSTGRES_PORT=6024
pytest -vvv
```
