# Setting up a Development Environment

This document details how to set up a local development environment that will
allow you to contribute changes to langchain-postgres.

## Setup

Clone the repository and create a virtualenv:
```shell
git clone https://github.com/langchain-ai/langchain-postgres
cd langchain-postgres
uv venv --python=3.13
source .venv/bin/activate
```

Install dependencies, required for local development:
```shell
uv sync --group test
```

Start a PostgreSQL instance with `pgvector` extension:
```shell
docker compose up -d
```

## Testing

Run all unit tests:
```shell
make test
```

Run unit tests from a single file:
```shell
make test TEST_FILE=tests/unit_tests/v2/test_engine.py
```

Run all unit tests in watch mode:
```shell
make test_watch
```

## Linting and formatting

Format all files using `ruff` and fix errors:
```shell
make format
```

Lint all files using `ruff` and `mypy`:
```shell
make lint
```

Spell check all files and fix errors:
```shell
make spell_fix
```