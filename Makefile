.PHONY: all lint format test help

# Default target executed when no arguments are given to make.
all: help

######################
# TESTING AND COVERAGE
######################

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/

test:
	uv run pytest --disable-socket --allow-unix-socket $(TEST_FILE)

test_watch:
	uv run ptw . -- $(TEST_FILE)


######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=. --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')

lint lint_diff:
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff check $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || uv run mypy $(PYTHON_FILES)

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff check --fix $(PYTHON_FILES)

spell_check:
	uv run codespell --toml pyproject.toml

spell_fix:
	uv run codespell --toml pyproject.toml -w

######################
# HELP
######################

help:
	@echo '===================='
	@echo '-- LINTING --'
	@echo 'format                               - run code formatters on entire project and fix errors'
	@echo 'format_diff                          - run code formatters on changed files (compared to main) and fix errors'
	@echo 'format PYTHON_FILES=<path>           - run code formatters on specific path and fix errors'
	@echo 'lint                                 - run linters on entire project'
	@echo 'lint_diff                            - run linters on changed files (compared to main)'
	@echo 'lint PYTHON_FILES=<path>             - run linters on specific path'
	@echo 'spell_check                          - run codespell on entire project'
	@echo 'spell_fix                            - run codespell on entire project and fix errors'
	@echo '-- TESTS --'
	@echo 'test                                 - run all unit tests'
	@echo 'test TEST_FILE=<test_file>           - run unit tests from file'
	@echo 'test_watch                           - run all unit tests (watch mode)'
	@echo 'test_watch TEST_FILE=<test_file>     - run unit tests from file (watch mode)'
	@echo '-- DOCUMENTATION tasks are from the top-level Makefile --'
