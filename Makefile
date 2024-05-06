.PHONY: clean test coverage build install lint

NPROCS:=$(shell grep -c ^processor /proc/cpuinfo)

# ============================================================================ #
# CLEAN COMMANDS
# ============================================================================ #

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage*
	rm -fr htmlcov/
	rm -fr .pytest_cache

# ============================================================================ #
# LINT COMMANDS
# ============================================================================ #

lint:
# Lint all files in the current directory (and any subdirectories).
	ruff check --fix

format:
# Format all files in the current directory (and any subdirectories).
	ruff format

# ============================================================================ #
# TEST COMMANDS
# ============================================================================ #

test: ## run tests quickly with the default Python
	pytest

test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	coverage run --source comp_bench_tools -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

# ============================================================================ #
# BUILD COMMANDS
# ============================================================================ #

build: clean ## builds wheel package
	NVTE_IS_PACKAGING=1 MAKEFLAGS="-j${NPROCS}" python setup.py bdist_wheel
	ls -l dist

build_jax: install ## install the package to the active Python's site-packages
# TODO: Build the JAX Plugin
# python setup.py sdist
	echo "ERROR, read the TODO"

build_paddle: install ## install the package to the active Python's site-packages
# TODO: Build the Paddle Plugin
# python setup.py sdist
	echo "ERROR, read the TODO"

build_torch: install ## install the package to the active Python's site-packages
# TODO: Build the Torch Plugin
# python setup.py sdist
	echo "ERROR, read the TODO"

# ============================================================================ #
# INSTALL COMMANDS
# ============================================================================ #

install: ## install the package to the active Python's site-packages
	MAKEFLAGS="-j${NPROCS}" pip install -e .

install_jax: install ## install the package to the active Python's site-packages
# TODO: Install the JAX Plugin
# pip install -e .
	echo "ERROR, read the TODO"

install_paddle: install ## install the package to the active Python's site-packages
# TODO: Install the Paddle Plugin
# pip install -e .
	echo "ERROR, read the TODO"

install_torch: install ## install the package to the active Python's site-packages
# TODO: Install the Torch Plugin
# pip install -e .
	echo "ERROR, read the TODO"