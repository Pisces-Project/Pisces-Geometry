# ----------------------- #
# Metadata for PYTHON     #
# ----------------------- #
# These settings MAY need to be modified by new
# users in order to get everything working vis-a-vis
# the make ... command style.
#
# If you're just a user, you DON'T want to be here. You should
# install via pip install pymetric instead.
PYTHON := python3                        # The python command from which to build the venv
VENV_DIR := .venv                        # Directory to build the .venv in.
ACTIVATE := . $(VENV_DIR)/bin/activate   # Activation command for the venv.

# -------------------------
# Installation & Setup
# -------------------------
# Commands for venv building and installation.
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip

venv-build:
	@echo "🐍 Building a fresh virtual environment at '$(VENV_DIR)'..."
	rm -rf $(VENV_DIR)
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "📦 Upgrading pip and installing dependencies..."
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -e .[dev]
	@echo "✅ Virtual environment setup complete."

venv-remove:
	@echo "🧹 Removing the virtual environment at '$(VENV_DIR)'..."
	rm -rf $(VENV_DIR)
	@echo "✅ Environment removed."

# ---------------------------- #
# Pre-Commit                   #
# ---------------------------- #
# These commands allow you to run the pre-commit system.
precommit-install:
	@echo "🔧 Installing pre-commit hooks..."
	$(VENV_PIP) install pre-commit
	$(VENV_PYTHON) -m pre_commit install

precommit-run:
	@echo "🧪 Running pre-commit hooks on all files..."
	$(VENV_PYTHON) -m pre_commit run --all-files

# ---------------------------- #
# Git Commands                 #
# ---------------------------- #
# These are simple developer getting started commands
# to generate the development branch you're going to be
# working on.
dev-branch:
	@echo "🌿 Creating and switching to a new development branch..."
	@git checkout -b dev/$(shell date +%Y-%m-%d)-$(USER)


# ---------------------------- #
# Testing                      #
# ---------------------------- #
# Run tests via pytest.

test:
	@echo "🧪 Running tests..."
	$(VENV_PYTHON) -m pytest -ra --strict-markers --log-cli-level=INFO


# ---------------------------- #
# Make Docs                    #
# ---------------------------- #
# Make the docs.
docs:
	@echo "📚 Building documentation..."
	sphinx-build -b html -j auto ./docs/source/ ./docs/build/html

docs-clean:
	@echo "🧹 Cleaning built documentation..."
	rm -rf docs/build
	rm -rf docs/source/_as_gen

# -------------------------
# Utility
# -------------------------

clean:
	@echo "🧹 Cleaning temporary files..."
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache .coverage .hypothesis .venv dist build
	find . -type d -name '__pycache__' -exec rm -r {} +

.PHONY: install update lock test lint format typecheck check docs docs-clean docs-serve clean