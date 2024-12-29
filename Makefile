###############################################################################
# Common make values.
lib    := par_ai_core
#run    := uv run --python 3.12
run    := uv run
python := $(run) python
pyright := $(run) pyright
ruff  := $(run) ruff
twine  := $(run) twine
#build  := $(python) -m build
build := uvx --from build pyproject-build --installer uv

export UV_LINK_MODE=copy
export PIPENV_VERBOSITY=-1
##############################################################################
# Run the app.
.PHONY: run
run:		# Run app
	$(run) $(lib)

.PHONY: app_help
app_help:		# Show app help
	$(run) $(lib)  --help


##############################################################################
.PHONY: uv-lock
uv-lock:
	uv lock

.PHONY: uv-sync
uv-sync:
	uv sync

.PHONY: setup
setup: uv-lock uv-sync	        # use this for first time run

.PHONY: resetup
resetup: remove-venv setup			# Recreate the virtual environment from scratch

.PHONY: remove-venv
remove-venv:			# Remove the virtual environment
	rm -rf .venv

.PHONY: depsupdate
depsupdate:			# Update all dependencies
	uv sync --python -U

.PHONY: depsshow
depsshow:			# Show the dependency graph
	uv tree

.PHONY: shell
shell:			# Start shell inside of .venv
	$(run) bash
##############################################################################
# Checking/testing/linting/etc.

.PHONY: format
format:				# Reformat the code with ruff.
	$(ruff) format src/$(lib)
	$(ruff) format tests

.PHONY: lint
lint:				# Run ruff over the library
	$(ruff) check src/$(lib) --fix

.PHONY: typecheck
typecheck:			# Perform static type checks with pyright
	$(pyright)

.PHONY: typecheck-stats
typecheck-stats:			# Perform static type checks with pyright and print stats
	$(pyright) --stats

.PHONY: checkall
checkall: format lint typecheck 	        # Check all the things

.PHONY: pre-commit              # run pre-commit checks on all files
pre-commit:
	pre-commit run --all-files

.PHONY: pre-commit-update               # run pre-commit and update hooks
pre-commit-update:
	pre-commit autoupdate

##############################################################################
# Package/publish.
.PHONY: package
package:			# Package the library
	$(build) -w

.PHONY: spackage
spackage:			# Create a source package for the library
	$(build) -s

.PHONY: packagecheck
packagecheck: clean package spackage docs		# Check the packaging.
	$(twine) check dist/*

.PHONY: testdist
testdist: packagecheck		# Perform a test distribution
	$(twine) upload --repository testpypi dist/*
	#$(twine) upload --skip-existing --repository testpypi dist/*

.PHONY: dist
dist: packagecheck		# Upload to pypi
	$(twine) upload --skip-existing dist/*

.PHONY: test
test:				# Run tests with coverage report
	$(run) pytest --cov=src/$(lib) --cov-report=term-missing --cov-report=html tests/

.PHONY: test-trace
test-trace:				# Run tests enable full trace
	$(run) pytest --full-trace

.PHONY: coverage
coverage:			# Generate coverage report and output xml report
	$(run) pytest --cov=src/$(lib) --cov-report=xml tests/

##############################################################################
# Utility.

.PHONY: docs
docs:			# Generate html documentation
	$(run) pdoc3 --html --force -o ./src/par_ai_core/html par_ai_core
	mv ./src/par_ai_core/html/par_ai_core ./src/par_ai_core/docs

.PHONY: repl
repl:				# Start a Python REPL
	$(python)

.PHONY: clean
clean:				# Clean the build directories
	rm -rf build dist $(lib).egg-info

.PHONY: help
help:				# Display this help
	@grep -Eh "^[a-z]+:.+# " $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.+# "}; {printf "%-20s %s\n", $$1, $$2}'

##############################################################################
# Housekeeping tasks.
.PHONY: housekeeping
housekeeping:			# Perform some git housekeeping
	git fsck
	git gc --aggressive
	git remote update --prune
