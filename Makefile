PY      ?= python3
PIP     ?= pip
PYTEST  ?= pytest
BUILD_DIR ?= build
JOBS    ?= $(shell sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

# Local dev: build with -march=native for max perf. Override with
# `make build CMAKE_OPTS=` to produce a portable build.
# Python3_EXECUTABLE pins cmake's find_package(Python3) to the python on
# PATH. Without this, pybind11 v3.x's new FindPython mode searches the
# system and picks the highest-version python it finds (often a Homebrew
# python rather than the one the user wants). We resolve via sys.executable
# rather than `command -v` because pyenv shims aren't real binaries and
# cmake won't accept them.
PYTHON_EXE := $(shell $(PY) -c "import sys; print(sys.executable)" 2>/dev/null)
CMAKE_OPTS ?= -DSCREAMER_NATIVE_ARCH=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DPython3_EXECUTABLE=$(PYTHON_EXE)

# clang-tidy: prefer Homebrew's llvm if installed (Apple Xcode does not ship
# clang-tidy by default). Override with CLANG_TIDY=... if needed.
CLANG_TIDY ?= clang-tidy
ifneq (,$(wildcard /opt/homebrew/opt/llvm/bin/clang-tidy))
  CLANG_TIDY = /opt/homebrew/opt/llvm/bin/clang-tidy
endif

.PHONY: help build test docs benchmark clean \
        regen-init install-dev tidy \
        patch minor major release-push \
        bump-tools

help:
	@echo "Targets:"
	@echo "  make build       Build C++ extension via cmake; copy .so into screamer/"
	@echo "  make test        Build, install -e, run pytest"
	@echo "  make tidy        Run clang-tidy (catches uninit class members, etc.)"
	@echo "  make docs        Build Sphinx HTML docs"
	@echo "  make benchmark   Run benchmark suite + plots"
	@echo "  make patch       Bump patch (0.1.46 -> 0.1.47), commit + tag"
	@echo "  make minor       Bump minor (0.1.46 -> 0.2.0), commit + tag"
	@echo "  make major       Bump major (0.1.46 -> 1.0.0), commit + tag"
	@echo "  make release-push  Push commits and tags to origin"
	@echo "  make bump-tools  Install bump-my-version (one-time)"
	@echo "  make clean       Remove build artifacts and compiled extension"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
build:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. -DCMAKE_BUILD_TYPE=Release $(CMAKE_OPTS)
	cd $(BUILD_DIR) && cmake --build . -j $(JOBS)
	cp $(BUILD_DIR)/screamer_bindings*.so screamer/
	$(MAKE) regen-init

regen-init:
	PYTHONPATH=. $(PY) devtools/generate_screamer__init__.py

install-dev: build
	$(PIP) install -e .

test: install-dev
	$(PYTEST)

# clang-tidy with cppcoreguidelines-pro-type-member-init catches uninitialised
# class members (the bug class behind RollingZscore's silent failure on
# Ubuntu+CPython 3.14). Run `make build` first so build/compile_commands.json
# exists.
tidy: build
	@echo ">>> clang-tidy: cppcoreguidelines-pro-type-member-init"
	$(CLANG_TIDY) -p build --quiet --warnings-as-errors='*' \
	  --checks='-*,cppcoreguidelines-pro-type-member-init' \
	  bindings/*.cpp src/screamer/common/*.cpp src/screamer/detail/*.cpp

# ---------------------------------------------------------------------------
# Docs
# ---------------------------------------------------------------------------
docs:
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	@echo "Docs built at docs/_build/html/index.html"

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
benchmark:
	$(PY) benchmarks/run_benchmarks.py
	$(PY) benchmarks/make_plots.py
	$(PY) benchmarks/make_rank_plot.py

# ---------------------------------------------------------------------------
# Release / version bumping (bump-my-version)
# ---------------------------------------------------------------------------
bump-tools:
	$(PIP) install --upgrade bump-my-version

patch: regen-init
	bump-my-version bump patch
	$(MAKE) release-push

minor: regen-init
	bump-my-version bump minor
	$(MAKE) release-push

major: regen-init
	bump-my-version bump major
	$(MAKE) release-push

# Push commit + tag to origin. Tag push triggers build-wheels.yml.
release-push:
	@echo ">>> pushing commit to origin/main"
	git push origin main
	@echo ">>> pushing tags to origin"
	git push origin --tags

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean:
	rm -rf $(BUILD_DIR) dist *.egg-info
	rm -f screamer/screamer_bindings*.so
