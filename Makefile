PY      ?= python3
PIP     ?= pip
PYTEST  ?= pytest
BUILD_DIR ?= build
JOBS    ?= $(shell sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

# Local dev: build with -march=native for max perf. Override with
# `make build CMAKE_OPTS=` to produce a portable build.
CMAKE_OPTS ?= -DSCREAMER_NATIVE_ARCH=ON

.PHONY: help build test docs benchmark clean \
        regen-init install-dev \
        patch minor major release-push \
        bump-tools

help:
	@echo "Targets:"
	@echo "  make build       Build C++ extension via cmake; copy .so into screamer/"
	@echo "  make test        Build, install -e, run pytest"
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

minor: regen-init
	bump-my-version bump minor

major: regen-init
	bump-my-version bump major

release-push:
	git push origin main
	git push origin --tags

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean:
	rm -rf $(BUILD_DIR) dist *.egg-info
	rm -f screamer/screamer_bindings*.so
