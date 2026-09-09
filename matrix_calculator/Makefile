# Matrix calculator — build/run automation
#
# Uses only the Python standard library plus a C compiler for the SIMD kernel.
# Targets: venv (create), run (demo), test (unit tests), libmatmul (build C SIMD
#          kernel), bench (benchmark), clean (remove venv + artifacts).

PYTHON := python3
VENV := .venv
VENV_PY := $(VENV)/bin/python

CC := cc
CFLAGS := -O3 -march=native -ffast-math -funroll-loops -fPIC -shared
# macOS uses .dylib, Linux uses .so; both are loadable by ctypes.
LIB_EXT := $(if $(filter Darwin,$(shell uname -s)),dylib,so)
LIB := libmatmul.$(LIB_EXT)

.PHONY: all venv run test libmatmul bench clean

all: run

venv: $(VENV_PY)

$(VENV_PY):
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created at $(VENV)"

run: venv
	$(VENV_PY) main.py

test: venv
	$(VENV_PY) -m unittest discover -p "test_*.py" -v

libmatmul: $(LIB)

$(LIB): matmul.c
	$(CC) $(CFLAGS) -o $(LIB) matmul.c
	@echo "Built SIMD kernel: $(LIB)"

bench: venv $(LIB)
	$(VENV_PY) benchmark.py

clean:
	rm -rf $(VENV)
	rm -f libmatmul.so libmatmul.dylib
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	@echo "Cleaned venv, native libs, and caches"
