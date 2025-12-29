BUILD_DIR_DEBUG = _build_debug
BUILD_DIR_RELEASE = _build_release
CXX = g++-12
CC = gcc-12
CUDA_HOST_COMPILER = g++-12
PYTHON = python3
OUTPUT_DIR = $(shell pwd)/culsh

.PHONY: all clean release debug stubs help

all: release

help:
	@echo "Available targets:"
	@echo "  release     - Build the extension in Release mode (default)"
	@echo "  debug       - Build the extension in Debug mode"
	@echo "  stubs       - Generate .pyi type stubs for the extension"
	@echo "  clean       - Remove build directories"

stubs:
	@$(PYTHON) scripts/generate_stubs.py

release: $(BUILD_DIR_RELEASE)/Makefile
	@cd $(BUILD_DIR_RELEASE) && $(MAKE) -j$(shell nproc)

debug: $(BUILD_DIR_DEBUG)/Makefile
	@cd $(BUILD_DIR_DEBUG) && $(MAKE) -j$(shell nproc)

$(BUILD_DIR_RELEASE)/Makefile: CMakeLists.txt
	@mkdir -p $(BUILD_DIR_RELEASE)
	@cd $(BUILD_DIR_RELEASE) && cmake -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_CXX_COMPILER=$(CXX) \
		-DCMAKE_C_COMPILER=$(CC) \
		-DCMAKE_CUDA_HOST_COMPILER=$(CUDA_HOST_COMPILER) \
		-DPython_EXECUTABLE=$(shell which $(PYTHON)) \
		-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(OUTPUT_DIR) \
		..

$(BUILD_DIR_DEBUG)/Makefile: CMakeLists.txt
	@mkdir -p $(BUILD_DIR_DEBUG)
	@cd $(BUILD_DIR_DEBUG) && cmake -DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_CXX_COMPILER=$(CXX) \
		-DCMAKE_C_COMPILER=$(CC) \
		-DCMAKE_CUDA_HOST_COMPILER=$(CUDA_HOST_COMPILER) \
		-DPython_EXECUTABLE=$(shell which $(PYTHON)) \
		-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(OUTPUT_DIR) \
		..

clean:
	@rm -rf $(BUILD_DIR_DEBUG) $(BUILD_DIR_RELEASE)
	@rm -f culsh/_culsh_core*.so
