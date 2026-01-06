BUILD_DIR_DEBUG = _build_debug
BUILD_DIR_RELEASE = _build_release
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
	./run_format.sh

release: $(BUILD_DIR_RELEASE)/Makefile
	@cd $(BUILD_DIR_RELEASE) && $(MAKE) -j$(shell nproc)

debug: $(BUILD_DIR_DEBUG)/Makefile
	@cd $(BUILD_DIR_DEBUG) && $(MAKE) -j$(shell nproc)

$(BUILD_DIR_RELEASE)/Makefile: CMakeLists.txt
	@mkdir -p $(BUILD_DIR_RELEASE)
	@cd $(BUILD_DIR_RELEASE) && cmake -DCMAKE_BUILD_TYPE=Release \
		-DPython_EXECUTABLE=$(shell which $(PYTHON)) \
		-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(OUTPUT_DIR) \
		..

$(BUILD_DIR_DEBUG)/Makefile: CMakeLists.txt
	@mkdir -p $(BUILD_DIR_DEBUG)
	@cd $(BUILD_DIR_DEBUG) && cmake -DCMAKE_BUILD_TYPE=Debug \
		-DPython_EXECUTABLE=$(shell which $(PYTHON)) \
		-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(OUTPUT_DIR) \
		..

clean:
	@rm -rf $(BUILD_DIR_DEBUG) $(BUILD_DIR_RELEASE)
	@rm -f culsh/_culsh_core*.so
