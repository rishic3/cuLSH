BUILD_DIR_DEBUG = _build_debug
BUILD_DIR_RELEASE = _build_release
CXX = g++-12
CC = gcc-12
CUDA_HOST_COMPILER = g++-12

.PHONY: all clean debug release help

all: release

help:
	@echo "Available targets:"
	@echo "  release  - Build in Release mode (default)"
	@echo "  debug    - Build in Debug mode"
	@echo "  clean    - Remove build directories"

release:
	@mkdir -p $(BUILD_DIR_RELEASE)
	@cd $(BUILD_DIR_RELEASE) && cmake -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_CXX_COMPILER=$(CXX) \
		-DCMAKE_C_COMPILER=$(CC) \
		-DCMAKE_CUDA_HOST_COMPILER=$(CUDA_HOST_COMPILER) \
		.. && $(MAKE) -j$$(nproc)

debug:
	@mkdir -p $(BUILD_DIR_DEBUG)
	@cd $(BUILD_DIR_DEBUG) && cmake -DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_CXX_COMPILER=$(CXX) \
		-DCMAKE_C_COMPILER=$(CC) \
		-DCMAKE_CUDA_HOST_COMPILER=$(CUDA_HOST_COMPILER) \
		.. && $(MAKE) -j$$(nproc)

clean:
	@rm -rf $(BUILD_DIR_DEBUG) $(BUILD_DIR_RELEASE)
