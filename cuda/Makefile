BUILD_DIR_DEBUG = _build_debug
BUILD_DIR_RELEASE = _build_release
TARGET = main
BENCHMARK_TARGET = benchmark
CXX = g++-12
CC = gcc-12
CUDA_HOST_COMPILER = g++-12

.PHONY: all clean run debug release run-debug bench-debug bench-release run-bench run-bench-debug

all: release

# Debug build
debug: $(BUILD_DIR_DEBUG)/$(TARGET)

$(BUILD_DIR_DEBUG)/$(TARGET): CMakeLists.txt
	@mkdir -p $(BUILD_DIR_DEBUG)
	@cd $(BUILD_DIR_DEBUG) && cmake -DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_CXX_COMPILER=$(CXX) \
		-DCMAKE_C_COMPILER=$(CC) \
		-DCMAKE_CUDA_HOST_COMPILER=$(CUDA_HOST_COMPILER) \
		.. && $(MAKE) -j$(nproc)

# Release build
release: $(BUILD_DIR_RELEASE)/$(TARGET)

$(BUILD_DIR_RELEASE)/$(TARGET): CMakeLists.txt
	@mkdir -p $(BUILD_DIR_RELEASE)
	@cd $(BUILD_DIR_RELEASE) && cmake -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_CXX_COMPILER=$(CXX) \
		-DCMAKE_C_COMPILER=$(CC) \
		-DCMAKE_CUDA_HOST_COMPILER=$(CUDA_HOST_COMPILER) \
		.. && $(MAKE) -j$(nproc)

clean:
	@rm -rf $(BUILD_DIR_DEBUG) $(BUILD_DIR_RELEASE)

# Run release
run: $(BUILD_DIR_RELEASE)/$(TARGET)
	@echo "==================================================="
	@echo "Running RELEASE build"
	@echo "==================================================="
	@echo
	@cd $(BUILD_DIR_RELEASE) && ./$(TARGET)

# Run debug
run-debug: $(BUILD_DIR_DEBUG)/$(TARGET)
	@echo "==================================================="
	@echo "Running DEBUG build"
	@echo "==================================================="
	@echo
	@cd $(BUILD_DIR_DEBUG) && ./$(TARGET)

# Benchmark builds
bench: $(BUILD_DIR_RELEASE)/$(BENCHMARK_TARGET)

bench-debug: $(BUILD_DIR_DEBUG)/$(BENCHMARK_TARGET)

$(BUILD_DIR_DEBUG)/$(BENCHMARK_TARGET): CMakeLists.txt
	@mkdir -p $(BUILD_DIR_DEBUG)
	@cd $(BUILD_DIR_DEBUG) && cmake -DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_CXX_COMPILER=$(CXX) \
		-DCMAKE_C_COMPILER=$(CC) \
		-DCMAKE_CUDA_HOST_COMPILER=$(CUDA_HOST_COMPILER) \
		.. && $(MAKE) -j$(nproc)

$(BUILD_DIR_RELEASE)/$(BENCHMARK_TARGET): CMakeLists.txt
	@mkdir -p $(BUILD_DIR_RELEASE)
	@cd $(BUILD_DIR_RELEASE) && cmake -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_CXX_COMPILER=$(CXX) \
		-DCMAKE_C_COMPILER=$(CC) \
		-DCMAKE_CUDA_HOST_COMPILER=$(CUDA_HOST_COMPILER) \
		.. && $(MAKE) -j$(nproc)

# Run benchmark
run-bench: $(BUILD_DIR_RELEASE)/$(BENCHMARK_TARGET)
	@echo "==================================================="
	@echo "Running BENCHMARK (Release build)"
	@echo "==================================================="
	@echo
	@cd $(BUILD_DIR_RELEASE) && ./$(BENCHMARK_TARGET)

# Run benchmark debug
run-bench-debug: $(BUILD_DIR_DEBUG)/$(BENCHMARK_TARGET)
	@echo "==================================================="
	@echo "Running BENCHMARK (Debug build)"
	@echo "==================================================="
	@echo
	@cd $(BUILD_DIR_DEBUG) && ./$(BENCHMARK_TARGET)
