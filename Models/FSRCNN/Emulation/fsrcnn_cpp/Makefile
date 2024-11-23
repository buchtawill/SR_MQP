# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -arch arm64 -Wno-unused-parameter

# macOS OpenCV settings (Homebrew)
OPENCV_PREFIX = $(shell brew --prefix opencv)
OPENCV_CFLAGS = -I$(OPENCV_PREFIX)/include/opencv4
OPENCV_LIBS = -L$(OPENCV_PREFIX)/lib \
              -lopencv_core \
              -lopencv_highgui \
              -lopencv_imgproc \
              -lopencv_imgcodecs

# PyTorch/LibTorch settings (using Conda path)
TORCH_PREFIX = $(shell python3 -c "import torch; print(torch.__path__[0])")
TORCH_CFLAGS = -I$(TORCH_PREFIX)/include \
               -I$(TORCH_PREFIX)/include/torch/csrc/api/include \
               -D_GLIBCXX_USE_CXX11_ABI=0

TORCH_LIBS = -L$(TORCH_PREFIX)/lib \
             -ltorch \
             -ltorch_cpu \
             -lc10

# Project settings
TARGET = fsrcnn

# Source files
SOURCES = main.cpp \
          adaptive_upscaler.cpp \
          processing_timer.cpp \
          quantization.cpp

OBJECTS = $(SOURCES:.cpp=.o)

# Include paths
INCLUDES = $(OPENCV_CFLAGS) $(TORCH_CFLAGS)

# Combined flags
ALL_CXXFLAGS = $(CXXFLAGS) $(INCLUDES)
ALL_LIBS = $(OPENCV_LIBS) $(TORCH_LIBS)

# Rpath settings for macOS
RPATH_FLAGS = -Wl,-rpath,$(OPENCV_PREFIX)/lib -Wl,-rpath,$(TORCH_PREFIX)/lib

# Default target
all: $(TARGET)

# Linking
$(TARGET): $(OBJECTS)
	@echo "Linking $@..."
	$(CXX) $(OBJECTS) -o $(TARGET) $(ALL_LIBS) $(RPATH_FLAGS)

# Compilation
%.o: %.cpp
	@echo "Compiling $<..."
	$(CXX) $(ALL_CXXFLAGS) -c $< -o $@

# Clean built files
clean:
	@echo "Cleaning..."
	@rm -f $(OBJECTS) $(TARGET)

# Run the program
run: $(TARGET)
	@echo "Running fsrcnn..."
	./$(TARGET) --image img.png

# Print debug info
info:
	@echo "Source files: $(SOURCES)"
	@echo "Object files: $(OBJECTS)"
	@echo "OpenCV prefix: $(OPENCV_PREFIX)"
	@echo "Torch prefix: $(TORCH_PREFIX)"
	@echo "Compiler flags: $(ALL_CXXFLAGS)"
	@echo "Libraries: $(ALL_LIBS)"

.PHONY: all clean run info