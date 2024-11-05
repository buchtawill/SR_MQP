# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -arch arm64

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
SRCS = main.cpp
OBJS = $(SRCS:.cpp=.o)

# Combined flags
ALL_CXXFLAGS = $(CXXFLAGS) $(OPENCV_CFLAGS) $(TORCH_CFLAGS)
ALL_LIBS = $(OPENCV_LIBS) $(TORCH_LIBS)

# Rpath settings for macOS
RPATH_FLAGS = -Wl,-rpath,$(OPENCV_PREFIX)/lib -Wl,-rpath,$(TORCH_PREFIX)/lib

# Default target
all: $(TARGET)

# Linking
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) $(ALL_LIBS) $(RPATH_FLAGS)

# Compilation
%.o: %.cpp
	$(CXX) $(ALL_CXXFLAGS) -c $< -o $@

# Clean built files
clean:
	rm -f $(OBJS) $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run