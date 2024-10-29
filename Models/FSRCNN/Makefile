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

# Project settings
TARGET = fsrcnn
SRCS = main.cpp
OBJS = $(SRCS:.cpp=.o)

# Combined flags
ALL_CXXFLAGS = $(CXXFLAGS) $(OPENCV_CFLAGS)
ALL_LIBS = $(OPENCV_LIBS)

# Default target
all: $(TARGET)

# Linking
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) $(ALL_LIBS)

# Compilation
%.o: %.cpp
	$(CXX) $(ALL_CXXFLAGS) -c $< -o $@

# Clean built files
clean:
	rm -f $(OBJS) $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Install dependencies (macOS)
install-deps:
	brew update
	brew install opencv
	brew install pkg-config
	pip install torch torchvision

# Create test directory and download sample image
setup-test:
	mkdir -p testImage
	curl -o testImage/img.jpg https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/inputs/00003.jpg

.PHONY: all clean run install-deps setup-test