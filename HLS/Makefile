# Define the compiler
CXX = g++
CXXFLAGS = -std=c++11 -Wall

# Define the target executable
TARGET = sim

# Define the source files
SRCS = Interpolation_v2.cpp Interpolation_v2_tb.cpp

# Define the object files
OBJS = $(SRCS:.cpp=.o)

# The default target to build the executable
all: $(TARGET)

# Link the object files to create the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Compile .cpp files into .o files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up object files and the executable
clean:
	rm -f $(OBJS) $(TARGET)

# Run the executable (optional)
run: $(TARGET)
	./$(TARGET)