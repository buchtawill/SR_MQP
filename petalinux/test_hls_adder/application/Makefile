# Cross-compiler prefix
CROSS_COMPILE = aarch64-linux-gnu-

# Compiler and linker
CC = $(CROSS_COMPILE)gcc
LD = $(CROSS_COMPILE)ld
FLAGS = -Wall

# Target application name
TARGET = test_app.bin

# Source files
SRC = main.c

# Object files
OBJ = $(SRC:.c=.o)

# Build rules
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) -o $(TARGET) $(OBJ) $(FLAGS)

%.o: %.c
	$(CC) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJ) $(TARGET)

# Install to a specific directory (optional)
# install:
# install -m 755 $(TARGET) /path/to/install/directory
