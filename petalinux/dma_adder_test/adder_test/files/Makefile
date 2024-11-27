APP = addertest

# Add any other object files to this list below
INCLUDE_DIR = /home/jwbuchta/Desktop/petabruh/dma_no_bsp/project-spec/meta-user/recipes-apps/addertest/files
SRC_DIR = /home/jwbuchta/Desktop/petabruh/dma_no_bsp/project-spec/meta-user/recipes-apps/addertest/files
OBJ_DIR = /home/jwbuchta/Desktop/petabruh/dma_no_bsp/project-spec/meta-user/recipes-apps/addertest/files
BIN_DIR = /home/jwbuchta/Desktop/petabruh/dma_no_bsp/project-spec/meta-user/recipes-apps/addertest/files

# Source files
SRCS := $(wildcard $(SRC_DIR)/*.c)
# Object files
OBJS := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))

# Output executable
TARGET := $(BIN_DIR)/$(APP)

# Default target
all: $(TARGET)

# Link the target executable
$(TARGET): $(OBJS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^

# Compile each source file to an object file
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c -o $@ $<

# Create directories if they don't exist
# $(BIN_DIR):
# 	mkdir -p $(BIN_DIR)

# $(OBJ_DIR):
# 	mkdir -p $(OBJ_DIR)

# Clean up generated files
clean:
	rm $(OBJS) $(TARGET)

# Add install target if needed (optional)
install: $(TARGET)
	install -d $(DESTDIR)
	install -m 0755 $(TARGET) $(DESTDIR)

.PHONY: all clean install