APP = interpolate2x

# Source directories
APP_SRC_DIR = .
COMMON_SRC_DIR = /home/jwbuchta/Desktop/SR_MQP/petalinux/kv260_plnx_proj/project-spec/meta-user/recipes-apps/common-src

# Source files
APP_SRCS = $(APP_SRC_DIR)/interpolate2x.cpp
COMMON_SRCS = $(COMMON_SRC_DIR)/axi-dma.cpp $(COMMON_SRC_DIR)/phys-mman.cpp

# Object files (all placed in APP_SRC_DIR in case header files need to be recompiled - i.e. axi-dma.h)
APP_OBJS = $(APP_SRC_DIR)/$(notdir $(APP_SRCS:.cpp=.o)) \
           $(APP_SRC_DIR)/$(notdir $(COMMON_SRCS:.cpp=.o))

# Compiler flags
FLAGS = -Wall -O2 -Wextra -I$(COMMON_SRC_DIR) -std=c++17 -DDMA_DIRECT_REG_MODE=1

all: build

build: $(APP)

$(APP): $(APP_OBJS)
	$(CXX) -o $@ $(APP_OBJS) $(LDFLAGS) $(LDLIBS)

# Pattern rule for compiling .cpp to .o (output in APP_SRC_DIR)
$(APP_SRC_DIR)/%.o: $(APP_SRC_DIR)/%.cpp
	$(CXX) $(FLAGS) -c $< -o $@

$(APP_SRC_DIR)/%.o: $(COMMON_SRC_DIR)/%.cpp
	$(CXX) $(FLAGS) -c $< -o $@

clean:
	rm -f $(APP) $(APP_SRC_DIR)/*.o

