APP = v4l-to-fb0-dma

# Source directories
APP_SRC_DIR = .
# COMMON_SRC_DIR = $(SR_MQP_ROOT)/petalinux/kv260_plnx_proj/project-spec/meta-user/recipes-apps/common-src
COMMON_SRC_DIR = /home/jwbuchta/Desktop/SR_MQP/petalinux/kv260_plnx_proj/project-spec/meta-user/recipes-apps/common-src

# Source files - add to this list as necessary
APP_SRCS = $(APP_SRC_DIR)/v4l-to-fb0-dma.cpp
COMMON_SRCS = $(COMMON_SRC_DIR)/axi-dma.cpp $(COMMON_SRC_DIR)/phys-mman.cpp 

# Object files (all placed in APP_SRC_DIR in case header files need to be recompiled - i.e. axi-dma.h)
APP_OBJS = $(APP_SRC_DIR)/$(notdir $(APP_SRCS:.cpp=.o)) \
           $(APP_SRC_DIR)/$(notdir $(COMMON_SRCS:.cpp=.o))

# Compiler flags
FLAGS = -Wall -O2 -Wextra -I$(COMMON_SRC_DIR)

all: build
build: $(APP)

$(APP): $(APP_OBJS)
	$(CXX) -o $@ $(APP_OBJS) $(LDFLAGS) $(LDLIBS) $(FLAGS) 

# Pattern rule for compiling .cpp to .o (output in APP_SRC_DIR)
$(APP_SRC_DIR)/%.o: $(APP_SRC_DIR)/%.cpp
	$(CXX) $(FLAGS) -c $< -o $@

$(APP_SRC_DIR)/%.o: $(COMMON_SRC_DIR)/%.cpp
	$(CXX) $(FLAGS) -c $< -o $@

# Clean target
clean:
	rm -f $(APP) $(APP_OBJS)
