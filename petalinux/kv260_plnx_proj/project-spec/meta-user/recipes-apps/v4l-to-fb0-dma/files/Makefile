APP = v4l-to-fb0-dma

# Source directories
APP_SRC_DIR = .
# COMMON_SRC_DIR = $(SR_MQP_ROOT)/petalinux/kv260_plnx_proj/project-spec/meta-user/recipes-apps/common-src
COMMON_SRC_DIR = /home/jwbuchta/Desktop/SR_MQP/petalinux/kv260_plnx_proj/project-spec/meta-user/recipes-apps/common-src

# Source files - add to this list as necessary
APP_SRCS = $(APP_SRC_DIR)/v4l-to-fb0-dma.cpp
COMMON_SRCS = $(COMMON_SRC_DIR)/axi-dma.cpp $(COMMON_SRC_DIR)/phys-mman.cpp 

# Object files
APP_OBJS = $(APP_SRCS:.cpp=.o) $(COMMON_SRCS:.cpp=.o)

# Compiler flags
FLAGS = -Wall -O2 -Wextra -I$(COMMON_SRC_DIR)

all: build
build: $(APP)

$(APP): $(APP_OBJS)
	$(CXX) -o $@ $(APP_OBJS) $(LDFLAGS) $(LDLIBS) $(FLAGS) 

# Pattern rule for compiling .cpp to .o
%.o: %.cpp
	$(CXX) $(FLAGS) -c $< -o $@

# Clean target
clean:
	rm -f $(APP) $(APP_OBJS)
	find $(COMMON_SRC_DIR) -name '*.o' -delete
