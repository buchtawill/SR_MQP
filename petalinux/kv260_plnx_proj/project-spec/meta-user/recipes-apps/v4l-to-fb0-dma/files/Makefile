APP = v4l-to-fb0-dma

# Add any other object files to this list below
APP_OBJS = v4l-to-fb0-dma.o axi-dma.o dma-sg-bd.o phys-mman.o

FLAGS = -Wall -O2 -Wextra

all: build

build: $(APP)

$(APP): $(APP_OBJS)
	$(CXX) -o $@ $(APP_OBJS) $(LDFLAGS) $(LDLIBS) $(FLAGS)
clean:
	rm -f $(APP) *.o
