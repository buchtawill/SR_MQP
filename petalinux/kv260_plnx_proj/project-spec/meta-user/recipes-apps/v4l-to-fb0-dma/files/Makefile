APP = v4l-to-fb0-dma

# Add any other object files to this list below
APP_OBJS = v4l-to-fb0-dma.o axi-dma.o dma-sg-bd.o

all: build

build: $(APP)

$(APP): $(APP_OBJS)
	$(CXX) -o $@ $(APP_OBJS) $(LDFLAGS) $(LDLIBS)
clean:
	rm -f $(APP) *.o
