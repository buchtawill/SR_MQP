APP = dma-to-fb

# Add any other object files to this list below
APP_OBJS = dma-to-fb.o

all: build

build: $(APP)

$(APP): $(APP_OBJS)
	$(CXX) -o $@ $(APP_OBJS) $(LDFLAGS) $(LDLIBS)
clean:
	rm -f $(APP) *.o
