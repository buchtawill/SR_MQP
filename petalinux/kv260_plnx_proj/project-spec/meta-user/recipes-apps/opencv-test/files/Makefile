APP = opencv-test

# Add any other object files to this list below
APP_OBJS = opencv-test.o

all: build

build: $(APP)

$(APP): $(APP_OBJS)
	$(CXX) -o $@ $(APP_OBJS) $(LDFLAGS) $(LDLIBS) `pkg-config --cflags --libs opencv4`
clean:
	rm -f $(APP) *.o
