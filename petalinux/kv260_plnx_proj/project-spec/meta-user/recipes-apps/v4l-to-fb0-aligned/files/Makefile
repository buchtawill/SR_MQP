APP = v4l-to-fb0-aligned

# Add any other object files to this list below
APP_OBJS = v4l-to-fb0-aligned.o

all: build

build: $(APP)

$(APP): $(APP_OBJS)
	$(CXX) -o $@ $(APP_OBJS) $(LDFLAGS) $(LDLIBS)
clean:
	rm -f $(APP) *.o
