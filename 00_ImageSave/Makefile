CC ?= gcc
CXX ?= g++
CPP ?= g++

APP_NAME = image_save
APP_OBJ_FILES = image_save.o

LIBS = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio -lopencv_imgcodecs 

all: $(APP_NAME)

$(APP_NAME) : $(APP_OBJ_FILES)
	$(CXX) $^ -o $@  $(LIBS)

%.o : %.cc
	$(CXX) -c $^ -o $@ -I/usr/include/opencv4/


clean:
	rm *.o $(APP_NAME)