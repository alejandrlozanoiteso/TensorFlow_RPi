CC ?= gcc
CXX ?= g++
CPP ?= g++

APP_NAME = tflite_classification
APP_OBJ_FILES = tflite_classification_example.o

LIBS = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio -lopencv_imgcodecs -ltensorflow-lite 

all: $(APP_NAME)

$(APP_NAME) : $(APP_OBJ_FILES)
	$(CXX) $^ -o $@  $(LIBS) -L.

%.o : %.cc
	$(CXX) -c $^ -o $@ -I/usr/include/opencv4/ -I/home/lozano/tensorflow/ -I/home/lozano/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include/


clean:
	rm *.o $(APP_NAME)