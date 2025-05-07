CC ?= gcc
CXX ?= g++
CPP ?= g++

APP_NAME = tflite_audio_classification_example
APP_OBJ_FILES = tflite_audio_classification_example.o

LIBS =  -ltensorflow-lite -lasound -lfftw3 

all: $(APP_NAME)

$(APP_NAME) : $(APP_OBJ_FILES)
	$(CXX) $^ -o $@  $(LIBS) -L.

%.o : %.cc
	$(CXX) -c $^ -o $@  -I/home/lozano/tensorflow/ -I/home/lozano/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include/


clean:
	rm *.o $(APP_NAME)