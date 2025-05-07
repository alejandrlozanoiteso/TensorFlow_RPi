#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/string_util.h>
#include <tensorflow/lite/examples/label_image/get_top_n.h>
#include <tensorflow/lite/model.h>

#define FILE    1
#define CAMERA  2
#define VLC     3

#define TEST  CAMERA

#define HEIGHT 480
#define WIDTH   640
#define FPS 30


/*****************************************************Load labels from File *************************************/

std::vector<std::string> load_labels(std::string labels_file)
{
    std::ifstream file(labels_file.c_str());
    if (!file.is_open())
    {
        fprintf(stderr, "unable to open label file\n");
        exit(-1);
    }
    std::string label_str;
    std::vector<std::string> labels;

    while (std::getline(file, label_str))
    {
        if (label_str.size() > 0)
            labels.push_back(label_str);
    }
    file.close();
    return labels;
}

int main(int argc, char **argv)
{

#if TEST==CAMERA
    cv::VideoCapture cam(0);
        /*Configura camara */
        cam.set(cv::CAP_PROP_FRAME_HEIGHT, HEIGHT);
        cam.set(cv::CAP_PROP_FRAME_WIDTH, WIDTH);
        cam.set(cv::CAP_PROP_FPS, FPS);
#elif TEST==VLC
        cv::VideoCapture cam("rtsp://192.168.100.14:8554/camera", cv::CAP_FFMPEG);
#endif

    // Get Model label and input image
    if (argc != 3)
    {
        fprintf(stderr, "TfliteClassification.exe modelfile image\n");
        exit(-1);
    }
    const char *modelFileName = argv[1];
    const char *imageFile = argv[2];

    // Load Model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(modelFileName);
    if (model == nullptr)
    {
        fprintf(stderr, "failed to load model\n");
        exit(-1);
    }
    // Initiate Interpreter
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);
    if (interpreter == nullptr)
    {
        fprintf(stderr, "Failed to initiate the interpreter\n");
        exit(-1);
    }

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        fprintf(stderr, "Failed to allocate tensor\n");
        exit(-1);
    }

    // Configure the interpreter
    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(8);

    /**********************************************Input Tensors *******************************************/
    std::cout << "/**********************************************Input Tensors *******************************************/" << std::endl;
    // Get Input Tensor Dimensions
    std::cout << "The model Input has " << interpreter->inputs().size() << " input tensors" << std::endl;  
    /*Get first input */
    int input = interpreter->inputs()[0];
    std::cout << "The first input tensor has " << interpreter->tensor(input)->dims->size << " dimensions" << std::endl;
    std::cout << "Type of tensor " << interpreter->tensor(input)->type << std::endl;
    std::cout << "Size of tensor is " << interpreter->tensor(input)->bytes << std::endl;

    int dimenssions  = interpreter->tensor(input)->dims->size;
    for(int dim = 0; dim < dimenssions; dim++ )
        std::cout << "Dimenssion " << dim << " is "  << interpreter->tensor(input)->dims->data[dim] << " size " << std::endl;

    /**********************************************Output Tensors *******************************************/
    std::cout << "/**********************************************Output Tensors *******************************************/" << std::endl;
    // Get Output Tensor Dimensions
    std::cout << "The model Output has " << interpreter->outputs().size() << "  output tensors" << std::endl;  
    /*Get first input */
    int output = interpreter->outputs()[0];
    std::cout << "The first output tensor has " << interpreter->tensor(output)->dims->size << " dimensions" << std::endl;
    std::cout << "Type of tensor " << interpreter->tensor(output)->type << std::endl;
    std::cout << "Size of tensor is " << interpreter->tensor(output)->bytes << std::endl;

    dimenssions  = interpreter->tensor(output)->dims->size;
    for(int dim = 0; dim < dimenssions; dim++ )
        std::cout << "Dimenssion " << dim << " is "  << interpreter->tensor(output)->dims->data[dim] << " size " << std::endl;    
    
    /********************************************** Get input data and copy to tensor *******************************************/

    // Load Input Image
    cv::Mat imageGray, imagGrayfloat;

#if TEST==FILE
    auto frame = cv::imread(imageFile);
#endif

    for(;;) {

#if TEST != FILE
        cv::Mat frame;
        cam >> frame;
#endif
        if (frame.empty())
        {
            fprintf(stderr, "Failed to load iamge\n");
            exit(-1);
        }
    
        /*Preprocess input */
        cv::cvtColor(frame, imageGray, cv::COLOR_BGR2GRAY);
        imageGray.convertTo(imagGrayfloat, CV_32F);
        cv::Mat scale_image = imagGrayfloat * (1.0/255.0);

        float32_t first_value = *(float32_t *)scale_image.data;
        printf("first value %f\n", first_value);

        std::cout << "Image size is: " << scale_image.total() * scale_image.elemSize() << std::endl;
        memcpy(interpreter->typed_input_tensor<float32_t>(0), scale_image.data, scale_image.total() * scale_image.elemSize());


    
        /********************************************** Execute Inference *******************************************/
        interpreter->Invoke();
       
        /********************************************** Get Output *******************************************/
        std::cout << "Prediction:  " << interpreter->typed_output_tensor<float32_t>(0)[0] << std::endl; 
        if (interpreter->typed_output_tensor<float32_t>(0)[0] >= 0.0) {
            std::cout << "SAD " << std::endl; 
        } else {
            std::cout << "HAPPY " << std::endl; 
        }

        cv::imshow("Camera",frame);
        cv::waitKey(1);
    }


    return 0;
}