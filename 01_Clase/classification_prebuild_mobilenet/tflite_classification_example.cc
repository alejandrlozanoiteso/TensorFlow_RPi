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

#define TEST  VLC

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
    if (argc != 4)
    {
        fprintf(stderr, "TfliteClassification.exe modelfile labels image\n");
        exit(-1);
    }
    const char *modelFileName = argv[1];
    const char *labelFile = argv[2];
    const char *imageFile = argv[3];

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
    auto channels = interpreter->tensor(input)->dims->data[3];
    auto height = interpreter->tensor(input)->dims->data[1];
    auto width = interpreter->tensor(input)->dims->data[2];

    std::cout << "channels " << channels << std::endl;
    std::cout << "height " << height << std::endl;
    std::cout << "width " << width << std::endl;

    // Load Input Image
    cv::Mat image;

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
    
        /*Resize input */
        cv::resize(frame, image, cv::Size(width, height), cv::INTER_NEAREST);
        cv::Mat imageRGB;
        cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
        memcpy(interpreter->typed_input_tensor<unsigned char>(0), imageRGB.data, imageRGB.total() * imageRGB.elemSize());
    
        /********************************************** Execute Inference *******************************************/
        interpreter->Invoke();
       
        /********************************************** Get Output *******************************************/
        TfLiteIntArray *output_dims = interpreter->tensor(output)->dims;
        auto output_size = output_dims->data[output_dims->size - 1];
        std::cout << "Output Size " << output_size << std::endl;

        /*Get results from priority queue */
        std::vector<std::pair<float, int>> top_results;
        float threshold = 0.01f;
        tflite::label_image::get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0), output_size, 1, threshold, &top_results, kTfLiteUInt8);

        /* Load Labels */
        auto labels = load_labels(labelFile);
    
        /* Print labels with confidence in input image */
        for (const auto &result : top_results)
        {
            const float confidence = result.first;
            const int index = result.second;
            std::string output_txt = "Label :" + labels[index] + " Confidence : " + std::to_string(confidence);
            cv::putText(frame, output_txt, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
            std::cout << output_txt << std::endl;
        }
    
        /*Display image*/
        cv::imshow("Output", frame);
        cv::waitKey(1);
    }


    return 0;
}