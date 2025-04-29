#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/string_util.h>
#include <tensorflow/lite/examples/label_image/get_top_n.h>
#include <tensorflow/lite/model.h>


int main(int argc, char **argv)
{
    const char *modelFileName = argv[1];
    // Load Model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(modelFileName);
    if (model == nullptr) {
        fprintf(stderr, "failed to load model\n");
        exit(-1);
    }
    // Initiate Interpreter
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);
    if (interpreter == nullptr) {
        fprintf(stderr, "Failed to initiate the interpreter\n");
        exit(-1);
    }

    if (interpreter->AllocateTensors() != kTfLiteOk){
        fprintf(stderr, "Failed to allocate tensor\n");
        exit(-1);
    }

    // Optional Configure the interpreter
    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(1);

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
    float32_t input_val;
    sscanf(argv[2], "%f", &input_val);
    // Copy data to input tensor
    memcpy(interpreter->typed_input_tensor<float32_t>(0), &input_val, sizeof(float32_t));
    

    /********************************************** Execute Inference *******************************************/
    interpreter->Invoke();

    /********************************************** Get Output *******************************************/
    std::cout << "/**********************************************Prediction *******************************************/" << std::endl;
    std::cout << "Prediction:  " << interpreter->typed_output_tensor<float32_t>(0)[0] << std::endl; 

    return 0;
}