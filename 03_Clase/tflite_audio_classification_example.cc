#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/interpreter_builder.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/string_util.h>
#include <tensorflow/lite/examples/label_image/get_top_n.h>
#include <tensorflow/lite/model.h>
#include <alsa/asoundlib.h>
#include "fftw3.h"

#define CHANNELS    1
#define TIME_S      2
#define RATE        16000
#define FRAMES      (TIME_S * RATE)
#define MICS    1


const double pi = 3.14159265358979323846;

float * stft(fftw_complex *input, int n, int window_size, int hop_size) {
    int num_frames = (n - window_size) / hop_size + 1;
    fftw_complex *window = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * window_size);
    
    float  * spectro = (float *)malloc(num_frames * (window_size/2 +1) * sizeof(float));

    fftw_plan plan = fftw_plan_dft_1d(window_size, window, window, FFTW_FORWARD, FFTW_ESTIMATE);

    int sample = 0;
    for (int i = 0; i < num_frames; i++) {
        int offset = i * hop_size;
        for (int j = 0; j < window_size; j++) {
            window[j][0] = input[offset + j][0] * 0.5 * (1 - cos(2 * pi * j / (window_size - 1)));
            window[j][1] = 0;
        }

        fftw_execute(plan);
        for (int j = 0; j <= window_size/2; j++) {
            spectro[sample] = sqrt(window[j][0] * window[j][0] + window[j][1] *  window[j][1]);
            sample++;      
        }

    }

    fftw_destroy_plan(plan);
    return spectro;
}

std::vector<std::string> load_labels(std::string labels_file)
{
    std::ifstream file(labels_file.c_str());
    if (!file.is_open()) {
        fprintf(stderr, "unable to open label file\n");
        exit(-1);
    }
    std::string label_str;
    std::vector<std::string> labels;

    while (std::getline(file, label_str)) {
        if (label_str.size() > 0)
            labels.push_back(label_str);
    }
    file.close();
    return labels;
}

int main(int argc, char **argv)
{

    const int n = 32000;
    const int window_size = 512;
    const int hop_size = 128;
    fftw_complex * audio_input = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * n);

    const char *modelFileName = argv[1];
    const char *labelFile = argv[2];

    int size = CHANNELS * FRAMES * sizeof(int16_t);
    int16_t * buffer = (int16_t * )malloc(size);
    memset(buffer, 0, size);

 #ifdef MICS   

    //Configure Mics
    snd_pcm_t * handle;
    snd_pcm_hw_params_t * hw_params; 

    /*Open Sound Card*/
    int ret = snd_pcm_open(&handle, "default", SND_PCM_STREAM_CAPTURE, 0);

    /*Configure Format, Rate, Channels*/
    snd_pcm_hw_params_alloca(&hw_params);
    ret = snd_pcm_hw_params_any(handle, hw_params);


    if( (ret = snd_pcm_hw_params_set_access(handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0) {
        printf("ERROR! Cannot set interleaved mode\n");
        return ret;
    }

    snd_pcm_format_t format = SND_PCM_FORMAT_S16_LE;

    if( (ret = snd_pcm_hw_params_set_format(handle, hw_params, format)) < 0) {
        printf("ERROR! Cannot set format\n");
        return ret;
    }

    int channels = CHANNELS;

    if( (ret = snd_pcm_hw_params_set_channels(handle, hw_params, channels)) < 0) {
        printf("ERROR! Cannot set Channels\n");
        return ret;
    }

    uint32_t rate = RATE;
    if( (ret = snd_pcm_hw_params_set_rate_near(handle, hw_params, &rate, 0)) < 0) {
        printf("ERROR! Cannot set Rate %d\n", rate);
        return ret;
    }

    
    if( (ret = snd_pcm_hw_params(handle, hw_params)) < 0) {
        printf("ERROR! Cannot set hw params\n");
        return ret;
    }

    //Read from Mics
    printf("Speak Now!!!\n");
    snd_pcm_sframes_t frames = snd_pcm_readi(handle, buffer, FRAMES);


#else
    FILE * rec_file = fopen(argv[3], "r");
    printf("Opening %s\n", argv[3]);
    fseek(rec_file, 0L, SEEK_END);
    int file_size = ftell(rec_file);
    fseek(rec_file, 44L, SEEK_SET);

    int n_bytes = 0;
    uint8_t * buff = (uint8_t *)buffer;

    printf("size %d file_size %d\n", size, file_size);
    printf("Reading file\n");
    while (!feof(rec_file)) {
        n_bytes = fread(buff, 1, 1000, rec_file);
        buff += n_bytes;
    }
#endif

    // Generate complex input signal
    for (int i = 0; i < n; i++) {
        audio_input[i][0] = ((float)buffer[i])/32768.0;
        audio_input[i][1] = 0.0;
    }

    //Calculate Spectrogram
    float * spectro = stft(audio_input, n, window_size, hop_size);
    int num_frames = (n - window_size) / hop_size + 1;

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

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        fprintf(stderr, "Failed to allocate tensor\n");
        exit(-1);
    }

    // Configure the interpreter
    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(1);
    // Get Input Tensor Dimensions
    int input = interpreter->inputs()[0];
    auto x = interpreter->tensor(input)->dims->data[1];
    auto y = interpreter->tensor(input)->dims->data[2];
    std::cout << "Input Dimensions : " << x << "," << y << std::endl;

    //Copy Data
    memcpy(interpreter->typed_input_tensor<float>(0), spectro, sizeof(float)* (window_size/2 + 1) * num_frames );
    // Inference
    interpreter->Invoke();
 
    // Get Output
    int output = interpreter->outputs()[0];
    TfLiteIntArray *output_dims = interpreter->tensor(output)->dims;
    auto output_size = output_dims->data[output_dims->size - 1];
    std::cout << "Output Size " << output_size << std::endl;
    std::vector<std::pair<float, int>> top_results;
    float threshold = 0.01f;

    std::cout << "Output Type " << interpreter->tensor(output)->type << std::endl;
    switch (interpreter->tensor(output)->type) {
    case kTfLiteFloat32:
        tflite::label_image::get_top_n<float>(interpreter->typed_output_tensor<float>(0), output_size, 1, threshold, &top_results, kTfLiteFloat32);
        break;
    case kTfLiteUInt8:
        tflite::label_image::get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0), output_size, 1, threshold, &top_results, kTfLiteUInt8);
        break;
    default:
        fprintf(stderr, "cannot handle output type\n");
        exit(-1);
    }
   
    // Load Labels
    auto labels = load_labels(labelFile);

    // Print labels with confidence
    for (const auto &result : top_results) {
        const float confidence = result.first;
        const int index = result.second;
        std::cout << "Label :" + labels[index] + " Confidence : " + std::to_string(confidence) << std::endl;
    }

    fftw_free(audio_input);
    return 0;
}