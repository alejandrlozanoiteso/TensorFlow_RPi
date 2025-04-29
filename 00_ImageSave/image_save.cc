#include <iostream>
#include <opencv2/opencv.hpp>

#define HEIGHT 480
#define WIDTH   640
#define FPS 30

#define CAMERA  1
#define VLC     2

#define TEST CAMERA

int main(int argc, char * argv[]) {

    struct cv::Mat image;
    struct cv::Mat grayImage;

    /*Folder where images are saved */
    std::string folder = argv[1];
    int n_images = atoi(argv[2]);

    std::cout << "Saving images at folder " <<  folder << std::endl;

    /*Crea Objeto de la Camara */
#if TEST == CAMERA 
    cv::VideoCapture cam(0);
#elif TEST == VLC
    cv::VideoCapture cam("rtsp://192.168.100.14:8554/camera", cv::CAP_FFMPEG);
#endif

    if (cam.isOpened()) {
        std::cout << "cam streaming" << std::endl;
    }
    else {
        std::cout << "Unable to open cam" << std::endl;
        return  -1;
    }
   
#if TEST == CAMERA 
    /*Configura camara */
    cam.set(cv::CAP_PROP_FRAME_HEIGHT, HEIGHT);
    cam.set(cv::CAP_PROP_FRAME_WIDTH, WIDTH);
    cam.set(cv::CAP_PROP_FPS, FPS);
#endif

        for(int i = 0; i < n_images; i ++) {
            /*Obtiene imagen*/
            cam >> image;
            /*Plasma Imagen */
            cv::imshow("Camara", image);
            /*Guarda Imagen */
            cv::imwrite(folder + "/" + folder + std::to_string(i)+".jpg", image);
            cv::waitKey(1);
        }      
    
    return 0;
}