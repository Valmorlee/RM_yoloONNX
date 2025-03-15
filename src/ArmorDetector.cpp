//
// Created by valmorx on 25-3-14.
//

#include "ArmorDetector.hpp"


ArmorDetector::ArmorDetector(){

}

ArmorDetector::~ArmorDetector() {

}

void ArmorDetector::init(int self_color) {

    gettimeofday(&startTime,NULL);

    if (self_color == BLUE) {
        this->self_color = BLUE;
        this->enemy_color = RED;
    }else {
        this->self_color = RED;
        this->enemy_color = BLUE;
    }
    std::cout << "Machine Init Complete!" << std::endl;
}

void ArmorDetector::loadImg() {
    cap.open(param.media_path);
    cap.set(cv::CAP_PROP_FOURCC,cv::VideoWriter::fourcc('M','J','P','G'));

    if (!cap.isOpened()) {
        std::cout << "Can't open video source" << std::endl;
        return;
    }

    std::string model_path = param.onnx_path;

    this->myNet = cv::dnn::readNetFromONNX(model_path); // 加载模型

    if (myNet.empty()) {
        std::cout<<"Empty ONNX Net"<<std::endl;
        return;
    }

    std::cout << "Img & Net Load Complete!" << std::endl;
    if (param.is_delayMonitor) {
        printf("load timeCost: %f ms\n", calDelay());
    }

}

//aim points clear
void ArmorDetector::detect() {
    cv::Mat input;

    while (true) {
        scriptCount++;
        cap.read(input);

        if (input.empty()) {
            std::cout << "input empty" << std::endl;
            break;
        }

        //yolo_detect
        cv::Size dstSize = cv::Size(param.onnx_width, param.onnx_height);
        object_rect effect_roi;
        cv::Mat resized_img = input.clone();

        resizeUniform(input, resized_img, dstSize, effect_roi); // 统一缩放
        auto results = preProcess(resized_img, myNet, param); // 前后处理以及推理
        cv::Mat final = draw_bboxes(input, results, effect_roi, param, AimPoints); // 绘制框

        //接入


        cv::imshow("final", final);

        int c = cv::waitKey(30);
        if (c==27) break;

        if (param.is_delayMonitor&&scriptCount % 100 == 0) {
            printf("Ave Delay: %f ms\n", calDelay() / 100.0);
        }

        AimPoints.clear();

    }

    cap.release();
}

float ArmorDetector::calDelay() {
    timeval endTime;
    gettimeofday(&endTime,NULL);
    float interval = (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_usec - startTime.tv_usec) / 1000000.0;
    startTime = endTime;

    return interval*1000;
}






