//
// Created by valmorx on 25-3-15.
//

#include <memory>
#include <opencv2/opencv.hpp>
#include <sys/time.h>

// 为了方便调用，模块除使用CUDA、TensorRT外，其余均使用标准库实现
#include "deploy/model.hpp"  // 包含模型推理相关的类定义
#include "deploy/option.hpp"  // 包含推理选项的配置类定义
#include "deploy/result.hpp"  // 包含推理结果的定义

#include<opencv2/opencv.hpp>

std::string Engine_Path = "/home/valmorx/CLionProjects/RM_yoloONNX/TensorRT-YOLOX/bestx.engine";
std::string Video_Path = "/home/valmorx/DeepLearningSource/video.mp4";

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);
    std::cout.tie(0);

    // init settings
    deploy::InferOption option;
    option.enableSwapRB();

    // init model
    auto detector = std::make_unique<deploy::DetectModel>(Engine_Path,option);

    // Video Load
    cv::VideoCapture cap(0,cv::CAP_V4L2);
    cap.set(cv::CAP_PROP_FOURCC,cv::VideoWriter::fourcc('M','J','P','G'));

    if (!cap.isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    cv::Mat input;
    deploy::Image image;
    deploy::DetectRes res;

    long long script = 0;
    long double scriptTime = 0;

    while (true) {
        script++;
        cap.read(input);

        if (input.empty()) {
            std::cout << "No frame" << std::endl;
            break;
        }

        // ===== timeNode 1 =====
        timeval t1;
        gettimeofday(&t1, NULL);
        // ======================

        image=deploy::Image(input.data,input.cols,input.rows);

        // inference at RTX 4060 laptop with 0.811989ms
        res = detector->predict(image);

        // ===== timeNode 2 =====
        timeval t2;
        gettimeofday(&t2, NULL);
        scriptTime += (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000.0;
        // std::cout<<"inferTime: "<<  (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000.0 << "ms" << std::endl;
        // ======================

        for (int i=0;i<res.boxes.size();i++) {

            cv::rectangle(input,
                cv::Point(res.boxes[i].left,res.boxes[i].top),
                cv::Point(res.boxes[i].right,res.boxes[i].bottom),
                cv::Scalar(0,0,255),
                2);

        }

        //cv::imshow("result",input);
        if (script == 1000) break;
        int c = cv::waitKey(1);
        if (c==27) break;

    }
    std::cout<<"AveScriptTime: "<<1.0 * scriptTime / script<<"ms"<<std::endl;

}