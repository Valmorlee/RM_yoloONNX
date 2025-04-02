//
// Created by valmorx on 25-3-20.
//

#ifndef PARAMS_HPP
#define PARAMS_HPP

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#define BLUE 0
#define RED 1

namespace params {
    // Init Needed 需要初始化的参数
    inline cv::Mat roiImg                        ;

    // Debug Switch 调试开关
    inline bool isDebug                          =   false;

    // Color option 颜色相关
    inline int Enemy_color                       =   BLUE;

    // File Path option 路径相关
    inline std::string Engine_Path               =   "/home/valmorx/CLionProjects/RM_yoloONNX/TensorRT-YOLOX/00x.engine";
    inline std::string Video_Path                =   "/home/valmorx/DeepLearningSource/video.mp4";

    // Engine option 引擎相关
    inline int max_batch_size                    =   1;
    inline int engine_width                      =   320;
    inline int engine_height                     =   320;

    // Camera option 相机相关
    inline int cap_width                         =   1280;
    inline int cap_height                        =   720;
    inline int cap_index                         =   4;  // 摄像头选择
    inline int roi_width                         =   640;
    inline int roi_height                        =   480;
    inline int tracker_frameRate                 =   60; // 追踪器帧率
    inline int tracker_bufferSize                =   30; // 追踪器缓冲区大小


    // Tracker option 追踪器相关
    inline std::vector trackClass                =   { // 有效目标类别标签

        0,1,2,3,4,5,6,7,8,9,10,11,12,13

    };

    // Basic Class Tag // 原始种类标签
    inline std::vector<std::string> class_names  =   {

        "B1","B2","B3","B4","B5","BO","BS",
        "R1","R2","R3","R4","R5","RO","RS"

    };

    //标准yolo输出数据类
    class dataBox {
      public:

        //边界版
        int leftBound;
        int rightBound;
        int topBound;
        int bottomBound;

        //坐标版
        cv::Point leftUp;
        int width;
        int height;

        cv::Point2f centerPoint;

        dataBox(int leftBound, int rightBound, int topBound, int bottomBound) {
            this->leftBound = leftBound;
            this->rightBound = rightBound;
            this->topBound = topBound;
            this->bottomBound = bottomBound;
            this->leftUp = cv::Point(leftBound, topBound);
            this->width = rightBound - leftBound;
            this->height = bottomBound - topBound;
            this->centerPoint = cv::Point2f(leftBound + width / 2.0, topBound + height / 2.0);
        }

        dataBox(cv::Point leftUp, int width, int height) {
            this->leftUp = leftUp;
            this->width = width;
            this->height = height;
            this->leftBound = leftUp.x;
            this->rightBound = leftUp.x + width;
            this->topBound = leftUp.y;
            this->bottomBound = leftUp.y + height;
            this->centerPoint = cv::Point2f(leftUp.x + width / 2.0, leftUp.y + height / 2.0);
        }

        dataBox() = default;

    };


}


#endif //PARAMS_HPP
