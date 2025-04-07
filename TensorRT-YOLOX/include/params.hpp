//
// Created by valmorx on 25-3-20.
//

#ifndef PARAMS_HPP
#define PARAMS_HPP

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "deploy/model.hpp"  // 包含模型推理相关的类定义
#include "deploy/option.hpp"  // 包含推理选项的配置类定义
#include "deploy/result.hpp"  // 包含推理结果的定义

#define NONE -1
#define BLUE 0
#define RED 1

namespace params {

    // Debug Switch 调试开关
    inline bool isDebug                          =   true;
    inline bool isMonitor                        =   true;

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


}

namespace base {

    inline cv::Mat roiImg; // ROI 图像

    // 待初始化参数
    inline deploy::InferOption option;
    inline unique_ptr<deploy::BaseModel<deploy::DetectRes>> detector;
    inline BYTETracker tracker;
    inline VideoCapture cap;
    inline cv::KalmanFilter kf;
    inline cv::Mat state; // (x, y, vx, vy)
    inline cv::Mat measurement; // (x, y)

    inline chrono::time_point<chrono::system_clock> start;
    inline chrono::time_point<chrono::system_clock> end;

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
        float prob;
        int classId;

        dataBox(int leftBound, int rightBound, int topBound, int bottomBound, float prob, int classId) {
            this->leftBound = leftBound;
            this->rightBound = rightBound;
            this->topBound = topBound;
            this->bottomBound = bottomBound;
            this->leftUp = cv::Point(leftBound, topBound);
            this->width = rightBound - leftBound;
            this->height = bottomBound - topBound;
            this->centerPoint = cv::Point2f(leftBound + width / 2.0, topBound + height / 2.0);
            this->prob = prob;
            this->classId = classId;
        }

        dataBox(cv::Point leftUp, int width, int height, float prob, int classId) {
            this->leftUp = leftUp;
            this->width = width;
            this->height = height;
            this->leftBound = leftUp.x;
            this->rightBound = leftUp.x + width;
            this->topBound = leftUp.y;
            this->bottomBound = leftUp.y + height;
            this->centerPoint = cv::Point2f(leftUp.x + width / 2.0, leftUp.y + height / 2.0);
            this->prob = prob;
            this->classId = classId;
        }

        dataBox(const dataBox &box) {
            this->leftBound = box.leftBound;
            this->rightBound = box.rightBound;
            this->topBound = box.topBound;
            this->bottomBound = box.bottomBound;
            this->leftUp = box.leftUp;
            this->width = box.width;
            this->height = box.height;
            this->centerPoint = box.centerPoint;
            this->prob = box.prob;
            this->classId = box.classId;
        }

        dataBox operator=(const dataBox &box) {
            this->leftBound = box.leftBound;
            this->rightBound = box.rightBound;
            this->topBound = box.topBound;
            this->bottomBound = box.bottomBound;
            this->leftUp = box.leftUp;
            this->width = box.width;
            this->height = box.height;
            this->centerPoint = box.centerPoint;
            this->prob = box.prob;
            this->classId = box.classId;
            return *this;
        }

        dataBox() = default;

    };

    inline int output_color = NONE; // 检测输出颜色 也可用作api
    inline dataBox output_dataBox = dataBox(); // 输出数据
    inline std::vector<dataBox> output_dataBoxes = std::vector<dataBox>(); // 输出群数据

}


#endif //PARAMS_HPP
