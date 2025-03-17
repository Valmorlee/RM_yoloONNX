//
// Created by valmorx on 25-3-14.
//

#include "Param.hpp"

#define BLUE 0
#define RED 1

Param::Param() {

    media_path = "/home/valmorx/DeepLearningSource/video.mp4";
    onnx_path = "/home/valmorx/DeepLearningSource/ultralytics-8.3.91/runs/detect/train7/weights/best.onnx";
    onnx_width = 320;
    onnx_height = 320;
    PointX = 0;

    num_class = 14; // 识别类别数量
    nms_threshold = 0.4;
    conf_threshold = 0.7;

    is_delayMonitor = true;
    is_debug = false; //调试模式 unrealized
    is_CUDA = true;

}

Param::~Param() = default;
