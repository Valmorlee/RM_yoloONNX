//
// Created by valmorx on 25-3-14.
//

#ifndef PARAM_HPP
#define PARAM_HPP

#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>
#include "sys/time.h"
#include <cuda_runtime.h>


#endif //PARAM_HPP

struct Param {
public:
    Param();
    ~Param();

    std::string media_path;
    std::string onnx_path;
    int onnx_width;
    int onnx_height;

    bool is_delayMonitor;
    bool is_debug;
    bool is_CUDA;


    int num_class;
    float nms_threshold;
    float conf_threshold;
    float PointX; // 总检测框数

    const int color_list[9][3] =
    {
        //{255 ,255 ,255}, //bg
        {216, 82, 24},
        {236, 176, 31},
        {118, 171, 47},
        {76, 189, 237},
        {238, 19, 46},
        {76, 76, 76},
        {153, 153, 153},
        {255, 0, 0},
    };

};

class object_rect {
public:

    object_rect() {
        x = 0;
        y = 0;
        width = 0;
        height = 0;
    }

    ~object_rect() = default;

    int x;
    int y;
    int width;
    int height;

};

struct Info {
public:
    Info() {
        x1 = 0;
        y1 = 0;
        x2 = 0;
        y2 = 0;
        conf = 0;
        label = -1;
    }

    __host__ __device__ Info(float x1, float y1, float x2, float y2, float conf, int label) {
        this->x1 = x1;
        this->y1 = y1;
        this->x2 = x2;
        this->y2 = y2;
        this->conf = conf;
        this->label = label;
    }

    __host__ __device__ void printInfo() {
        printf("x1: %f, y1: %f, x2: %f, y2: %f, conf: %f, label: %d\n", x1, y1, x2, y2, conf, label);
    }

    ~Info() = default;

    float x1;
    float y1;
    float x2;
    float y2;

    float conf;
    int label;
};
