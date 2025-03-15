//
// Created by valmorx on 25-3-14.
//

#ifndef ARMORDETECTOR_HPP
#define ARMORDETECTOR_HPP

#include<bits/stdc++.h>
#include "Yolo.hpp"

#endif //ARMORDETECTOR_HPP


#define BLUE 0
#define RED 1

class ArmorDetector {
  public:

    ArmorDetector();
    ~ArmorDetector();

    void init(int self_color); // 初始化函数
    void loadImg(); // 加载图像设置
    void detect(); // 推理

    Param param;
    cv::Mat display_img;
    int self_color;
    int enemy_color;

    float calDelay();

    cv::dnn::Net myNet;
    cv::VideoCapture cap;
    int scriptCount;
    timeval startTime;
    std::vector<cv::Point> AimPoints;


};