//
// Created by valmorx on 25-4-2.
//

#ifndef FUNC_HPP
#define FUNC_HPP

#include <BYTETracker.h>
#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>
#include <deploy/result.hpp>

#include "params.hpp"

namespace tools {
    bool isTracking(int classId);                                                        // 判断是否为追踪目标
    base::dataBox filterBoxes(std::vector<base::dataBox> &boxes);
    void drawRes(cv::Mat &input, base::dataBox &output, Scalar &color);
    void drawRes_tracker(cv::Mat &input, std::vector<STrack> &output);
    cv::Rect get_centerRect(cv::Point2f center, float width, float height);                    // 获取以定点作为矩形中心点的矩形
    std::vector<base::dataBox> revert2Box(const deploy::DetectRes &res);               // yoloBox转换为dataBox | 数据格式转换
    std::vector<Object> revert2Tracker(const deploy::DetectRes &res);                    // yoloBox转换为bytetracker | 数据格式转换
    std::vector<base::dataBox> tracker2Box(vector<STrack> &output);                    // bytetracker转换为dataBox | 数据格式转换

}

namespace func {

    void ioOptimize(); // 输入输出优化

    void init(int selfColor); // 初始化
    void detect(); // 精检测
    void detect_multi(); // 泛检测

    base::dataBox KalmanFilterPre(base::dataBox &box); // kf滤波
    void exPnP(); // epnp解算
    void exKalmanFilterPre(); // ekf拓展

}


#endif //FUNC_HPP
