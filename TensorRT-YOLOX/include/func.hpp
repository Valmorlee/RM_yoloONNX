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

    bool isTracking(int classId);

    cv::Rect get_centerRect(cv::Point center, int width, int height); // 获取以定点作为矩形中心点的矩形
    std::vector<params::dataBox> revert2Box(const deploy::DetectRes &res);
    std::vector<Object> revert2Tracker(const deploy::DetectRes &res);
    std::vector<params::dataBox> tracker2Box(vector<STrack> &output);

}

namespace func {



}


#endif //FUNC_HPP
