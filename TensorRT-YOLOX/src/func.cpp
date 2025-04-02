//
// Created by valmorx on 25-4-2.
//

#include "func.hpp"
#include <opencv2/opencv.hpp>
#include "params.hpp"

namespace tools {

    bool isTracking(int classId) {
        for (auto i: params::trackClass) {
            if (classId == i) return true;
        }
        return false;
    }

    cv::Rect get_centerRect(cv::Point center, int width, int height){
        int leftBound = center.x - width / 2 > 0 ? center.x - width / 2 : 0;
        int rightBound = center.x + width / 2 < params::cap_width ? center.x + width / 2 : params::cap_width;
        int upperBound = center.y - height / 2 > 0 ? center.y - height / 2 : 0;
        int lowerBound = center.y + height / 2 < params::cap_height ? center.y + height / 2 : params::cap_height;

        return cv::Rect(leftBound, upperBound, rightBound - leftBound, lowerBound - upperBound);
    }

    std::vector<params::dataBox> revert2Box(const deploy::DetectRes &res) {
        std::vector<params::dataBox> boxes;
        int heightX = params::cap_height / 2 - params::roi_height / 2;
        int widthX = params::cap_width / 2 - params::roi_width / 2;
        for (int i = 0; i < res.boxes.size(); i++) {
            if (tools::isTracking(res.classes[i])) {
                boxes.emplace_back(params::dataBox(res.boxes[i].left + widthX, res.boxes[i].right + widthX, res.boxes[i].top + heightX, res.boxes[i].bottom + heightX));
            }
        }

        return boxes;
    }

    std::vector<Object> revert2Tracker(const deploy::DetectRes &res) {
        std::vector<Object> objects;
        int heightX = params::cap_height / 2 - params::roi_height / 2;
        int widthX = params::cap_width / 2 - params::roi_width / 2;
        for (int i = 0; i < res.boxes.size(); i++) {
            if (tools::isTracking(res.classes[i])) {
                Rect_<float> rect(res.boxes[i].left + widthX,res.boxes[i].top + heightX,res.boxes[i].right-res.boxes[i].left,res.boxes[i].bottom-res.boxes[i].top);
                Object tmp {rect,res.classes[i],res.scores[i]};
                objects.push_back(tmp);
            }
        }
        return objects;
    }

    std::vector<params::dataBox> tracker2Box(vector<STrack> &output) {
        std::vector<params::dataBox> boxes;
        for (int i = 0 ; i < output.size(); i++) {
            std::vector<float> _tlwh = output[i].tlwh;
            if (_tlwh[2]*_tlwh[3] > 20) {
                boxes.emplace_back(
                    params::dataBox(
                        Point2f(_tlwh[0],_tlwh[1]),
                        _tlwh[2],
                        _tlwh[3]
                    )
                );
            }
        }
        return boxes;
    }



}

namespace func {

}
