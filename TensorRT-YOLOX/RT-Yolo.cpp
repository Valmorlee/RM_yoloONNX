//
// Created by valmorx on 25-3-15.
//

#include <memory>
#include <opencv2/opencv.hpp>

// 为了方便调用，模块除使用CUDA、TensorRT外，其余均使用标准库实现
#include <func.hpp>

#include "deploy/model.hpp"  // 包含模型推理相关的类定义
#include "deploy/option.hpp"  // 包含推理选项的配置类定义
#include "deploy/result.hpp"  // 包含推理结果的定义

#include <Eigen/Dense>  // Eigen库
#include <ceres/ceres.h> // Ceres库

#include "BYTETracker.h" // ByteTrack
#include "params.hpp" // 参数配置

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);
    std::cout.tie(0);

    // init settings
    deploy::InferOption option;
    option.enableSwapRB();

    // init model & tracker
    auto detector = std::make_unique<deploy::DetectModel>(params::Engine_Path,option);
    BYTETracker tracker(params::tracker_frameRate, params::tracker_bufferSize);

    // Video Load
    cv::VideoCapture cap(params::cap_index, cv::CAP_V4L2);
    cap.set(cv::CAP_PROP_FOURCC,cv::VideoWriter::fourcc('M','J','P','G'));

    if (!cap.isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    cv::Mat input;
    deploy::Image image;
    deploy::DetectRes res;

    long long num_frames = 0;
    long long total_ms = 0;

    while (true) {
        num_frames++;
        cap.read(input);
        flip(input,input,1);

        if (input.empty()) {
            std::cout << "No frame" << std::endl;
            break;
        }

        // test
        params::roiImg = input(tools::get_centerRect(cv::Point(input.cols/2,input.rows/2),params::roi_width,params::roi_height)).clone();
        rectangle(input,tools::get_centerRect(cv::Point(input.cols/2,input.rows/2),params::roi_width,params::roi_height),cv::Scalar(0,0,255),2);
        //cv::imshow("roiImg",params::roiImg);

        // =================== timeNode 1 ===================

        auto start = std::chrono::system_clock::now();

        // ==================================================

        image=deploy::Image(params::roiImg.data,params::roiImg.cols,params::roiImg.rows);

        // inference at RTX 4060 laptop with 0.811989ms
        res = detector->predict(image);

        //变换为 tracker
        std::vector<Object> objects;
        objects = tools::revert2Tracker(res);

        //track update
        std::vector<STrack> output = tracker.update(objects);
        vector<params::dataBox> outBoxes = tools::tracker2Box(output); // 输出数据盒


        // ==================== timeNode 2 ====================

        auto end = std::chrono::system_clock::now();
        total_ms = total_ms + std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // ====== using roi totally in 4060 with 0.741ms ======

        for (int i=0;i<output.size();i++) {
            std::vector<float> _tlwh = output[i].tlwh;
            if (_tlwh[2]*_tlwh[3] > 20) {

                cv::Scalar color = tracker.get_color(output[i].track_id);
                cv::putText(input,
                    cv::format("%d",output[i].track_id),
                    cv::Point(_tlwh[0],_tlwh[1]-5),
                    0,0.6,cv::Scalar(0,0,255),2,LINE_AA);

                cv::Rect preBox = cv::Rect(_tlwh[0],_tlwh[1],_tlwh[2],_tlwh[3]);
                output[i].centerPoint = cv::Point2f((preBox.br().x+preBox.tl().x)/2.0,(preBox.br().y+preBox.tl().y)/2.0);

                circle(input,output[i].centerPoint,6,cv::Scalar(0, 0, 255),-1);

                cv::rectangle(input,
                    preBox,
                    color,2);

            }
        }

        // for (int i=0;i<res.boxes.size();i++) {
        //
        //     // cv::rectangle(input,
        //     //     cv::Point(res.boxes[i].left,res.boxes[i].top),
        //     //     cv::Point(res.boxes[i].right,res.boxes[i].bottom),
        //     //     cv::Scalar(0,0,255),
        //     //     2);
        //
        //     cv::circle(input,res.boxes[i].centerPoint,6,cv::Scalar(0,255,0),-1);
        //
        //     int baseLine=0;
        //     std::string label_text=params::class_names[res.classes[i]]+" "+std::to_string(res.scores[i]*100)+"%";
        //     cv::Size label_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);
        //
        //     putText(input,
        //         label_text,
        //         cv::Point(res.boxes[i].right - label_size.width / 2,res.boxes[i].top - label_size.height / 2),
        //         cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(0,0,255),
        //         1,
        //         cv::LINE_AA
        //     );
        // }

        cv::putText(input,cv::format("frame: %lld fps: %lld num: %lld", num_frames, num_frames * 1000000 / total_ms, output.size()),cv::Point(5,30),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,0,255),2,cv::LINE_AA);

        cv::imshow("result",input);

        int c = cv::waitKey(1);
        if (c==27) break;

    }

}