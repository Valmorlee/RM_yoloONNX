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

    cv::Rect get_centerRect(cv::Point2f center, float width, float height){
        float leftBound = center.x - width / 2.0 > 0 ? center.x - width / 2.0 : 0;
        float rightBound = center.x + width / 2.0 < params::cap_width ? center.x + width / 2.0 : params::cap_width;
        float upperBound = center.y - height / 2.0 > 0 ? center.y - height / 2.0 : 0;
        float lowerBound = center.y + height / 2.0 < params::cap_height ? center.y + height / 2.0 : params::cap_height;

        return cv::Rect(leftBound, upperBound, rightBound - leftBound, lowerBound - upperBound);
    }

    base::dataBox filterBoxes(std::vector<base::dataBox> &boxes) {
        if (boxes.empty()) return {};
        sort(boxes.begin(), boxes.end(), [](base::dataBox &a, base::dataBox &b) {return a.prob > b.prob;});
        return boxes[0];
    }

    void drawRes_tracker(cv::Mat &input, std::vector<STrack> &output) {
        for (int i=0;i<output.size();i++) {
            std::vector<float> _tlwh = output[i].tlwh;
            if (_tlwh[2]*_tlwh[3] > 20) {

                cv::Scalar color = base::tracker.get_color(output[i].track_id);
                cv::putText(input,
                    cv::format("%d",output[i].track_id),
                    cv::Point(_tlwh[0],_tlwh[1]-5),
                    0,0.6,cv::Scalar(0,0,255),2,LINE_AA);

                cv::Rect preBox = cv::Rect(_tlwh[0],_tlwh[1],_tlwh[2],_tlwh[3]);
                output[i].centerPoint = cv::Point2f((preBox.br().x+preBox.tl().x)/2.0,(preBox.br().y+preBox.tl().y)/2.0);

                circle(input,output[i].centerPoint,6,cv::Scalar(0, 0, 255),-1);

                cv::rectangle(input,
                    preBox,
                    color,1);

            }
        }

    }

    void drawRes(cv::Mat &input, base::dataBox &output, const Scalar &color) {
        cv::putText(input,
            cv::format("%d",output.classId),
            cv::Point2f(output.leftBound,output.topBound-5),
            0,0.6,color,2,LINE_AA);

        Rect2f preBox = cv::Rect2f(output.leftBound,output.topBound,output.width,output.height);
        circle(input,output.centerPoint,6,color,-1);
        cv::rectangle(input,
            preBox,
            color,1);
    }

    std::vector<base::dataBox> revert2Box(const deploy::DetectRes &res) {
        std::vector<base::dataBox> boxes;
        float heightX = params::cap_height / 2.0 - params::roi_height / 2.0;
        float widthX = params::cap_width / 2.0 - params::roi_width / 2.0;
        for (int i = 0; i < res.boxes.size(); i++) {
            if (tools::isTracking(res.classes[i])) {
                boxes.emplace_back(base::dataBox(res.boxes[i].left + widthX, res.boxes[i].right + widthX, res.boxes[i].top + heightX, res.boxes[i].bottom + heightX, res.scores[i], res.classes[i]));
            }
        }

        return boxes;
    }

    std::vector<Object> revert2Tracker(const deploy::DetectRes &res) {
        std::vector<Object> objects;
        float heightX = params::cap_height / 2.0 - params::roi_height / 2.0;
        float widthX = params::cap_width / 2.0 - params::roi_width / 2.0;
        for (int i = 0; i < res.boxes.size(); i++) {
            if (tools::isTracking(res.classes[i])) {
                Rect_<float> rect(res.boxes[i].left + widthX,res.boxes[i].top + heightX,res.boxes[i].right-res.boxes[i].left,res.boxes[i].bottom-res.boxes[i].top);
                Object tmp {rect,res.classes[i],res.scores[i]};
                objects.push_back(tmp);
            }
        }
        return objects;
    }

    std::vector<base::dataBox> tracker2Box(vector<STrack> &output) {
        std::vector<base::dataBox> boxes;
        for (int i = 0 ; i < output.size(); i++) {
            std::vector<float> _tlwh = output[i].tlwh;
            if (_tlwh[2]*_tlwh[3] > 20) {
                boxes.emplace_back(
                    base::dataBox(
                        Point2f(_tlwh[0],_tlwh[1]),
                        _tlwh[2],
                        _tlwh[3],
                        output[i].score,
                        output[i].track_id
                    )
                );
            }
        }
        return boxes;
    }

}

namespace func {

    void init(int selfColor) {

        base::option.enableSwapRB();
        base::detector = std::make_unique<deploy::BaseModel<deploy::DetectRes>>(params::Engine_Path,base::option);

        base::tracker = BYTETracker(params::tracker_frameRate, params::tracker_bufferSize);

        base::cap = VideoCapture(params::cap_index, cv::CAP_V4L2);
        base::cap.set(cv::CAP_PROP_FOURCC,cv::VideoWriter::fourcc('M','J','P','G'));

        if (!base::cap.isOpened()) {
            std::cout << "Error opening video stream or file | init Failed" << std::endl;
            return;
        }

        if (selfColor == RED) {
            params::trackClass = {0,1,2,3,4,5,6};
        }else if (selfColor == BLUE){
            params::trackClass = {7,8,9,10,11,12,13};
        }else {
            std::cout << "Error input selfColor | init Failed" << std::endl;
            return;
        }

        base::output_color = selfColor;

        base::kf = KalmanFilter(4,2,0);
        base::kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0,
                                                            0, 1, 0, 1,
                                                            0, 0, 1, 0,
                                                            0, 0, 0, 1);
        base::kf.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0,
                                                              0, 1, 0, 0);
        base::kf.processNoiseCov = (cv::Mat_<float>(4, 4) << 1e-2, 0, 0, 0,
                                                            0, 1e-2, 0, 0,
                                                            0, 0, 1, 0,
                                                            0, 0, 0, 1) * 1e-2;
        base::kf.measurementNoiseCov = (cv::Mat_<float>(2, 2) << 1e-1, 0,
                                                            0, 1e-1) * 1e-1;
        base::kf.errorCovPost = cv::Mat::eye(4, 4, CV_32F);
        base::state = cv::Mat::zeros(4, 1, CV_32F);
        base::measurement = cv::Mat::zeros(2, 1, CV_32F);

    }

    void detect() { // 空数据盒处理待解决

        cv::Mat input;
        deploy::Image image;
        deploy::DetectRes res;

        long long num_frames = 0;
        long long total_ms = 0;

        while (true) {
            num_frames++;
            base::cap.read(input);
            flip(input,input,1); // 翻转图像

            if (input.empty()) {
                std::cout << "No frame" << std::endl;
                break;
            }

            base::roiImg = input(tools::get_centerRect(cv::Point2f(input.cols/2.0,input.rows/2.0),params::roi_width,params::roi_height)).clone();

            if (params::isDebug) {
                rectangle(input,tools::get_centerRect(cv::Point2f(input.cols/2.0,input.rows/2.0),params::roi_width,params::roi_height),cv::Scalar(0,0,255),2);
            }

            if (params::isMonitor) {
                // =================== timeNode 1 ===================
                base::start = std::chrono::system_clock::now();
                // ==================================================
            }

            image=deploy::Image(base::roiImg.data,base::roiImg.cols,base::roiImg.rows);
            res = base::detector->predict(image);

            vector<base::dataBox> outBoxes = tools::revert2Box(res); // 输出数据
            base::dataBox output = tools::filterBoxes(outBoxes);
            base::dataBox preX = KalmanFilterPre(output);

            base::output_dataBox = preX; // 最终输出结果

            if (params::isMonitor) {
                // ==================== timeNode 2 ====================
                base::end = std::chrono::system_clock::now();
                total_ms = total_ms + std::chrono::duration_cast<std::chrono::microseconds>(base::end - base::start).count();
                // ====== using roi totally in 4060 with 0.741ms ======
            }

            if (params::isDebug) {
                if (output.ExistTag) tools::drawRes(input,output, Scalar(0,0,255));
                if (output.ExistTag) tools::drawRes(input, preX, Scalar(0,255,0));
                if (params::isMonitor) cv::putText(input,cv::format("frame: %lld fps: %lld", num_frames, num_frames * 1000000 / total_ms),cv::Point(5,30),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,0,255),2,cv::LINE_AA);
                cv::imshow("result",input);

                int c = cv::waitKey(1);
                if (c==27) break;
            }

        }
    }

    void detect_multi() {

        cv::Mat input;
        deploy::Image image;
        deploy::DetectRes res;

        long long num_frames = 0;
        long long total_ms = 0;

        while (true) {
            num_frames++;
            base::cap.read(input);
            flip(input,input,1); // 翻转图像

            if (input.empty()) {
                std::cout << "No frame" << std::endl;
                break;
            }

            base::roiImg = input(tools::get_centerRect(cv::Point(input.cols/2,input.rows/2),params::roi_width,params::roi_height)).clone();

            if (params::isDebug) {
                rectangle(input,tools::get_centerRect(cv::Point(input.cols/2,input.rows/2),params::roi_width,params::roi_height),cv::Scalar(0,0,255),2);
            }

            if (params::isMonitor) {
                // =================== timeNode 1 ===================
                base::start = std::chrono::system_clock::now();
                // ==================================================
            }

            image=deploy::Image(base::roiImg.data,base::roiImg.cols,base::roiImg.rows);
            res = base::detector->predict(image);

            std::vector<Object> objects;
            objects = tools::revert2Tracker(res);

            std::vector<STrack> output = base::tracker.update(objects);
            vector<base::dataBox> outBoxes = tools::tracker2Box(output); // 输出数据盒

            base::output_dataBoxes = outBoxes; // 最终输出结果

            if (params::isMonitor) {
                // ==================== timeNode 2 ====================
                base::end = std::chrono::system_clock::now();
                total_ms = total_ms + std::chrono::duration_cast<std::chrono::microseconds>(base::end - base::start).count();
                // ====== using roi totally in 4060 with 0.741ms ======
            }

            tools::drawRes_tracker(input,output);
            if (params::isDebug) {
                if (params::isMonitor) cv::putText(input,cv::format("frame: %lld fps: %lld num: %lld", num_frames, num_frames * 1000000 / total_ms, output.size()),cv::Point(5,30),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,0,255),2,cv::LINE_AA);
                cv::imshow("result",input);

                int c = cv::waitKey(1);
                if (c==27) break;
            }

        }
    }

    base::dataBox KalmanFilterPre(base::dataBox &box) {
        // 非跟踪目标直接返回空数据盒
        if (!box.ExistTag) return {};

        base::measurement.at<float>(0) = box.centerPoint.x;
        base::measurement.at<float>(1) = box.centerPoint.y;
        //cout<<box.centerPoint.x<<" "<<box.centerPoint.y<<endl;

        base::kf.correct(base::measurement);

        base::state = base::kf.predict();
        float predict_x = base::state.at<float>(0);
        float predict_y = base::state.at<float>(1);
        // cout<<"predict_x:"<<predict_x<<" predict_y:"<<predict_y<<endl;

        Rect x = tools::get_centerRect(cv::Point2f(predict_x,predict_y),box.width,box.height);

        return base::dataBox(x.tl(),x.width,x.height,box.prob,box.classId);
    }

    void ioOptimize() {
        std::ios::sync_with_stdio(false);
        std::cin.tie(0);
        std::cout.tie(0);
    }

}
