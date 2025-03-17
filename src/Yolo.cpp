//
// Created by valmorx on 25-3-14.
//

#include "Yolo.hpp"

void resizeUniform(const cv::Mat &src, cv::Mat &resized_img, cv::Size dstSize, object_rect &effect_roi) {

    int w = src.cols;
    int h = src.rows;
    int dstW = dstSize.width;
    int dstH = dstSize.height;

    resized_img = cv::Mat(cv::Size(dstW, dstH), CV_8UC3, cv::Scalar(0, 0, 0));

    double ratio_src = w * 1.0 / h;
    double ratio_dst = dstW * 1.0 / dstH;

    int tmp_w = 0, tmp_h = 0;

    if (ratio_src > ratio_dst) {
        tmp_w = dstW;
        tmp_h = floor(1.0 * dstW / ratio_src);
    }else if(ratio_src < ratio_dst) {
        tmp_h = dstH;
        tmp_w = floor(1.0 * dstH * ratio_src);
    }else {
        cv::resize(src, resized_img, dstSize);
        effect_roi.x = 0;
        effect_roi.y = 0;
        effect_roi.width = dstW;
        effect_roi.height = dstH;

    }

    cv::Mat tmp;
    cv::resize(src, tmp, cv::Size(tmp_w, tmp_h));
    if (tmp_w != dstW)
    {
        int index_w = floor((dstW - tmp_w) / 2.0);
        for (int i = 0; i < dstH; i++)
        {
            memcpy(resized_img.data + i * dstW * 3 + index_w * 3, tmp.data + i * tmp_w * 3, tmp_w * 3);
            //像素重叠复制
        }
        effect_roi.x = index_w;
        effect_roi.y = 0;
        effect_roi.width = tmp_w;
        effect_roi.height = tmp_h;
    }
    else if (tmp_h != dstH)
    {
        int index_h = floor((dstH - tmp_h) / 2.0);

        memcpy(resized_img.data + index_h * dstW * 3, tmp.data, tmp_w * tmp_h * 3);
        effect_roi.x = 0;       // 实际图像区域在dst中x轴的相对坐标是 0
        effect_roi.y = index_h; // 实际图像区域在dst中y轴的相对坐标是 index_h
        effect_roi.width = tmp_w;
        effect_roi.height = tmp_h;
    }
    else
    {
        std::cout<<"error: resizeUniform"<<std::endl;
    }

}


void decode2Box(cv::Mat &out, double threshold, std::vector<std::vector<Info>> &res_infos, Param &param) {
    param.PointX = out.total() / (param.num_class+4); // 2100个检测框

    for (int i=0;i<param.PointX;i++) {

        float conf = 0;
        int label_max_conf = 0;

        for (int index=4;index<param.num_class+4;index++) {

            float tmp = out.at<float>(0,index,i);
            //std::cout << out.at<float>(0,index,i)<<" ";

            if (tmp > conf) {
                conf = tmp;
                label_max_conf = index - 4;
            }
        }
        //std::cout<<std::endl;

        if (conf > threshold) {

            float x = out.at<float>(0,0,i);
            float y = out.at<float>(0,1,i);
            float w = out.at<float>(0,2,i);
            float h = out.at<float>(0,3,i);

            float x1 = x-w/2.0 < 0 ? 0 : x-w/2.0;//左上
            float y1 = y-h/2.0 < 0 ? 0 : y-h/2.0;
            float x2 = x+w/2.0 > param.onnx_width ? param.onnx_width : x+w/2.0;//右下
            float y2 = y+h/2.0 > param.onnx_height ? param.onnx_height : y+h/2.0;

            auto tmp = Info(x1, y1, x2, y2, conf, label_max_conf);
            res_infos[label_max_conf].emplace_back(tmp);
        }
    }
}

std::vector<Info> preProcess(cv::Mat &resized_img, cv::dnn::Net &myNet, Param &param) {

    cv::Mat blob = cv::dnn::blobFromImage(resized_img, 1.0/255.0, cv::Size(param.onnx_width, param.onnx_height), cv::Scalar(0, 0, 0), true, false);



    myNet.setInput(blob);
    std::vector<cv::Mat> outs;
    myNet.forward(outs, myNet.getUnconnectedOutLayersNames());

    return postProcess(outs[0],resized_img,param);
}

std::vector<Info> postProcess(cv::Mat &out, cv::Mat &resized_img, Param &param) {

    std::vector<std::vector<Info>> res;
    res.resize(param.PointX+5);

    if (param.is_CUDA) decode2BoxCUDA(out, param.conf_threshold, res, param); // 等待实现
    else decode2Box(out, param.conf_threshold, res, param);

    std::vector<Info> res_infos;
    for (int i = 0 ; i < res.size(); i++) {
        NMS(res[i], param.nms_threshold);
        for (auto box : res[i]) {
            box.printInfo();
            res_infos.emplace_back(box);
        }
    }
    return res_infos;
}

void NMS(std::vector<Info> &input_boxes, float NMS_THRESH)
{
    std::sort(input_boxes.begin(), input_boxes.end(), [](Info a, Info b)
              { return a.conf > b.conf; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}

cv::Mat draw_bboxes(const cv::Mat &bgr, std::vector<Info> &bboxes, object_rect &effect_roi, Param &param, std::vector<cv::Point> &AimPoints) //unchecked
{
    static const char *class_names[] = {"blue_Armor", "blue_Car"};

    cv::Mat image = bgr.clone();

    const int src_w = image.cols;
    const int src_h = image.rows;
    const int dst_w = effect_roi.width;
    const int dst_h = effect_roi.height;
    const float width_ratio = static_cast<float>(src_w) / static_cast<float>(dst_w);
    const float height_ratio = static_cast<float>(src_h) / static_cast<float>(dst_h);

    for (size_t i = 0; i < bboxes.size(); i++)
    {
        Info &bbox = bboxes[i];

        cv::Scalar color = cv::Scalar(param.color_list[bbox.label][0], param.color_list[bbox.label][1], param.color_list[bbox.label][2]);

        cv::rectangle(image, cv::Rect(cv::Point((bbox.x1 - effect_roi.x) * width_ratio, (bbox.y1 - effect_roi.y) * height_ratio), cv::Point((bbox.x2 - effect_roi.x) * width_ratio, (bbox.y2 - effect_roi.y) * height_ratio)), color, 2);

        cv::Point YOLO_Point = getCenterPoint(cv::Point((bbox.x1 - effect_roi.x) * width_ratio, (bbox.y1 - effect_roi.y) * height_ratio),cv::Point((bbox.x2 - effect_roi.x) * width_ratio, (bbox.y2 - effect_roi.y) * height_ratio));
        AimPoints.emplace_back(YOLO_Point); //每轮要clear

        drawPoint(image,YOLO_Point,color);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[bbox.label], bbox.conf * 100); //

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (bbox.x1 - effect_roi.x) * width_ratio;
        int y = (bbox.y1 - effect_roi.y) * height_ratio - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      color, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    }

    return image;
}

cv::Point getCenterPoint(cv::Point pt1, cv::Point pt2) {
    cv::Point centerPt;

    centerPt.x = (pt1.x + pt2.x) / 2;
    centerPt.y = (pt1.y + pt2.y) / 2;

    return centerPt;

}

void drawPoint(const cv::Mat &img, cv::Point2f pt,const cv::Scalar &color) {
    cv::circle(img, pt, 5, color, -1);
}