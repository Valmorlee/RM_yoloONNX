//
// Created by valmorx on 25-3-14.
//

#ifndef YOLO_HPP
#define YOLO_HPP

#include "Param.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include<sm_20_atomic_functions.h>
#include <device_launch_parameters.h>

#endif //YOLO_HPP

void resizeUniform(const cv::Mat &src, cv::Mat &resized_img, cv::Size dstSize, object_rect &effect_roi);

std::vector<Info> preProcess(cv::Mat &resized_img, cv::dnn::Net &myNet, Param &param);

std::vector<Info> postProcess(cv::Mat &out, cv::Mat &resized_img, Param &param);

void decode2Box(cv::Mat &out, double threshold, std::vector<std::vector<Info>> &res_infos, Param &param);

void NMS(std::vector<Info> &input_boxes, float NMS_THRESH);

cv::Mat draw_bboxes(const cv::Mat &bgr, std::vector<Info> &bboxes, object_rect &effect_roi, Param &param, std::vector<cv::Point> &AimPoints);//unchecked

cv::Point getCenterPoint(cv::Point pt1, cv::Point pt2);

void drawPoint(const cv::Mat &img, cv::Point2f pt,const cv::Scalar &color);

void decode2BoxCUDA(cv::Mat &out, double threshold, std::vector<std::vector<Info>> &res_infos, Param &param);

__global__ void decode2BoxKernel(float* out, int out_width, int out_height, int num_class, int onnx_width, int onnx_height, double threshold, Info* res_infos, int* res_infos_sizes, int* res_infos_offsets, Param param);