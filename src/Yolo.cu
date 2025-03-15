//
// Created by valmorx on 25-3-15.
//

#include "Yolo.hpp"

__global__ void decode2BoxKernel(float* out, int out_width, int out_height, int num_class, int onnx_width, int onnx_height, double threshold, Info* res_infos, int* res_infos_sizes, int* res_infos_offsets, Param param) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= param.PointX) return;

    float conf = 0;
    int label_max_conf = 0;

    for (int index = 4; index < num_class + 4; index++) {
        float tmp = out[idx * out_width + index];
        if (tmp > conf) {
            conf = tmp;
            label_max_conf = index - 4;
        }
    }

    if (conf > threshold) {
        float x = out[idx * out_width + 0];
        float y = out[idx * out_width + 1];
        float w = out[idx * out_width + 2];
        float h = out[idx * out_width + 3];

        float x1 = x - w / 2.0f < 0 ? 0 : x - w / 2.0f;
        float y1 = y - h / 2.0f < 0 ? 0 : y - h / 2.0f;
        float x2 = x + w / 2.0f > onnx_width ? onnx_width : x + w / 2.0f;
        float y2 = y + h / 2.0f > onnx_height ? onnx_height : y + h / 2.0f;

        int offset = atomicAdd(&res_infos_offsets[label_max_conf], 1);
        int linear_index = offset + label_max_conf * param.PointX;
        Info(x1, y1, x2, y2, conf, label_max_conf).printInfo();
        res_infos[linear_index] = Info(x1, y1, x2, y2, conf, label_max_conf);
    }
}

void decode2BoxCUDA(cv::Mat &out, double threshold, std::vector<std::vector<Info>> &res_infos, Param &param) {
    param.PointX = out.total() / (param.num_class + 4);

    // Convert cv::Mat to cv::cuda::GpuMat
    cv::cuda::GpuMat d_out;
    d_out.upload(out);

    // Allocate device memory for res_infos
    int max_boxes_per_class = param.PointX;
    Info* d_res_infos;
    cudaMalloc(&d_res_infos, max_boxes_per_class * param.num_class * sizeof(Info));

    int* d_res_infos_sizes;
    cudaMalloc(&d_res_infos_sizes, param.num_class * sizeof(int));
    cudaMemset(d_res_infos_sizes, 0, param.num_class * sizeof(int));

    int* d_res_infos_offsets;
    cudaMalloc(&d_res_infos_offsets, param.num_class * sizeof(int));
    cudaMemset(d_res_infos_offsets, 0, param.num_class * sizeof(int));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (param.PointX + threadsPerBlock - 1) / threadsPerBlock;
    decode2BoxKernel<<<blocksPerGrid, threadsPerBlock>>>(d_out.ptr<float>(), out.cols, out.rows, param.num_class, param.onnx_width, param.onnx_height, threshold, d_res_infos, d_res_infos_sizes, d_res_infos_offsets, param);

    // Copy results back to host
    int* h_res_infos_sizes = new int[param.num_class];
    cudaMemcpy(h_res_infos_sizes, d_res_infos_sizes, param.num_class * sizeof(int), cudaMemcpyDeviceToHost);

    Info* h_res_infos = new Info[max_boxes_per_class * param.num_class];
    cudaMemcpy(h_res_infos, d_res_infos, max_boxes_per_class * param.num_class * sizeof(Info), cudaMemcpyDeviceToHost);

    // Resize res_infos
    for (int i = 0; i < param.num_class; i++) {
        res_infos[i].resize(h_res_infos_sizes[i]);
        for (int j = 0; j < h_res_infos_sizes[i]; j++) {
            res_infos[i][j] = h_res_infos[j + i * max_boxes_per_class];
        }
    }

    // Free device memory
    cudaFree(d_res_infos);
    cudaFree(d_res_infos_sizes);
    cudaFree(d_res_infos_offsets);

    // Free host memory
    delete[] h_res_infos_sizes;
    delete[] h_res_infos;
}