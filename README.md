# YoloONNX Proj

### *Valmorlee*


## 性能与测试环境 **（ Perf & Env ）**

---

> 基础配置： ``Ubuntu 24.04`` ``i9-13900H`` ``RTX4060-Laptop``

> 测试环境： ``Input: 320 x 320`` ``FP16`` 

**预测时间对比：**

|            | **yoloONNX --CPU** | **TensorRT-YOLOX --GPU** | **GPU With ROI** |
|------------|--------------------|--------------------------|------------------|
| **Video**  | **43.81ms**        | **1.2ms**                |                  |
| **Camera** | **41.08ms**        | **0.9ms**                | **0.74ms**       |


## 目录架构 **（ CataStructure )**

---
``` YOLOONNX
yoloONNX
   ├── TensorRT-YOLOX (GPU)
   ├── ${else} (CPU)
```



## TODO

---
- [x] **添加 ROI**
- [x] **添加 ByteTrack** 
- [ ] **添加 EKF** 
- [ ] **添加 EPNP**
- [ ] **添加 YOLO-Pose**