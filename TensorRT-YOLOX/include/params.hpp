//
// Created by valmorx on 25-3-20.
//

#ifndef PARAMS_HPP
#define PARAMS_HPP

#include <string>
#include <cstring>
#include <vector>

#define BLUE 0
#define RED 1

namespace params {

    inline int Enemy_color = BLUE;

    inline std::string Engine_Path = "/home/valmorx/CLionProjects/RM_yoloONNX/TensorRT-YOLOX/00x.engine";
    inline std::string Video_Path = "/home/valmorx/DeepLearningSource/video.mp4";

    inline std::vector trackClass = {
        0,1,2,3,4,5,6,7,8,9,10,11,12,13 //
    };

    inline std::vector<std::string> class_names = {
        "B1","B2","B3","B4","B5","BO","BS","R1","R2","R3","R4","R5","RO","RS"
    };


}


#endif //PARAMS_HPP
