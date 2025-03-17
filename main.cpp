#include "ArmorDetector.hpp"

#define VIDEO 0
#define CAMERA "0"

int main() {

    // std::ios::sync_with_stdio(false);
    // std::cin.tie(0);
    // std::cout.tie(0);

    ArmorDetector detector;
    detector.init(RED);
    detector.loadImg();
    detector.detect();

}