//
// Created by tunm on 2021/4/20.
//

#include <iostream>
#include "src/dnn_detector/code_detect.h"
#include "opencv2/opencv.hpp"
#include "src/common.h"

using namespace std;

int main(int argc, char **argv) {
    if (argc < 3) {
        cout << "error params" << endl;
        return -1;
    }
    string model_path = argv[1];
    CodeDetector detector(model_path);
    for (int i = 2; i < argc; i++) {
        string img_path = argv[i];
        cv::Mat image = cv::imread(img_path);
        int64 tall = cv::getTickCount();
        std::vector<CodeBoxInfo> res = detector.DetectCode(image, 0.3, 0.5);
        std::cout << "time：" << (cv::getTickCount() - tall) / cv::getTickFrequency() << " sec." << std::endl;
        draw_bboxes(image, res);
        cv::imshow("s", image);
        cv::waitKey(0);
    }

}