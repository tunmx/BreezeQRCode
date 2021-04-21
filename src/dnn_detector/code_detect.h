//
// Created by tunm on 2021/4/19.
//

#ifndef HYPERQRCODE_CODE_DETECT_H
#define HYPERQRCODE_CODE_DETECT_H

#include "opencv2/imgproc.hpp"
#include "opencv2/dnn.hpp"
#include "../common.h"

class CodeDetector {
public:

    CodeDetector(const std::string &model_path);

    ~CodeDetector() {};

    std::vector<CodeBoxInfo> DetectCode(const cv::Mat &rgb, float score_threshold,
                                        float nms_threshold);

private:

    std::vector<CodeBoxInfo> _detection(const cv::Mat &image, float score_threshold, float nms_threshold);

    static void _preProcess(const cv::Mat &image, cv::Mat &blob);

    void _decodeInfer(cv::Mat &cls_pred, cv::Mat &dis_pred, int stride,
                      float threshold, std::vector<std::vector<CodeBoxInfo>> &results);

    CodeBoxInfo _disPred2Bbox(const float *&dfl_det, int label, float score, int x,
                              int y, int stride) const;

    static void _nms(std::vector<CodeBoxInfo> &input_boxes, float nms_threshold);

private:
    int input_size = 320;
    int num_class = 2;
    int reg_max = 7;
    std::vector<HeadInfo> heads_info{
            // cls_pred|dis_pred|stride
            {"792", "795", 8},
            {"814", "817", 16},
            {"836", "839", 32},
    };

    cv::dnn::Net Net_;


};


#endif //HYPERQRCODE_CODE_DETECT_H
