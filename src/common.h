//
// Created by tunm on 2021/4/19.
//

#ifndef HYPERQRCODE_COMMON_H
#define HYPERQRCODE_COMMON_H
#include "opencv2/opencv.hpp"

typedef struct CodeBoxInfo{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int cls;
} CodeBoxInfo;

typedef struct HeadInfo {
    std::string cls_layer;
    std::string dis_layer;
    int stride;
};

struct object_rect {
    int x;
    int y;
    int width;
    int height;
};

static const char *class_names[] = {"barcode", "qrcode"};

static void draw_bboxes(const cv::Mat &bgr, const std::vector<CodeBoxInfo> &bboxes) {

    for (size_t i = 0; i < bboxes.size(); i++) {
        const CodeBoxInfo &bbox = bboxes[i];
        cv::Scalar color = cv::Scalar(255, 0, 0);
        cv::Rect rect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);
        cv::rectangle(bgr, rect, color, 2);

        char text[256];
        sprintf(text, "%s %.2f%%", class_names[bbox.cls], bbox.score * 100);

        int baseLine = 0;
        cv::Size label_size =
                cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);

        int x = bbox.x1;
        int y = bbox.y1 - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > bgr.cols)
            x = bgr.cols - label_size.width;

        cv::rectangle(
                bgr,
                cv::Rect(cv::Point(x, y),
                         cv::Size(label_size.width, label_size.height + baseLine)),
                color, -1);

        cv::putText(bgr, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255));
    }

}

#endif //HYPERQRCODE_COMMON_H
