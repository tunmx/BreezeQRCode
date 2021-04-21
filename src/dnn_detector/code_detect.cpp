//
// Created by tunm on 2021/4/19.
//

#include <iostream>
#include "code_detect.h"

inline float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float *rows(const cv::Mat &cls_pred, const int y) {
    // N H W
    int w = cls_pred.size[2];
    return (float *) ((unsigned char *) cls_pred.data + (size_t) w * y * 4);
}


int resize_uniform(const cv::Mat &src, cv::Mat &dst, cv::Size dst_size,
                   object_rect &effect_area) {
    int w = src.cols;
    int h = src.rows;
    int dst_w = dst_size.width;
    int dst_h = dst_size.height;
    // std::cout << "src: (" << h << ", " << w << ")" << std::endl;
    dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(0));

    float ratio_src = w * 1.0 / h;
    float ratio_dst = dst_w * 1.0 / dst_h;

    int tmp_w = 0;
    int tmp_h = 0;
    if (ratio_src > ratio_dst) {
        tmp_w = dst_w;
        tmp_h = floor((dst_w * 1.0 / w) * h);
    } else if (ratio_src < ratio_dst) {
        tmp_h = dst_h;
        tmp_w = floor((dst_h * 1.0 / h) * w);
    } else {
        cv::resize(src, dst, dst_size);
        effect_area.x = 0;
        effect_area.y = 0;
        effect_area.width = dst_w;
        effect_area.height = dst_h;
        return 0;
    }

    // std::cout << "tmp: (" << tmp_h << ", " << tmp_w << ")" << std::endl;
    cv::Mat tmp;
    cv::resize(src, tmp, cv::Size(tmp_w, tmp_h));

    if (tmp_w != dst_w) {
        int index_w = floor((dst_w - tmp_w) / 2.0);
        // std::cout << "index_w: " << index_w << std::endl;
        for (int i = 0; i < dst_h; i++) {
            memcpy(dst.data + i * dst_w * 3 + index_w * 3, tmp.data + i * tmp_w * 3,
                   tmp_w * 3);
        }
        effect_area.x = index_w;
        effect_area.y = 0;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    } else if (tmp_h != dst_h) {
        int index_h = floor((dst_h - tmp_h) / 2.0);
        // std::cout << "index_h: " << index_h << std::endl;
        memcpy(dst.data + index_h * dst_w * 3, tmp.data, tmp_w * tmp_h * 3);
        effect_area.x = 0;
        effect_area.y = index_h;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    } else {
        printf("error\n");
    }
    return 0;
}

template<typename _Tp>
int activation_function_softmax(const _Tp *src, _Tp *dst, int length) {
    const _Tp alpha = *std::max_element(src, src + length);
    _Tp denominator{0};

    for (int i = 0; i < length; ++i) {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }

    return 0;
}

CodeDetector::CodeDetector(const std::string &model_path) {
    this->Net_ = cv::dnn::readNetFromONNX(model_path);
}

std::vector<CodeBoxInfo> CodeDetector::_detection(const cv::Mat &image, float score_threshold, float nms_threshold) {
    cv::Mat input;
    _preProcess(image, input);
    Net_.setInput(input);
    std::vector<std::vector<CodeBoxInfo>> results;
    results.resize(this->num_class);

    std::vector<std::string> nodes;

    for (const auto &head_info : this->heads_info) {
        nodes.push_back(head_info.dis_layer.c_str());
        nodes.push_back(head_info.cls_layer.c_str());
    }

    std::vector<cv::Mat> outputs;
    Net_.forward(outputs, nodes);
    for (int i = 0; i < nodes.size() / 2; i++) {
        auto dis_pred = outputs[i * 2];
        auto cls_pred = outputs[i * 2 + 1];
        int stride = this->heads_info[i].stride;
        this->_decodeInfer(cls_pred, dis_pred, stride, score_threshold, results);


    }

    std::vector<CodeBoxInfo> dets;
    for (int i = 0; i < (int) results.size(); i++) {
        this->_nms(results[i], nms_threshold);

        for (auto box : results[i]) {
            dets.push_back(box);
        }
    }
    return dets;
}

void CodeDetector::_nms(std::vector<CodeBoxInfo> &input_boxes, float nms_threshold) {
    std::sort(input_boxes.begin(), input_boxes.end(),
              [](CodeBoxInfo a, CodeBoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < input_boxes.size(); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) *
                   (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < input_boxes.size(); ++i) {
        for (int j = i + 1; j < input_boxes.size();) {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= nms_threshold) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            } else {
                j++;
            }
        }
    }
}

CodeBoxInfo CodeDetector::_disPred2Bbox(const float *&dfl_det, int label, float score, int x, int y, int stride) const {
    float ct_x = (x + 0.5) * stride;
    float ct_y = (y + 0.5) * stride;
    std::vector<float> dis_pred;
    dis_pred.resize(4);
    for (int i = 0; i < 4; i++) {
        float dis = 0;
        float *dis_after_sm = new float[this->reg_max + 1];
        activation_function_softmax(dfl_det + i * (this->reg_max + 1), dis_after_sm,
                                    this->reg_max + 1);
        for (int j = 0; j < this->reg_max + 1; j++) {
            dis += j * dis_after_sm[j];
        }
        dis *= stride;
        dis_pred[i] = dis;
        delete[] dis_after_sm;
    }
    float xmin = (std::max)(ct_x - dis_pred[0], .0f);
    float ymin = (std::max)(ct_y - dis_pred[1], .0f);
    float xmax = (std::min)(ct_x + dis_pred[2], (float) this->input_size);
    float ymax = (std::min)(ct_y + dis_pred[3], (float) this->input_size);
    return CodeBoxInfo{xmin, ymin, xmax, ymax, score, label};
}


void CodeDetector::_preProcess(const cv::Mat &image, cv::Mat &blob) {
    int img_w = image.cols;
    int img_h = image.rows;

    const float mean_vals[3] = {104.04f, 113.9f, 119.8f};
    const float norm_vals[3] = {0.013569f, 0.014312f, 0.014106f};
    blob = cv::dnn::blobFromImage(image, 1.0, cv::Size(image.cols, image.rows),
                                  cv::Scalar(0.0, 0.0, 0.0), true, false);
    float mean_rgb[3] = {104.04f, 113.9f, 119.8f};
    float std_rgb[3] = {1 / 0.013569f, 1 / 0.014312f, 1 / 0.014106f};
    float scale = 1.0;
    float *header = (float *) blob.data;
    int size = blob.size[2] * blob.size[3];
    for (int c = 0; c < 3; c++) {
        for (int k = 0; k < size; k++)
            header[c * size + k] = static_cast<float>(
                    (header[c * size + k] / scale - mean_rgb[c]) / std_rgb[c]);
    }
}

void CodeDetector::_decodeInfer(cv::Mat &cls_pred, cv::Mat &dis_pred, int stride, float threshold,
                                std::vector<std::vector<CodeBoxInfo>> &results) {
    int feature_h = this->input_size / stride;
    int feature_w = this->input_size / stride;
    // cv::Mat debug_heatmap = cv::Mat(feature_h, feature_w, CV_8UC3);
//    float *data_out = (float *) cls_pred.data;
    for (int idx = 0; idx < feature_h * feature_w; idx++) {
        const float *scores = rows(cls_pred, idx);
        int row = idx / feature_w;
        int col = idx % feature_w;
        float score = 0;
        int cur_label = 0;
        for (int label = 0; label < this->num_class; label++) {
            if (scores[label] > score) {
                score = scores[label];
                cur_label = label;
            }
        }
        if (score > threshold) {
            const float *bbox_pred = rows(dis_pred, idx);
            results[cur_label].push_back(
                    this->_disPred2Bbox(bbox_pred, cur_label, score, col, row, stride));
        }
    }

}

std::vector<CodeBoxInfo>
CodeDetector::DetectCode(const cv::Mat &rgb, float score_threshold, float nms_threshold) {
    object_rect effect_roi;
    cv::Mat resized_img;
    resize_uniform(rgb, resized_img, cv::Size(320, 320), effect_roi);
    int src_w = rgb.cols;
    int src_h = rgb.rows;
    int dst_w = effect_roi.width;
    int dst_h = effect_roi.height;
    float width_ratio = (float) src_w / (float) dst_w;
    float height_ratio = (float) src_h / (float) dst_h;
    //    cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);
    std::vector<CodeBoxInfo> results =
            this->_detection(resized_img, score_threshold, nms_threshold);
    std::vector<std::tuple<int, double>> rec_results;
    std::vector<CodeBoxInfo> final_results;
    for (auto &bbox : results) {
        cv::Rect rect(cv::Point((bbox.x1 - effect_roi.x) * width_ratio,
                                (bbox.y1 - effect_roi.y) * height_ratio),
                      cv::Point((bbox.x2 - effect_roi.x) * width_ratio,
                                (bbox.y2 - effect_roi.y) * height_ratio));
        CodeBoxInfo boxInfo;
        boxInfo.x1 = rect.x;
        boxInfo.y1 = rect.y;
        boxInfo.x2 = rect.x + rect.width;
        boxInfo.y2 = rect.y + rect.height;
        boxInfo.score = bbox.score;
        boxInfo.cls = bbox.cls;
        final_results.push_back(boxInfo);
    }

    return final_results;
}


