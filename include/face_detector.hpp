#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct Face {
  cv::Rect box;
  cv::Mat landmarks;   // 5x2 float
  float score = 0.f;
  cv::Mat crop;        // BGR face crop
};

class YuNetDetector {
 public:
  explicit YuNetDetector(const std::string& model_path,
                         float score_threshold = 0.6f);

  std::vector<Face> detect(const cv::Mat& image_bgr);

 private:
  cv::Ptr<cv::FaceDetectorYN> model_;
};
