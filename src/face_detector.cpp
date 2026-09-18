#include "face_detector.hpp"

#include <algorithm>
#include <iostream>

YuNetDetector::YuNetDetector(const std::string& model_path, float score_threshold) {
  model_ = cv::FaceDetectorYN::create(
      model_path,
      "",                    // config (unused for ONNX)
      cv::Size(320, 320),    // default input size; overwritten per-frame
      score_threshold,
      0.3f,                  // nms_threshold
      5000                   // top_k
  );
  if (model_.empty()) {
    throw std::runtime_error("Failed to load YuNet model: " + model_path);
  }
}

std::vector<Face> YuNetDetector::detect(const cv::Mat& image_bgr) {
  std::vector<Face> results;
  if (image_bgr.empty()) return results;

  const int h = image_bgr.rows;
  const int w = image_bgr.cols;
  model_->setInputSize(cv::Size(w, h));

  cv::Mat faces;
  model_->detect(image_bgr, faces);

  if (faces.empty()) return results;

  for (int i = 0; i < faces.rows; ++i) {
    float* f = faces.ptr<float>(i);
    int x  = static_cast<int>(f[0]);
    int y  = static_cast<int>(f[1]);
    int bw = static_cast<int>(f[2]);
    int bh = static_cast<int>(f[3]);

    // Clamp to image bounds
    x  = std::max(0, x);
    y  = std::max(0, y);
    bw = std::min(bw, w - x);
    bh = std::min(bh, h - y);
    if (bw < 8 || bh < 8) continue;

    Face face;
    face.box = cv::Rect(x, y, bw, bh);
    face.landmarks = cv::Mat(5, 2, CV_32F);
    for (int k = 0; k < 5; ++k) {
      face.landmarks.at<float>(k, 0) = f[4 + 2 * k];
      face.landmarks.at<float>(k, 1) = f[4 + 2 * k + 1];
    }
    face.score = f[14];
    face.crop  = image_bgr(face.box).clone();
    results.push_back(std::move(face));
  }
  return results;
}
