#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <utility>

// Order matches OpenCV Zoo MobileFaceNet FER export
inline const std::vector<std::string> EMOTION_LABELS = {
    "angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"};

class FacialExpressionRecog {
 public:
  explicit FacialExpressionRecog(
      const std::string& model_path,
      int backend_id = cv::dnn::DNN_BACKEND_OPENCV,
      int target_id = cv::dnn::DNN_TARGET_CPU,
      cv::Size input_size = cv::Size(112, 112));

  // Returns softmax probabilities (size 7)
  cv::Mat infer(const cv::Mat& face_bgr);

  // Returns {label, confidence, probs}
  std::tuple<std::string, float, cv::Mat> predict_label(const cv::Mat& face_bgr);

 private:
  cv::Mat preprocess(const cv::Mat& face_bgr) const;

  cv::dnn::Net net_;
  cv::Size input_size_;
};
