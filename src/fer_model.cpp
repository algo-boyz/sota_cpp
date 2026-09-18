#include "fer_model.hpp"

#include <cmath>
#include <stdexcept>

FacialExpressionRecog::FacialExpressionRecog(const std::string& model_path,
                                             int backend_id, int target_id,
                                             cv::Size input_size)
    : input_size_(input_size) {
  net_ = cv::dnn::readNet(model_path);
  if (net_.empty()) {
    throw std::runtime_error("Failed to load FER ONNX model: " + model_path);
  }
  net_.setPreferableBackend(backend_id);
  net_.setPreferableTarget(target_id);
}

cv::Mat FacialExpressionRecog::preprocess(const cv::Mat& face_bgr) const {
  // blobFromImage: scale 1/127.5, mean 127.5, swapRB=true → matches Python
  return cv::dnn::blobFromImage(
      face_bgr,
      1.0 / 127.5,
      input_size_,
      cv::Scalar(127.5, 127.5, 127.5),
      true,   // swapRB
      false   // crop
  );
}

cv::Mat FacialExpressionRecog::infer(const cv::Mat& face_bgr) {
  if (face_bgr.empty()) {
    return cv::Mat::zeros(1, 7, CV_32F);
  }

  cv::Mat face = face_bgr;
  const int min_side = std::min(face.rows, face.cols);
  if (min_side < 48) {
    const double scale = 112.0 / min_side;
    cv::resize(face, face, cv::Size(), scale, scale, cv::INTER_CUBIC);
  }

  cv::Mat blob = preprocess(face);
  net_.setInput(blob);
  cv::Mat logits = net_.forward();
  logits = logits.reshape(1, 1);  // 1x7

  // Softmax (numerically stable)
  double max_val = 0.0;
  cv::minMaxLoc(logits, nullptr, &max_val);
  cv::Mat exp;
  cv::exp(logits - max_val, exp);
  const double sum = cv::sum(exp)[0];
  return exp / sum;  // 1x7 float32 probabilities
}

std::tuple<std::string, float, cv::Mat>
FacialExpressionRecog::predict_label(const cv::Mat& face_bgr) {
  cv::Mat probs = infer(face_bgr);
  cv::Point max_loc;
  double max_val = 0.0;
  cv::minMaxLoc(probs, nullptr, &max_val, nullptr, &max_loc);
  const int idx = max_loc.x;
  return {EMOTION_LABELS[idx], static_cast<float>(max_val), probs};
}
