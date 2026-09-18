/**
 * SOTA-style Facial Expression Recognition – C++ port
 *
 * Model : Progressive Teacher + MobileFaceNet (OpenCV Zoo ONNX)
 * Face  : YuNet
 * ~88% RAF-DB  |  classic VGG19 baseline ~66% FER-2013
 *
 * Usage (after building):
 *   ./sota_fer                     # run on assets/test samples if present
 *   ./sota_fer --image photo.jpg
 *   ./sota_fer --webcam
 *   ./sota_fer --show              # with --image
 */

#include "face_detector.hpp"
#include "fer_model.hpp"

#include <opencv2/opencv.hpp>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Resolve project root: look for models/ next to cwd or one level up (build/)
static fs::path find_root() {
  const fs::path cwd = fs::current_path();
  if (fs::exists(cwd / "models")) return cwd;
  if (fs::exists(cwd.parent_path() / "models")) return cwd.parent_path();
  return cwd;  // fallback
}

static const fs::path ROOT = find_root();
static const fs::path MODELS_DIR = ROOT / "models";
static const fs::path ASSETS_DIR = ROOT / "assets" / "test";

static const fs::path FER_ONNX =
    MODELS_DIR / "facial_expression_recognition_mobilefacenet_2022july.onnx";
static const fs::path YUNET_ONNX =
    MODELS_DIR / "face_detection_yunet_2023mar.onnx";

// BGR colour map
static const std::map<std::string, cv::Scalar> COLOURS = {
    {"angry",     {0, 0, 255}},
    {"disgust",   {0, 128, 0}},
    {"fearful",   {128, 0, 128}},
    {"happy",     {0, 255, 255}},
    {"neutral",   {200, 200, 200}},
    {"sad",       {255, 0, 0}},
    {"surprised", {0, 165, 255}},
};

static void ensure_models() {
  if (fs::exists(FER_ONNX) && fs::exists(YUNET_ONNX)) return;

  std::cerr
      << "Models not found in " << MODELS_DIR << "\n"
      << "Download them once with:\n"
      << "  mkdir -p models\n"
      << "  # using huggingface-cli or curl / browser from:\n"
      << "  # https://huggingface.co/opencv/facial_expression_recognition\n"
      << "  # https://huggingface.co/opencv/face_detection_yunet\n"
      << "  # files:\n"
      << "  #   facial_expression_recognition_mobilefacenet_2022july.onnx\n"
      << "  #   face_detection_yunet_2023mar.onnx\n"
      << "Or run the original Python download_models.py and copy the models/ folder here.\n";
  std::exit(1);
}

static cv::Mat annotate(const cv::Mat& frame,
                        const std::vector<Face>& faces,
                        const std::vector<std::tuple<std::string, float, cv::Mat>>& predictions) {
  cv::Mat out = frame.clone();
  for (size_t i = 0; i < faces.size() && i < predictions.size(); ++i) {
    const auto& face = faces[i];
    const auto& [label, conf, probs] = predictions[i];
    const cv::Scalar colour = COLOURS.count(label) ? COLOURS.at(label) : cv::Scalar(255, 255, 255);

    cv::rectangle(out, face.box, colour, 2);
    const std::string caption =
        label + " " + cv::format("%.1f%%", conf * 100.f);
    cv::putText(out, caption,
                {face.box.x, std::max(20, face.box.y - 8)},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv::LINE_AA);

    // probability bars
    const int bar_x = face.box.x + face.box.width + 6;
    for (int j = 0; j < static_cast<int>(EMOTION_LABELS.size()); ++j) {
      const float p = probs.at<float>(0, j);
      const int by = face.box.y + j * 14;
      const auto& name = EMOTION_LABELS[j];
      const cv::Scalar c = COLOURS.count(name) ? COLOURS.at(name) : cv::Scalar(128, 128, 128);
      cv::rectangle(out, {bar_x, by}, {bar_x + static_cast<int>(60 * p), by + 10}, c, -1);
      cv::putText(out, name.substr(0, 3),
                  {bar_x + 62, by + 10},
                  cv::FONT_HERSHEY_SIMPLEX, 0.35, {220, 220, 220}, 1);
    }
  }
  return out;
}

static void run_on_image(const fs::path& path,
                         FacialExpressionRecog& fer,
                         YuNetDetector& detector,
                         bool show) {
  cv::Mat img = cv::imread(path.string());
  if (img.empty()) {
    std::cerr << "Cannot read " << path << "\n";
    return;
  }

  auto faces = detector.detect(img);
  if (faces.empty()) {
    // Fallback: treat whole image as face (for small FER sample crops)
    Face full;
    full.box = {0, 0, img.cols, img.rows};
    full.score = 1.f;
    full.crop = img.clone();
    faces.push_back(std::move(full));
    std::cout << "  (no YuNet face found – using full frame as crop)\n";
  }

  std::vector<std::tuple<std::string, float, cv::Mat>> predictions;
  for (const auto& face : faces) {
    auto pred = fer.predict_label(face.crop);
    const auto& [label, conf, probs] = pred;
    predictions.push_back(pred);

    std::cout << "  → " << std::left << std::setw(10) << label
              << "  " << std::fixed << std::setprecision(1) << (conf * 100.f) << "%   ";
    for (int j = 0; j < static_cast<int>(EMOTION_LABELS.size()); ++j) {
      std::cout << EMOTION_LABELS[j].substr(0, 3) << "="
                << static_cast<int>(probs.at<float>(0, j) * 100.f) << " ";
    }
    std::cout << "\n";
  }

  cv::Mat vis = annotate(img, faces, predictions);

  fs::path out_path = path;
  out_path.replace_filename(path.stem().string() + "_sota.jpg");
  // Prefer writing next to executable / cwd if original path is awkward
  if (!fs::exists(out_path.parent_path()) ||
      path.string().find("assets") != std::string::npos) {
    out_path = ROOT / (path.stem().string() + "_sota.jpg");
  }
  cv::imwrite(out_path.string(), vis);
  std::cout << "  saved " << out_path << "\n";

  if (show) {
    cv::imshow("SOTA FER", vis);
    cv::waitKey(0);
    cv::destroyAllWindows();
  }
}

static void run_webcam(FacialExpressionRecog& fer, YuNetDetector& detector) {
  cv::VideoCapture cap(0);
  if (!cap.isOpened()) {
    std::cerr << "Cannot open webcam\n";
    std::exit(1);
  }
  std::cout << "Webcam demo – press q to quit\n";
  while (true) {
    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) break;

    auto faces = detector.detect(frame);
    std::vector<std::tuple<std::string, float, cv::Mat>> predictions;
    for (const auto& f : faces) {
      predictions.push_back(fer.predict_label(f.crop));
    }
    cv::Mat vis = annotate(frame, faces, predictions);
    cv::imshow("SOTA FER (webcam)", vis);
    if ((cv::waitKey(1) & 0xFF) == 'q') break;
  }
  cap.release();
  cv::destroyAllWindows();
}

static void print_usage(const char* argv0) {
  std::cout
      << "Usage:\n"
      << "  " << argv0 << "                     # assets/test samples\n"
      << "  " << argv0 << " --image <path>      # single image\n"
      << "  " << argv0 << " --webcam            # live camera\n"
      << "  " << argv0 << " --image <path> --show\n";
}

int main(int argc, char** argv) {
  std::string image_path;
  bool webcam = false;
  bool show = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--webcam") {
      webcam = true;
    } else if (arg == "--show") {
      show = true;
    } else if (arg == "--image" && i + 1 < argc) {
      image_path = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  ensure_models();

  FacialExpressionRecog fer(FER_ONNX.string());
  YuNetDetector detector(YUNET_ONNX.string());

  std::cout << "============================================================\n"
            << "SOTA-style FER demo (C++)\n"
            << "  Model : Progressive Teacher + MobileFaceNet (ONNX)\n"
            << "  Face  : YuNet\n"
            << "  Note  : ~88% RAF-DB  |  classic VGG19 baseline ~66% FER-2013\n"
            << "============================================================\n";

  if (webcam) {
    run_webcam(fer, detector);
    return 0;
  }

  if (!image_path.empty()) {
    run_on_image(image_path, fer, detector, show);
    return 0;
  }

  // Default: sample emotion images
  const std::vector<std::string> samples = {
      "angry.png", "disgust.png", "fear.png", "happy.png",
      "neutral.png", "sad.png", "surprise.png"};

  bool any = false;
  for (const auto& name : samples) {
    fs::path path = ASSETS_DIR / name;
    if (!fs::exists(path)) {
      // also try next to executable
      path = ROOT / "assets" / "test" / name;
    }
    if (!fs::exists(path)) {
      std::cout << "missing " << path << "\n";
      continue;
    }
    any = true;
    std::cout << "\n" << name << "\n";
    run_on_image(path, fer, detector, /*show=*/false);
  }

  if (!any) {
    std::cout << "\nNo sample images found under assets/test/.\n"
              << "Place a few emotion crops there or run with --image /path/to/photo.jpg\n";
  } else {
    std::cout << "\nDone. Annotated images written next to the samples or cwd.\n";
  }
  return 0;
}
