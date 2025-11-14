#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <chrono>
#include <string>
#include <vector>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/gkdcontrol.hpp"
#include "tools/math_tools.hpp"

const std::string keys =
  "{help h usage ? |                          | 输出命令行参数说明}"
  "{config-path c  | configs/calibration.yaml | yaml配置文件路径 }"
  "{samples s      | 40                       | 采集多少组有效数据 }";

std::vector<cv::Point3f> centers_3d(const cv::Size & pattern_size, const float center_distance)
{
  std::vector<cv::Point3f> centers_3d;

  for (int i = 0; i < pattern_size.height; i++)
    for (int j = 0; j < pattern_size.width; j++)
      centers_3d.push_back({j * center_distance, i * center_distance, 0});

  return centers_3d;
}

void print_yaml(
  const std::vector<double> & R_gimbal2imubody_data, const cv::Mat & R_camera2gimbal,
  const cv::Mat & t_camera2gimbal, const Eigen::Vector3d & ypr)
{
  YAML::Emitter result;
  std::vector<double> R_camera2gimbal_data(
    R_camera2gimbal.begin<double>(), R_camera2gimbal.end<double>());
  std::vector<double> t_camera2gimbal_data(
    t_camera2gimbal.begin<double>(), t_camera2gimbal.end<double>());

  result << YAML::BeginMap;
  result << YAML::Key << "R_gimbal2imubody";
  result << YAML::Value << YAML::Flow << R_gimbal2imubody_data;
  result << YAML::Newline;
  result << YAML::Newline;
  result << YAML::Comment(fmt::format(
    "相机同理想情况的偏角: yaw{:.2f} pitch{:.2f} roll{:.2f} degree", ypr[0], ypr[1], ypr[2]));
  result << YAML::Key << "R_camera2gimbal";
  result << YAML::Value << YAML::Flow << R_camera2gimbal_data;
  result << YAML::Key << "t_camera2gimbal";
  result << YAML::Value << YAML::Flow << t_camera2gimbal_data;
  result << YAML::Newline;
  result << YAML::EndMap;

  fmt::print("\n{}\n", result.c_str());
}

bool detect_chessboard(
  const cv::Mat & gray, const cv::Size & pattern_size, std::vector<cv::Point2f> & centers_2d)
{
  auto flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK;
  auto found = cv::findChessboardCorners(gray, pattern_size, centers_2d, flags);
  if (found) {
    cv::cornerSubPix(
      gray, centers_2d, cv::Size(11, 11), cv::Size(-1, -1),
      cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));
  }
  return found;
}

int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>("config-path");
  auto target_samples = cli.get<int>("samples");
  fmt::print("按空格保存棋盘格，按q退出。目标采集 {} 组数据。\n", target_samples);

  // 读取标定配置
  auto yaml = YAML::LoadFile(config_path);
  auto pattern_cols = yaml["pattern_cols"].as<int>();
  auto pattern_rows = yaml["pattern_rows"].as<int>();
  auto center_distance_mm = yaml["center_distance_mm"].as<double>();
  std::vector<double> R_gimbal2imubody_data =
    yaml["R_gimbal2imubody"].as<std::vector<double>>();
  auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();

  cv::Size pattern_size(pattern_cols, pattern_rows);
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R_gimbal2imubody(R_gimbal2imubody_data.data());
  cv::Matx33d camera_matrix(camera_matrix_data.data());
  cv::Mat distort_coeffs(distort_coeffs_data);
  auto object_points = centers_3d(pattern_size, center_distance_mm);

  io::Camera camera(config_path);
  io::GKDControl gkdcontrol(config_path);

  std::vector<cv::Mat> R_gimbal2world_list, t_gimbal2world_list;
  std::vector<cv::Mat> rvecs, tvecs;

  cv::namedWindow("HandEye Calibration", cv::WINDOW_NORMAL);
  int collected = 0;

  while (target_samples <= 0 || collected < target_samples) {
    cv::Mat frame;
    std::chrono::steady_clock::time_point timestamp;
    camera.read(frame, timestamp);
    if (frame.empty()) continue;

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> centers_2d;
    bool found = detect_chessboard(gray, pattern_size, centers_2d);

    cv::Mat drawing = frame.clone();
    if (found) {
      cv::drawChessboardCorners(drawing, pattern_size, centers_2d, found);
    }

    Eigen::Quaterniond q = gkdcontrol.imu_at(timestamp);
    Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
    Eigen::Matrix3d R_gimbal2world =
      R_gimbal2imubody.transpose() * R_imubody2imuabs * R_gimbal2imubody;
    Eigen::Vector3d ypr = tools::eulers(R_gimbal2world, 2, 1, 0) * 57.3;

    auto text_color = found ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
    cv::putText(
      drawing, fmt::format("Samples: {}", collected), {30, 40}, cv::FONT_HERSHEY_SIMPLEX, 1.0,
      cv::Scalar(0, 255, 255), 2);
    cv::putText(
      drawing, fmt::format("Yaw {:.2f}", ypr[0]), {30, 90}, cv::FONT_HERSHEY_SIMPLEX, 0.7,
      text_color, 2);
    cv::putText(
      drawing, fmt::format("Pitch {:.2f}", ypr[1]), {30, 130}, cv::FONT_HERSHEY_SIMPLEX, 0.7,
      text_color, 2);
    cv::putText(
      drawing, fmt::format("Roll {:.2f}", ypr[2]), {30, 170}, cv::FONT_HERSHEY_SIMPLEX, 0.7,
      text_color, 2);
    cv::putText(
      drawing, "SPACE: Capture   Q: Quit", {30, drawing.rows - 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
      cv::Scalar(255, 255, 255), 2);
    cv::imshow("HandEye Calibration", drawing);

    int key = cv::waitKey(1);
    if (key == 'q' || key == 'Q') break;

    if ((key == ' ' || key == 's' || key == 'S') && found) {
      cv::Mat R_gimbal2world_cv;
      cv::eigen2cv(R_gimbal2world, R_gimbal2world_cv);
      cv::Mat t_gimbal2world = (cv::Mat_<double>(3, 1) << 0, 0, 0);
      cv::Mat rvec, tvec;
      cv::solvePnP(
        object_points, centers_2d, camera_matrix, distort_coeffs, rvec, tvec, false,
        cv::SOLVEPNP_IPPE);

      R_gimbal2world_list.emplace_back(R_gimbal2world_cv.clone());
      t_gimbal2world_list.emplace_back(t_gimbal2world.clone());
      rvecs.emplace_back(rvec.clone());
      tvecs.emplace_back(tvec.clone());
      collected++;
      fmt::print("记录第 {} 组数据。\n", collected);
    }
  }

  cv::destroyWindow("HandEye Calibration");

  if (R_gimbal2world_list.size() < 3) {
    fmt::print("有效数据不足，至少需要3组。\n");
    return 1;
  }

  // 手眼标定
  cv::Mat R_camera2gimbal, t_camera2gimbal;
  cv::calibrateHandEye(
    R_gimbal2world_list, t_gimbal2world_list, rvecs, tvecs, R_camera2gimbal, t_camera2gimbal);
  t_camera2gimbal /= 1e3;  // mm to m

  // 计算相机同理想情况的偏角
  Eigen::Matrix3d R_camera2gimbal_eigen;
  cv::cv2eigen(R_camera2gimbal, R_camera2gimbal_eigen);
  Eigen::Matrix3d R_gimbal2ideal{{0, -1, 0}, {0, 0, -1}, {1, 0, 0}};
  Eigen::Matrix3d R_camera2ideal = R_gimbal2ideal * R_camera2gimbal_eigen;
  Eigen::Vector3d ypr = tools::eulers(R_camera2ideal, 1, 0, 2) * 57.3;  // degree

  // 输出yaml
  print_yaml(R_gimbal2imubody_data, R_camera2gimbal, t_camera2gimbal, ypr);
}
