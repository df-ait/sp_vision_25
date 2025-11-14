#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <fstream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "tools/img_tools.hpp"
#include "tools/math_tools.hpp"
#include "io/camera.hpp"
#include "io/gkdcontrol.hpp"

const std::string keys =
  "{help h usage ? |                          | 输出命令行参数说明}"
  "{config-path c  | configs/calibration.yaml | yaml配置文件路径 }"
  "{@input-folder  | assets/img_with_q        | 输入文件夹路径   }";

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

  std::ofstream ofs("camera2gimbal.yaml");
  if (ofs) ofs << result.c_str();
}

int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  // 使用实时相机和 gkdcontrol 进行在线采集
  auto config_path = cli.get<std::string>("config-path");
  // init camera and gkdcontrol
  io::Camera camera(config_path);
  io::GKDControl gkdcontrol(config_path);

  std::vector<double> R_gimbal2imubody_data;
  // 从yaml读取 R_gimbal2imubody 等参数
  {
    auto yaml = YAML::LoadFile(config_path);
    R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  }

  std::vector<cv::Mat> R_gimbal2world_list, t_gimbal2world_list;
  std::vector<cv::Mat> rvecs, tvecs;

  fmt::print("实时标定模式: 按 's' 保存当前观测(检测到棋盘时)，按 'r' 运行标定并打印结果，按 'q' 退出\n");

  while (true) {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
    camera.read(img, timestamp);

    if (img.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      continue;
    }

    // 查找棋盘/圆点格
    auto yaml = YAML::LoadFile(config_path);
    cv::Size pattern_size(yaml["pattern_cols"].as<int>(), yaml["pattern_rows"].as<int>());
    double center_distance_mm = yaml["center_distance_mm"].as<double>();
    std::vector<cv::Point2f> centers_2d;
    bool success = cv::findCirclesGrid(img, pattern_size, centers_2d);

    cv::Mat drawing = img.clone();
    cv::drawChessboardCorners(drawing, pattern_size, centers_2d, success);
    cv::imshow("Live hand-eye", drawing);
    int key = cv::waitKey(1);

    if (key == 'q') break;

    if (key == 's' && success) {
      // 获取云台在该时刻的四元数
      Eigen::Quaterniond q = gkdcontrol.imu_at(timestamp);
      Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
      Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R_gimbal2imubody(R_gimbal2imubody_data.data());
      Eigen::Matrix3d R_gimbal2world =
        R_gimbal2imubody.transpose() * R_imubody2imuabs * R_gimbal2imubody;

      cv::Mat R_gimbal2world_cv;
      cv::eigen2cv(R_gimbal2world, R_gimbal2world_cv);

      // 求解相机位姿相对于棋盘
      std::vector<cv::Point3f> centers_3d_;
      for (int i = 0; i < pattern_size.height; i++)
        for (int j = 0; j < pattern_size.width; j++)
          centers_3d_.push_back({j * (float)center_distance_mm, i * (float)center_distance_mm, 0});

      auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
      auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();
      cv::Matx33d camera_matrix(camera_matrix_data.data());
      cv::Mat distort_coeffs(distort_coeffs_data);

      cv::Mat rvec, tvec;
      cv::solvePnP(centers_3d_, centers_2d, camera_matrix, distort_coeffs, rvec, tvec, false,
                   cv::SOLVEPNP_IPPE);

      R_gimbal2world_list.emplace_back(R_gimbal2world_cv);
      t_gimbal2world_list.emplace_back(cv::Mat::zeros(3, 1, CV_64F));
      rvecs.emplace_back(rvec);
      tvecs.emplace_back(tvec);

      fmt::print("保存观测: total {}\n", rvecs.size());
    }

    if (key == 'r') {
      if (R_gimbal2world_list.size() < 3) {
        fmt::print("样本太少，至少需要3个观测点\n");
        continue;
      }

      cv::Mat R_camera2gimbal, t_camera2gimbal;
      cv::calibrateHandEye(R_gimbal2world_list, t_gimbal2world_list, rvecs, tvecs,
                           R_camera2gimbal, t_camera2gimbal);
      t_camera2gimbal /= 1e3;  // mm to m

      Eigen::Matrix3d R_camera2gimbal_eigen;
      cv::cv2eigen(R_camera2gimbal, R_camera2gimbal_eigen);
      Eigen::Matrix3d R_gimbal2ideal{{0, -1, 0}, {0, 0, -1}, {1, 0, 0}};
      Eigen::Matrix3d R_camera2ideal = R_gimbal2ideal * R_camera2gimbal_eigen;
      Eigen::Vector3d ypr = tools::eulers(R_camera2ideal, 1, 0, 2) * 57.3;  // degree

      print_yaml(R_gimbal2imubody_data, R_camera2gimbal, t_camera2gimbal, ypr);
    }
  }
}
