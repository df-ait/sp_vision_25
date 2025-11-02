#include <fmt/core.h>

#include <chrono>
#include <cmath>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/gkdcontrol.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"


using namespace std::chrono;

const std::string keys =
  "{help h usage ? |      | 输出命令行参数说明}"
  "{@config-path   | configs/standard3.yaml | 位置参数，yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder;

  io::GKDControl gkdcontrol(config_path);
  io::Camera camera(config_path);

  auto_aim::YOLO detector(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);

  cv::Mat img;
  Eigen::Quaterniond q;
  std::chrono::steady_clock::time_point t;
  io::Command last_command{};
  bool has_last_command = false;
  int frame_count = 0;

  while (!exiter.exit()) {
    camera.read(img, t);
    if (img.empty()) continue;

    q = gkdcontrol.imu_at(t - 1ms);

    // recorder.record(img, q, t);

    solver.set_R_gimbal2world(q);

    auto yolo_start = std::chrono::steady_clock::now();
    auto armors = detector.detect(img, frame_count);

    auto tracker_start = std::chrono::steady_clock::now();
    auto targets = tracker.track(armors, t);

    auto aimer_start = std::chrono::steady_clock::now();
    auto command = aimer.aim(targets, t, gkdcontrol.bullet_speed);
    auto finish = std::chrono::steady_clock::now();

    if (
      !targets.empty() && aimer.debug_aim_point.valid && has_last_command &&
      std::abs(command.yaw - last_command.yaw) * 57.3 < 2.0)
      command.shoot = true;

    if (command.control) {
      last_command = command;
      has_last_command = true;
    }

    Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

    tools::draw_text(
      img,
      fmt::format(
        "command {}, yaw {:.2f}, pitch {:.2f}, shoot {}", command.control,
        command.yaw * 57.3, command.pitch * 57.3, command.shoot),
      {10, 40}, {154, 50, 205});

    tools::draw_text(
      img, fmt::format("gimbal yaw {:.2f}", ypr[0] * 57.3), {10, 70}, {255, 255, 255});

    nlohmann::json data;
    data["frame"] = frame_count;
    data["gimbal_yaw"] = ypr[0] * 57.3;
    data["cmd_yaw"] = command.yaw * 57.3;
    data["cmd_pitch"] = command.pitch * 57.3;
    data["shoot"] = command.shoot;
    data["yolo_ms"] = tools::delta_time(tracker_start, yolo_start) * 1e3;
    data["tracker_ms"] = tools::delta_time(aimer_start, tracker_start) * 1e3;
    data["aimer_ms"] = tools::delta_time(finish, aimer_start) * 1e3;

    if (!armors.empty()) {
      const auto & armor = armors.front();
      data["armor_num"] = armors.size();
      data["armor_x"] = armor.xyz_in_world[0];
      data["armor_y"] = armor.xyz_in_world[1];
      data["armor_yaw"] = armor.ypr_in_world[0] * 57.3;
      data["armor_yaw_raw"] = armor.yaw_raw * 57.3;
      data["armor_center_x"] = armor.center_norm.x;
      data["armor_center_y"] = armor.center_norm.y;
    } else {
      data["armor_num"] = 0;
    }

    if (!targets.empty()) {
      const auto & target = targets.front();
      const auto state = target.ekf_x();
      data["x"] = state[0];
      data["vx"] = state[1];
      data["y"] = state[2];
      data["vy"] = state[3];
      data["z"] = state[4];
      data["vz"] = state[5];
      data["a"] = state[6] * 57.3;
      data["w"] = state[7];
      data["r"] = state[8];
      data["l"] = state[9];
      data["h"] = state[10];
      data["last_id"] = target.last_id;

      // 卡方检验相关数据
      data["residual_yaw"] = target.ekf().data.at("residual_yaw");
      data["residual_pitch"] = target.ekf().data.at("residual_pitch");
      data["residual_distance"] = target.ekf().data.at("residual_distance");
      data["residual_angle"] = target.ekf().data.at("residual_angle");
      data["nis"] = target.ekf().data.at("nis");
      data["nees"] = target.ekf().data.at("nees");
      data["nis_fail"] = target.ekf().data.at("nis_fail");
      data["nees_fail"] = target.ekf().data.at("nees_fail");
      data["recent_nis_failures"] = target.ekf().data.at("recent_nis_failures");

      for (const Eigen::Vector4d & xyza : target.armor_xyza_list()) {
        auto image_points =
          solver.reproject_armor(xyza.head<3>(), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});
      }

      const auto aim_point = aimer.debug_aim_point;
      if (aim_point.valid) {
        auto aim_points = solver.reproject_armor(
          aim_point.xyza.head<3>(), aim_point.xyza[3], target.armor_type, target.name);
        tools::draw_points(img, aim_points, {0, 0, 255});
      }

      tools::logger()->info(
        "[Infantry][{}] Target state -> x {:.3f} m, y {:.3f} m, z {:.3f} m, yaw {:.3f} rad",
        frame_count, state[0], state[2], state[4], state[6]);
    }

    tools::logger()->info(
      "[Infantry][{}] yolo: {:.1f}ms, tracker: {:.1f}ms, aimer: {:.1f}ms", frame_count,
      tools::delta_time(tracker_start, yolo_start) * 1e3,
      tools::delta_time(aimer_start, tracker_start) * 1e3,
      tools::delta_time(finish, aimer_start) * 1e3);

    plotter.plot(data);

    cv::resize(img, img, {}, 0.5, 0.5);
    cv::imshow("gkdinfantry", img);
    cv::waitKey(1);

    gkdcontrol.send(command);
    frame_count++;
  }

  return 0;
}
