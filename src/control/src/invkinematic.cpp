#include "control/ik_controller.hpp"

#include <exception>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc != 1 && argc != 7) {
    std::cerr << "Usage: invkinematic [q1 q2 q3 q4 q5 q6] (radians)\n";
    return 1;
  }
  try {
    std::vector<double> positions{-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0};
    for (int i = 1; i < argc; ++i) {
      positions[i - 1] = std::stod(argv[i]);
    }
    const std::vector<std::string> names{
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"};
    control::IKController controller(
        ARM_MODEL_PATH, "attachment_site",
        pinocchio::SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0.35, 0.15, 0.5)));
    const auto target = controller.solve(names, positions);
    for (size_t i = 0; i < target.size(); ++i) {
      std::cout << names[i] << ": " << target[i] << '\n';
    }
  } catch (const std::exception &exc) {
    std::cerr << exc.what() << '\n';
    return 1;
  }
  return 0;
}
