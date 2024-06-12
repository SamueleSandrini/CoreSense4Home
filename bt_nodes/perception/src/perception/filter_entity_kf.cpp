// Copyright 2024 Intelligent Robotics Lab - Gentlebots
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "perception/filter_entity_kf.hpp"

#include <limits>
#include <string>
#include <utility>

#include "behaviortree_cpp_v3/behavior_tree.h"

namespace perception
{

using namespace std::chrono_literals;
using namespace std::placeholders;

FilterEntityKf::FilterEntityKf(const std::string & xml_tag_name, const BT::NodeConfiguration & conf)
: BT::ActionNodeBase(xml_tag_name, conf)
{
  config().blackboard->get("node", node_);

  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(node_->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  getInput("lambda", lambda_);
  getInput("update_dT", update_dt_);

  double dT;
  getInput("dT", dT);

  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(node_);
  setOutput("filtered_frame", frame_ + "_filtered");
  RCLCPP_INFO(node_->get_logger(), "FilterEntity initialized");

  // create eigen matrix for A matrix of the kalman filter for a constant velocity model in 3d x,y,z x_dot,y_dot,z_dot
  Eigen::MatrixXd A(6, 6);
  A <<  1, 0, 0, dT, 0, 0,  // x
        0, 1, 0, 0, dT, 0,  // y
        0, 0, 1, 0, 0, dT,  // z
        0, 0, 0, 1, 0, 0,  // x_dot
        0, 0, 0, 0, 1, 0,  // y_dot
        0, 0, 0, 0, 0, 1;  // z_dot
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(6, 1);
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(3, 6);
  C.diagonal().setOnes();

  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(6, 6);
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(3, 3);

  kf_ = std::make_shared<state_observer::KalmanFilter>(A, B, C, Q, R);
  // kf_ = state_observer::KalmanFilter(A,B,C,Q,R);
}

void FilterEntityKf::halt() {RCLCPP_INFO(node_->get_logger(), "FilterEntity halted");}

geometry_msgs::msg::TransformStamped FilterEntityKf::update_state_observer(
  const geometry_msgs::msg::TransformStamped & entity)
{
  if(update_dt_)
  {
    //update A if not fixed dT
    double dT;
    //compite dT as the difference between the current time and the last time ros2 time now
    // update A matrix
    dT = entity.header.stamp.sec + entity.header.stamp.nanosec * 1e9 - 
        filtered_entity_.header.stamp.sec - entity.header.stamp.nanosec * 1e9;
    Eigen::MatrixXd A(6, 6);
    A <<  1, 0, 0, dT, 0, 0,  // x
          0, 1, 0, 0, dT, 0,  // y
          0, 0, 1, 0, 0, dT,  // z
          0, 0, 0, 1, 0, 0,  // x_dot
          0, 0, 0, 0, 1, 0,  // y_dot
          0, 0, 0, 0, 0, 1;  // z_dot
    kf_->set_state_transition_matrix(A);
  }

  // measurements 
  Eigen::VectorXd z(3);
  z << entity.transform.translation.x, entity.transform.translation.y, entity.transform.translation.z;
  kf_->update(z);
  Eigen::VectorXd filtered_output = kf_->get_output();

  filtered_entity_.transform.translation.x =
    filtered_output(0);
  filtered_entity_.transform.translation.y =
    filtered_output(1);
  filtered_entity_.transform.translation.z =
    filtered_output(2);
  filtered_entity_.header.stamp = entity.header.stamp;
  return filtered_entity_;
}
geometry_msgs::msg::TransformStamped FilterEntityKf::initialize_state_observer(
  const geometry_msgs::msg::TransformStamped & entity)
{
  // Create a eigen vector with the x y z of the transform stamped entity 
  Eigen::VectorXd x0(6);
  x0 << entity.transform.translation.x, entity.transform.translation.y, entity.transform.translation.z, 0, 0, 0;
  kf_->initialize(x0);
  filtered_entity_ = entity;
  filtered_entity_.child_frame_id = entity.child_frame_id + "_filtered";
  return filtered_entity_;
}

BT::NodeStatus FilterEntityKf::tick()
{
  getInput("frame", frame_);
  RCLCPP_INFO(node_->get_logger(), "IsMoving filtering frame %s", frame_.c_str());

  geometry_msgs::msg::TransformStamped entity_transform_now_msg;

  try {
    entity_transform_now_msg = tf_buffer_->lookupTransform("odom", frame_, tf2::TimePointZero);
    RCLCPP_INFO(
      node_->get_logger(), "Position %s to %s: %f %f %f", frame_.c_str(), "odom",
      entity_transform_now_msg.transform.translation.x,
      entity_transform_now_msg.transform.translation.y,
      entity_transform_now_msg.transform.translation.z);
  } catch (const tf2::TransformException & ex) {
    RCLCPP_INFO(
      node_->get_logger(), "Could not transform %s to %s: %s", frame_.c_str(), "odom", ex.what());
    RCLCPP_INFO(node_->get_logger(), "Cannot transform");

    return BT::NodeStatus::SUCCESS;
  }
  geometry_msgs::msg::TransformStamped filtered_entity;
  if (state_obs_initialized_) {
    filtered_entity = update_state_observer(entity_transform_now_msg);
  } else {
    filtered_entity = initialize_state_observer(entity_transform_now_msg);
    state_obs_initialized_ = true;
  }
  filtered_entity.child_frame_id = frame_ + "_filtered";
  tf_broadcaster_->sendTransform(filtered_entity);
  setOutput("filtered_frame", filtered_entity.child_frame_id);
  return BT::NodeStatus::SUCCESS;
}

}  // namespace perception

BT_REGISTER_NODES(factory) {
  factory.registerNodeType<perception::FilterEntityKf>("FilterEntityKf");
}
