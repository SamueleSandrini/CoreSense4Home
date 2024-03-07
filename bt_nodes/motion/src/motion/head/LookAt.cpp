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

#include "motion/navigation/LookAt.hpp"

namespace navigation
{

LookAt::LookAt(
  const std::string & xml_tag_name,
  const BT::NodeConfiguration & conf)
: BT::ActionNodeBase(xml_tag_name, conf)
{
  config().blackboard->get("node", node_);
  attention_points_pub_ = node_->create_publisher<attention_system_msgs::msg::AttentionPoints>(
    "/attention/attention_points", 100);
  getInput("tf_frame", tf_frame_);
}

BT::NodeStatus
LookAt::tick()
{
  RCLCPP_INFO(node_->get_logger(), "LookAt ticked");

  attention_system_msgs::msg::AttentionPoints attention_points_msg;
  RCLCPP_INFO(node_->get_logger(), "LookAt tf_frame_: %s", tf_frame_.c_str());

  attention_points_msg.instance_id = "look_at";
  attention_points_msg.lifeness = rclcpp::Duration(5, 0);
  attention_points_msg.time_in_point = rclcpp::Duration(0, 0);

  geometry_msgs::msg::PointStamped point;
  point.header.frame_id = tf_frame_;
  point.point.x = 0.0;
  point.point.y = 0.0;
  point.point.z = 0.0;

  attention_points_msg.attention_points.push_back(point);
  attention_points_pub_->publish(attention_points_msg);

  rclcpp::spin_some(node_);
  RCLCPP_INFO(node_->get_logger(), "LookAt published attention points");
  return BT::NodeStatus::SUCCESS;
}

void
LookAt::halt()
{
  RCLCPP_INFO(node_->get_logger(), "LookAt halted");
}



}  // namespace navigation

BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<navigation::LookAt>("LookAt");
}