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

#include <string>
#include <utility>
#include <limits>

#include "perception/filter_entity.hpp"

#include "behaviortree_cpp_v3/behavior_tree.h"


namespace perception
{

using namespace std::chrono_literals;
using namespace std::placeholders;

FilterEntity::FilterEntity(
  const std::string & xml_tag_name,
  const BT::NodeConfiguration & conf)
: BT::ActionNodeBase(xml_tag_name, conf)
{
  config().blackboard->get("node", node_);

  tf_buffer_ =
    std::make_unique<tf2_ros::Buffer>(node_->get_clock());
  tf_listener_ =
    std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  getInput("frame", frame_);
  getInput("lambda", lambda_);
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(node_);
  setOutput("filtered_frame", frame_  + "_filtered");


}

void
FilterEntity::halt()
{
  RCLCPP_INFO(node_->get_logger(), "FilterEntity halted");
}

geometry_msgs::msg::TransformStamped FilterEntity::update_state_observer(
  const geometry_msgs::msg::TransformStamped & entity)
{
  filtered_entity_.transform.translation.x = filtered_entity_.transform.translation.x + lambda_ *
    (entity.transform.translation.x - filtered_entity_.transform.translation.x);
  filtered_entity_.transform.translation.y = filtered_entity_.transform.translation.y + lambda_ *
    (entity.transform.translation.y - filtered_entity_.transform.translation.y);
  filtered_entity_.transform.translation.z = filtered_entity_.transform.translation.z + lambda_ *
    (entity.transform.translation.z - filtered_entity_.transform.translation.z);
  filtered_entity_.header.stamp = entity.header.stamp;

  return filtered_entity_;
}
geometry_msgs::msg::TransformStamped FilterEntity::initialize_state_observer(
  const geometry_msgs::msg::TransformStamped & entity)
{
  filtered_entity_ = entity;
  filtered_entity_.child_frame_id = entity.child_frame_id + "_filtered";
  return filtered_entity_;
}


BT::NodeStatus
FilterEntity::tick()
{
  RCLCPP_INFO(node_->get_logger(), "IsMoving ticked");

  geometry_msgs::msg::TransformStamped entity_transform_now_msg;
  rclcpp::Time when = node_->get_clock()->now();

  try {
    entity_transform_now_msg = tf_buffer_->lookupTransform(
      "map",
      frame_,
      tf2::TimePointZero);

  } catch (const tf2::TransformException & ex) {
    RCLCPP_INFO(
      node_->get_logger(), "Could not transform %s to %s: %s",
      frame_.c_str(), "map", ex.what());
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
  tf_broadcaster_->sendTransform(filtered_entity);
  setOutput("filtered_frame", filtered_entity.child_frame_id);
  return BT::NodeStatus::SUCCESS;
}

}  // namespace perception


BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<perception::FilterEntity>("FilterEntity");
}
