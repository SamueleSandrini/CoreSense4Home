#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer
from rclpy.publisher import Publisher

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from dataclasses import dataclass, field
from yolov8_msgs.msg import DetectionArray, Detection
import cv2
from typing import List

@dataclass
class DepthProcessorNode(Node):

    depth_sub: Subscriber = field(default=None, init=False)
    yolo_sub: Subscriber = field(default=None, init=False)
    sync_subs: TimeSynchronizer = field(default=None, init=False)
    depth_pub: Publisher = field(default=None, init=False)
    cv_bridge: CvBridge = field(default=None, init=False)
    depth_image: np.ndarray = field(default=None, init=False)
    detected_objects: DetectionArray = field(default=None, init=False)

    def __post_init__(self):
        super().__init__('depth_processor_node')

        self.depth_pub = self.create_publisher(
            Image,
            '/processed_depth',
            10
        )
        self.depth_sub = Subscriber(self, Image, '/head_front_camera/depth/image_raw')
        self.detection_sub = Subscriber(self, DetectionArray, '/perception_system/detections_3d')

        self.cv_bridge = CvBridge()
        self.sync_subs = ApproximateTimeSynchronizer([self.depth_sub, self.detection_sub], 10,0.1)
        self.sync_subs.registerCallback(self.callback)

    def callback(self, depth_msg: Image, detection_msg: DetectionArray):
        if not depth_msg.header.frame_id == detection_msg.header.frame_id:
            self.get_logger().warn(f"Depth Frames id: {depth_msg.header.frame_id}")
            self.get_logger().warn(f"Detection Frames id: {detection_msg.header.frame_id}")
            
            self.get_logger().warn("Frames id do not match")
            # return
        detected_objects = detection_msg.detections
        
        try:
            depth_frame = self.convert_depth_msg_to_cv2(depth_msg)
        except CvBridgeError as exc:
            self.get_logger().error("Error while converting depth image to cv2")
            return
        
        filtered_depth_image = self.remove_objects_from_depth(depth_frame, detected_objects)

        self.depth_pub.publish(self.cv_bridge.cv2_to_imgmsg(filtered_depth_image, encoding='passthrough'))

    def convert_depth_msg_to_cv2(self, image_msg: Image):
        try:
            depth_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
        except CvBridgeError as exc:
            raise exc("Error while converting depth image to cv2")
        return depth_image

    def remove_objects_from_depth(self, depth_image: np.ndarray, detected_objects: List[Detection]):
        # mask.astype(np.uint8)
        # mask = np.zeros(depth_image.shape[:2], dtype=depth_image.dtype)
        image_width =  depth_image.shape[0]
        image_height =depth_image.shape[1]
        print("New frame")
        for detected_object in detected_objects:
            if detected_object.class_name == "person":
                continue
            center_x = detected_object.bbox.center.position.x
            center_y = detected_object.bbox.center.position.y
            print(f"Center: ({center_x},{center_y})")
            width = detected_object.bbox.size.x
            height = detected_object.bbox.size.y
            print(f"Width Height: ({width},{height})")

            top_left_x = int(center_x - width / 2)
            top_left_y = int(center_y - height / 2)
            bottom_right_x = int(center_x + width / 2)
            bottom_right_y = int(center_y + height / 2)
            top_left_x = max(0, top_left_x)
            top_left_y = max(0, top_left_y)
            bottom_right_x = min(image_width - 1, bottom_right_x)
            bottom_right_y = min(image_height - 1, bottom_right_y)

            # max_value = np.iinfo(mask.dtype).max

            # Disegna il rettangolo sulla maschera
            cv2.rectangle(depth_image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), 0, -1)
        # print(type(mask))
        # print(mask.dtype)
        # print(depth_image.dtype)
        # print(type(depth_image))
        # Applica la maschera all'immagine
        # masked = cv2.bitwise_and(depth_image, depth_image, mask=~mask)
        return depth_image


    def __eq__(self, other):
        return isinstance(other, DepthProcessorNode) and \
               self.get_name() == other.get_name()

    def __hash__(self):
        return hash(self.get_name())

def main(args=None):
    rclpy.init(args=args)
    depth_processor_node = DepthProcessorNode()
    try:
        rclpy.spin(depth_processor_node)
    except KeyboardInterrupt:
        pass

    depth_processor_node.destroy_node()
    # depth_processor_node.destroy_node()
    # rclpy.shutdown()

if __name__ == '__main__':
    main()

    # def depth_callback(self, image_msg: Image):
    #     depth_image = self.convert_depth_msg_to_cv2(image_msg)
    #     detected_objects = self.detected_objects.copy()
    #     if detected_objects is not None:
    #         for detected_object in self.detected_objects:
    #             filtered_depth_image = self.remove_object_from_depth(depth_image, detected_object)
    #             self.depth_pub.publish(self.cv_bridge.cv2_to_imgmsg(filtered_depth_image, encoding='passthrough'))
    #         pass

    # def yolo_callback(self, msg: DetectionArray):
    #     self.detected_objects = msg.detections
    #     # for detected_object in msg.detections:
    #     #     self.depth_image = self.remove_object_from_depth(detected_object)
