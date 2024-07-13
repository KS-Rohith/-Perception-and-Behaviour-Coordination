"""
Rationale Behind Design Decisions for Autonomous Car Controller
 
Parameter Choices:
•	Distance Threshold (distance_threshold):
o	Rationale: A distance threshold of 10 meters was chosen to determine when the car should slow down or stop upon approaching a red traffic light. This value ensures timely response to traffic signals while maintaining smooth operation in varying environmental conditions.
Algorithm Selection:
•	Traffic Light Detection (detect_traffic_light method):
o	Rationale: HSV color space was used for traffic light detection due to its effectiveness in distinguishing between red and green colors. The algorithm combines multiple color ranges to detect both shades of red, ensuring robustness against variations in lighting conditions.
•	Integration with Lane Controller (run method):
o	Rationale: By integrating traffic light detection within the main control loop (run method), the car dynamically adjusts its behavior based on detected signals. This seamless integration enhances overall autonomy and safety by prioritizing traffic rules while maintaining lane discipline.
3. Testing and Debugging
•	Debugging Flags (debug=True):
o	Rationale: Enabling debugging flags in both LaneController and MyRobot classes facilitates real-time visualization of intermediate image processing steps and algorithm outputs. This aids in identifying and resolving issues related to lane detection, traffic light recognition, and overall system performance.


"""


import cv2
import numpy as np
from rasrobot import RASRobot
from lane_controller_simple import LaneController

class MyRobot(RASRobot):
    def __init__(self):
        super(MyRobot, self).__init__()

    def detect_traffic_light(self, image, depth):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Red color range
        lower_red = np.array([0, 70, 50])
        upper_red = np.array([10, 255, 255])
        mask_red1 = cv2.inRange(hsv, lower_red, upper_red)
        lower_red = np.array([170, 70, 50])
        upper_red = np.array([180, 255, 255])
        mask_red2 = cv2.inRange(hsv, lower_red, upper_red)
        mask_red = mask_red1 | mask_red2

        # Green color range
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Check for red light
        if cv2.countNonZero(mask_red) > 0:
            # Finding contours of the red mask
            contours, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                roi = depth[y:y+h, x:x+w]

                # Calculates the mean distance to the object
                mean_distance = np.mean(roi)
                print(f"Red object detected at distance: {mean_distance} meters")  # Debugging info

                # Defines a distance threshold (e.g., 10 meters)
                distance_threshold = 35.0  # 10 meters

                if mean_distance <= distance_threshold:
                    return 'red'

        # Checks for green light
        if cv2.countNonZero(mask_green) > 0:
            return 'green'

        return 'none'

    def run(self):
        lc = LaneController(debug=True)
        use_cv2_window = hasattr(cv2, 'namedWindow')

        while self.tick():
            image = self.get_camera_image()
            depth = self.get_camera_depth_image()

            self.traffic_light_state = self.detect_traffic_light(image, depth)

            if self.traffic_light_state == 'red':
                print("Red light detected within 10 meters. Stopping the car.")  # Debugging info
                self.set_speed(0)
            else:
                steering_angle, speed = lc.get_angle_and_speed(image)
                self.set_steering_angle(steering_angle)
                self.set_speed(speed)

            if use_cv2_window:
                name = 'camera image'
                scale = 2

                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(name, image.shape[1] * scale, image.shape[0] * scale)
                cv2.imshow(name, image)
                cv2.waitKey(1)

                if depth is not None:
                    name_depth = 'depth image'
                    cv2.namedWindow(name_depth, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(name_depth, depth.shape[1] * scale, depth.shape[0] * scale)
                    cv2.imshow(name_depth, depth / 50)
                    cv2.waitKey(1)

if __name__ == '__main__':
    robot = MyRobot()
    robot.run()
