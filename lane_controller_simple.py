"""
Rationale Behind Design Decisions for Autonomous Car Controller
Parameter Choices:
•	Steering Row (self.steering_row):
o	Rationale: The choice of 100 for self.steering_row was based on the specific position in the image where the lane detection algorithm should focus. By adjusting this parameter, we can optimize the lane detection process for the specific setup and camera perspective.
•	Crossroad Row (self.crossroad_row):
o	Rationale: This parameter was set to 95 to define the row where crossroad detection begins. It helps in distinguishing between regular road sections and junctions, ensuring that the car maintains lane discipline until it reaches a junction.
•	Maximum Steering Angle (self.max_steering_angle):
o	Rationale: A steering angle of 0.25 radians was chosen to allow smooth turns without excessive oscillation. This value balances between responsiveness to lane deviations and stability during straight driving.
•	Speed Limits (self.max_speed and self.min_speed):
o	Rationale: The maximum speed of 45 units and minimum speed of 25 units were selected to provide flexibility in controlling the car's speed based on steering angle. Higher speeds are maintained on straight roads, while lower speeds ensure safer turns.
Algorithm Selection:
•	Image Processing (_get_road_mask method):
o	Rationale: Utilizing HSV color space for image processing allows robust detection of road surfaces and yellow lane markings. This choice is based on the distinct properties of color channels in HSV, which simplify thresholding operations compared to RGB.
•	Lane Center Calculation (_get_road_center method):
o	Rationale: The method calculates the center of the detected road by analyzing the binary mask. This approach leverages the spatial information provided by the mask to determine the car's lateral position relative to the lane markings.
•	Steering Angle Calculation (_get_steering_angle method):
o	Rationale: The steering angle is computed based on the deviation of the detected road center from a preferred position (-0.15 normalized). This method ensures that the car remains centered within the lane while allowing for necessary adjustments to follow curved paths.
•	Speed Control (_get_speed method):
o	Rationale: Speed adjustment is based on the current steering angle, ensuring smoother operation during turns by reducing speed proportionally to the magnitude of the steering angle. This approach enhances vehicle stability and safety while navigating curves.

"""

import cv2
import numpy as np

class LaneController:
    def __init__(self, debug=False):
        self.steering_row = 100  # Increased the row for steering determination
        self.crossroad_row = 95  # Increased the row for crossroad detection
        
        self.max_steering_angle = 0.25  # Reduced the maximum steering angle for smoother turns
        self.max_speed = 45  # Maximum speed in units 
        self.min_speed = 25  # Minimum speed in units 
        
        self.debug = debug  # Debug flag for displaying images
        self.prev_steering_angle = 0  # Previous steering angle for smoothing

    def _display_image(self, image, name, scale=2):
        """ Display image for debugging purposes. """
        if not self.debug:
            return
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, image.shape[1]*scale, image.shape[0]*scale)
        cv2.imshow(name, image)
        cv2.waitKey(1)

    def _get_road_mask(self, image):
        """ 
        TO Get binary mask for road detection.
        Adjusting color thresholds and morphology operations as needed.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_tarmac = np.array([0, 0, 0], dtype=np.uint8)
        upper_tarmac = np.array([200, 50, 100], dtype=np.uint8)
        mask_tarmac = cv2.inRange(hsv, lower_tarmac, upper_tarmac)

        lower_yellow_markings = np.array([25, 100, 100], dtype=np.uint8)
        upper_yellow_markings = np.array([50, 255, 255], dtype=np.uint8)
        mask_yellow_markings = cv2.inRange(hsv, lower_yellow_markings, upper_yellow_markings)

        mask_road = mask_tarmac | mask_yellow_markings
        mask_road = cv2.morphologyEx(mask_road, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        
        self._display_image(mask_road, 'road mask')  # Display for debugging
        return mask_road

    def _get_road_center(self, road_mask):
        """
        Calculates the center of the road from the binary mask.
        Return value in the range [-1, 1] where -1 means left, 0 means center, +1 means right.
        """
        road_img_line = road_mask[self.steering_row, :]
        if road_img_line.sum() == 0:
            return None
        
        idx = road_img_line.nonzero()[0]
        cx = np.mean(idx)
        center_normalized = (cx / len(road_img_line) - 0.5) * 2
        return center_normalized

    def _smooth_steering_angle(self, current_angle):
        """
        Smoothen the steering angle using a low-pass filter.
        """
        alpha = 0.1  # Smoothing factor
        smoothed_angle = alpha * current_angle + (1 - alpha) * self.prev_steering_angle
        self.prev_steering_angle = smoothed_angle
        return smoothed_angle

    def _get_steering_angle(self, road_center):
        """
        Calculates steering angle based on road center position.
        """
        if road_center is None:
            return self.prev_steering_angle
        
        preferred_road_center_pos = -0.15
        angle = road_center - preferred_road_center_pos
        angle = np.clip(angle, -self.max_steering_angle, self.max_steering_angle)
        smoothed_angle = self._smooth_steering_angle(angle)
        return smoothed_angle

    def _get_speed(self, steering_angle):
        """
        Calculates speed based on steering angle.
        Adjust speed calculations as needed.
        """
        speed_factor = (self.max_steering_angle - abs(steering_angle)) / self.max_steering_angle
        speed = (self.max_speed - self.min_speed) * speed_factor + self.min_speed
        return speed

    def get_angle_and_speed(self, image):
        """
        Main function to get steering angle and speed based on lane detection.
        """
        road_mask = self._get_road_mask(image)
        road_center = self._get_road_center(road_mask)
        steering_angle = self._get_steering_angle(road_center)
        speed = self._get_speed(steering_angle)
        
        return steering_angle, speed
