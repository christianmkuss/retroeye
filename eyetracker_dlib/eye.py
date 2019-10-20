from scipy.spatial import distance as dist
import dlib
import cv2
import numpy as np
import math

from utils import Side
from pupil import Pupil

class Eye(object):
    """
    Class for identifying eyes within frames and analyzing for basic
    movement, such as blinking.
    """
    def __init__(self, frame, landmarks, side, calibration):
        LEFT_EYE_LANDMARKS = [36, 37, 38, 39, 40, 41]
        RIGHT_EYE_LANDMARKS = [42, 43, 44, 45, 46, 47]

        self.ear = None
        self.center = None
        self.origin = None

        # self.old_iris = old_iris

        self.calibration = calibration
        self.frame = frame

        self.side = side

        if side == Side.LEFT:
            points = LEFT_EYE_LANDMARKS
        elif side == Side.RIGHT:
            points = RIGHT_EYE_LANDMARKS
        else:
            return

        self.region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])

        self.run()

    
    def eye_aspect_ratio(self):
        # Compute euclidean distances between vertical and horizontal
        # eye landmark points
        left_vert = dist.euclidean(self.region[1], self.region[5])
        right_vert = dist.euclidean(self.region[2], self.region[4])

        horiz = dist.euclidean(self.region[0], self.region[3])

        # Compute eye aspect ratio
        ear = (left_vert + right_vert)/(2.0 * horiz)

        return ear

    @staticmethod
    def middle_point(p1, p2):
        x = int((p1[0] + p2[0]) / 2)
        y = int((p1[1] + p2[1]) / 2)
        return (x, y)

    # def eye_aspect_ratio(self):
    #     left = (self.region[0][0], self.region[0][1])
    #     right = (self.region[3][0], self.region[3][1])
    #     top = self.middle_point(self.region[1], self.region[2])
    #     bottom = self.middle_point(self.region[5], self.region[4])

    #     eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
    #     eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))

    #     try:
    #         ratio = eye_width / eye_height
    #     except ZeroDivisionError:
    #         ratio = None

    #     return ratio


    def find_eye(self):
        region = self.region.astype(np.int32)

        # Get height and width of frame
        height, width = self.frame.shape[:2]

        # Create frame for filling later
        empty_frame = np.zeros((height, width), np.uint8)

        # Fill area of frame with black
        mask = np.full((height, width), 255, np.uint8)

        # Fill in region of eye with white
        cv2.fillPoly(mask, [region], (0, 0, 0))

        # Return frame of eye based on mask
        eye = cv2.bitwise_not(empty_frame, self.frame.copy(), mask=mask)

        # Cropping on the eye
        margin = 5
        min_x = np.min(region[:, 0]) - margin
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin

        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)

        height, width = self.frame.shape[:2]
        self.center = (width / 2, height / 2)


    def run(self):
        self.ear = self.eye_aspect_ratio()
        self.find_eye()


        if not self.calibration.is_complete():
            self.calibration.evaluate(self.frame, self.side)

        self.threshold = self.calibration.threshold(self.side)

        self.pupil = Pupil(self.frame, self.threshold)
        # self.old_iris = self.pupil.old_iris
        pass
    pass