import dlib
import argparse
import time

import dlib
import cv2
import numpy as np

from utils import Side
from eye import Eye
from calibration import Calibration

class EyeTracker(object):
    def __init__(self):
        self.left_eye = None
        self.right_eye = None
        self.old_left_iris = None
        self.old_right_iris = None
        self.frame = None

        self.calibration = Calibration()

        self.face_detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("eyetracker/shape_predictor_68_face_landmarks.dat")

        self.irises = []
        self.blinks = None
        self.blink_in_previous = None
        self.lost_track = True

        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.blink_threshold = 9

    def optical_flow_filter(self):
        lost_track = False
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_frame, self.frame, self.irises, None, **self.lk_params)
        # cv2.imshow('image', self.old_frame)
        # cv2.waitKey(0)
        # cv2.imshow('image', self.frame)
        # cv2.waitKey(0)
        if st[0][0] == 0 or st[1][0] == 0:  # lost track on eyes
            lost_track = True
            self.blink_in_previous = False
        elif err[0][0] > self.blink_threshold or err[1][0] > self.blink_threshold:  # high error rate in klt tracking
            lost_track = True
            if not self.blink_in_previous:
                self.blinks += 1
                self.blink_in_previous = True
        else:
            self.blink_in_previous = False
            self.irises = []
            for w, h in p1:
                self.irises.append([w, h])
            self.irises = numpy.array(irises)
        return lost_track

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.left_eye.pupil.x)
            int(self.left_eye.pupil.y)
            int(self.right_eye.pupil.x)
            int(self.right_eye.pupil.y)
            return True
        except Exception:
            return False

    def run(self):
        """ Function to check frames for pupils and eyes.

        Arguments:

        """
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(frame)
        try:
            self.landmarks = self.predictor(frame, faces[0])
            self.left_eye = Eye(self.frame, self.landmarks, Side.LEFT, self.calibration)
            self.right_eye = Eye(self.frame, self.landmarks, Side.RIGHT, self.calibration)
            # if len(self.irises) >= 2:  # irises detected, track eyes
            #     self.lost_track = self.optical_flow_filter()
                
            #     if self.lost_track:
            #         self.landmarks = self.predictor(frame, faces[0])
            #         self.left_eye = Eye(self.frame, self.landmarks, Side.LEFT, self.calibration)
            #         self.right_eye = Eye(self.frame, self.landmarks, Side.RIGHT, self.calibration)
            #         self.irises = np.array([self.left_eye.pupil.get_position(), self.right_eye.pupil.get_position()])
            # else:  # cannot track for some reason -> find irises
            #     self.landmarks = self.predictor(frame, faces[0])
            #     self.left_eye = Eye(self.frame, self.landmarks, Side.LEFT, self.calibration)
            #     self.right_eye = Eye(self.frame, self.landmarks, Side.RIGHT, self.calibration)
            #     self.irises = np.array([self.left_eye.pupil.get_position(), self.right_eye.pupil.get_position()])
        except IndexError:
            self.left_eye = None
            self.right_eye = None

        

        self.old_frame = self.frame.copy()
        # self.old_left_iris = self.left_eye.pupil.iris_frame
        # self.old_right_iris = self.right_eye.pupil.iris_frame
        pass
    
    def refresh(self, frame):
        """ Function to set frame and rescan image.
        """
        self.frame = frame
        self.run()
        pass

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.left_eye.origin[0] + self.left_eye.pupil.x
            y = self.left_eye.origin[1] + self.left_eye.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.right_eye.origin[0] + self.right_eye.pupil.x
            y = self.right_eye.origin[1] + self.right_eye.pupil.y
            return (x, y)

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            pupil_left = self.left_eye.pupil.x / (self.left_eye.center[0] * 2 - 10)
            pupil_right = self.right_eye.pupil.x / (self.right_eye.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            pupil_left = self.left_eye.pupil.y / (self.left_eye.center[1] * 2 - 10)
            pupil_right = self.right_eye.pupil.y / (self.right_eye.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.35

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            print(self.horizontal_ratio())
            return self.horizontal_ratio() >= 0.65

    def is_center(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def is_blinking(self):
        """Returns true if the user closes his eyes"""
        if self.pupils_located:
            # return not self.lost_track
            ear_ratio = (self.left_eye.ear + self.right_eye.ear) / 2
            return ear_ratio < 0.24

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        return frame
