import cv2
import numpy as np

class Pupil(object):
    """
    Class for parsing frames for pupils and identifying position.
    """
    def __init__(self, eye_frame, threshold):
        self.eye_frame = eye_frame
        self.threshold = threshold

        self.iris_frame = None
        self.x = None
        self.y = None

        # self.old_iris = old_iris
        # self.irises = None
        # self.blinks = None
        # self.blink_in_previous = None

        # # Parameters for lucas kanade optical flow
        # self.lk_params = dict(winSize=(15, 15),
        #                  maxLevel=2,
        #                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # self.blink_threshold = blink_threshold

        self.find_iris()
        pass

    @staticmethod
    def filter_iris(eye_frame, threshold):
        """Filters image to return region with iris.
        """
        kernel = np.ones((3, 3), np.uint8)
        new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)
        new_frame = cv2.erode(new_frame, kernel, iterations=3)
        new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1]

        return new_frame
    
    def get_position(self):
        return [np.float32(self.x), np.float32(self.y)]

    def find_iris(self):
        self.iris_frame = self.filter_iris(self.eye_frame, self.threshold)

        contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        contours = sorted(contours, key=cv2.contourArea)

        try:
            moments = cv2.moments(contours[-2])
            self.x = int(moments['m10'] / moments['m00'])
            self.y = int(moments['m01'] / moments['m00'])
        except (IndexError, ZeroDivisionError):
            pass

    # def optical_flow_filter(self):
    #     lost_track = False
    #     p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_iris, self.iris_frame, np.array([self.x, self.y]), None, **self.lk_params)
    #     if st[0][0] == 0 or st[1][0] == 0:  # lost track on eyes
    #         lost_track = True
    #         self.blink_in_previous = False
    #     elif err[0][0] > self.blink_threshold or err[1][0] > self.blink_threshold:  # high error rate in klt tracking
    #         lost_track = True
    #         if not self.blink_in_previous:
    #             blinks += 1
    #             blink_in_previous = True
    #     else:
    #         blink_in_previous = False
    #         irises = []
    #         for w, h in p1:
    #             irises.append([w, h])
    #         irises = numpy.array(irises)
    #     return irises, blinks, blink_in_previous, lost_track
    
    # def run(self):
    #     if not self.x is None and not self.y is None:  # irises detected, track eyes
    #         lost_track = self.optical_flow_filter()
    #         if lost_track:
    #             self.find_iris()
    #     else:  # cannot track for some reason -> find irises
    #         self.find_iris()

    #     show_image_with_data(frame, self.blinks, self.irises)
    #     k = cv2.waitKey(30) & 0xff
    #     old_gray = gray.copy()

    # pass