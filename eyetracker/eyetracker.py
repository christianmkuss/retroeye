from google.cloud import vision
import cv2
import numpy as np
import time

client = vision.ImageAnnotatorClient()

class EyeController:

    def __init__(self):
        self.left_eye_bb = None
        self.right_eye_bb = None
        self.left_pupil = None
        self.right_pupil = None
        self.gaze_dir = None

        self.offset_l = None
        self.offset_r = None

    def calc_gaze_dir(self):
        dir = "center"
        center_l = self.get_center_pt(self.left_eye_bb)
        center_r = self.get_center_pt(self.right_eye_bb)

        self.offset_l = self.left_pupil[0] - center_l[0]
        self.offset_r = self.right_pupil[0] - center_r[0]
        if self.offset_l < -1 and self.offset_r < 1:
            dir = "right"
        elif self.offset_l > 1 and self.offset_r > 1:
            dir = "left"
        else:
            dir = "center"

        self.gaze_dir = dir

    def get_gaze_dir(self):
        return self.gaze_dir

    def get_bbox(self, l, r, t, b):
        bbox = [[0, 0], [0, 0]]
        bbox[0][0] = int(l[0])
        bbox[0][1] = int(t[1])
        bbox[1][0] = int(r[0])
        bbox[1][1] = int(b[1])
        return bbox

    def get_center_pt(self, bbox):
        cx = int((bbox[1][0] + bbox[0][0])/2)
        cy = int((bbox[1][1] + bbox[0][1])/2)
        return [cx, cy]

    def detect_faces(self, img):
        print("Before call: {}".format(time.time()-self.initial_time))
        image = vision.types.Image(content=cv2.imencode('.jpg', img)[1].tostring())
        print("Between call: {}".format(time.time()-self.initial_time))
        response = client.face_detection(image=image)
        print("After call: {}".format(time.time()-self.initial_time))
        faces = response.face_annotations

        for face in faces:
            l_eye_top = l_eye_r_corner = l_eye_l_corner = l_eye_bottom = r_eye_bottom = r_eye_l_corner = r_eye_r_corner\
                = r_eye_top = 0
            for landmark in face.landmarks:
                if landmark.type == vision.enums.FaceAnnotation.Landmark.Type.LEFT_EYE_RIGHT_CORNER:
                    l_eye_r_corner = [landmark.position.x, landmark.position.y]
                    continue
                elif landmark.type == vision.enums.FaceAnnotation.Landmark.Type.LEFT_EYE_LEFT_CORNER:
                    l_eye_l_corner = [landmark.position.x, landmark.position.y]
                    continue
                elif landmark.type == vision.enums.FaceAnnotation.Landmark.Type.LEFT_EYE_TOP_BOUNDARY:
                    l_eye_top = [landmark.position.x, landmark.position.y]
                    continue
                elif landmark.type == vision.enums.FaceAnnotation.Landmark.Type.LEFT_EYE_BOTTOM_BOUNDARY:
                    l_eye_bottom = [landmark.position.x, landmark.position.y]
                    continue
                elif landmark.type == vision.enums.FaceAnnotation.Landmark.Type.RIGHT_EYE_RIGHT_CORNER:
                    r_eye_r_corner = [landmark.position.x, landmark.position.y]
                    continue
                elif landmark.type == vision.enums.FaceAnnotation.Landmark.Type.RIGHT_EYE_LEFT_CORNER:
                    r_eye_l_corner = [landmark.position.x, landmark.position.y]
                    continue
                elif landmark.type == vision.enums.FaceAnnotation.Landmark.Type.RIGHT_EYE_TOP_BOUNDARY:
                    r_eye_top = [landmark.position.x, landmark.position.y]
                    continue
                elif landmark.type == vision.enums.FaceAnnotation.Landmark.Type.RIGHT_EYE_BOTTOM_BOUNDARY:
                    r_eye_bottom = [landmark.position.x, landmark.position.y]
                    continue
                elif landmark.type == vision.enums.FaceAnnotation.Landmark.Type.LEFT_EYE:
                    x = int(landmark.position.x)
                    y = int(landmark.position.y)
                    self.left_pupil = [x, y]
                    cv2.circle(img, (x, y), 1, (0, 255, 255), 2)
                    continue
                elif landmark.type == vision.enums.FaceAnnotation.Landmark.Type.RIGHT_EYE:
                    x = int(landmark.position.x)
                    y = int(landmark.position.y)
                    self.right_pupil = [x, y]
                    cv2.circle(img, (x, y), 1, (0, 255, 255), 2)
                    continue

            self.left_eye_bb = self.get_bbox(l_eye_l_corner, l_eye_r_corner, l_eye_top, l_eye_bottom)
            self.right_eye_bb = self.get_bbox(r_eye_l_corner, r_eye_r_corner, r_eye_top, r_eye_bottom)
            self.calc_gaze_dir()
            # cv2.rectangle(img, (self.left_eye_bb[0][0], self.left_eye_bb[0][1]),
            #               (self.left_eye_bb[1][0], self.left_eye_bb[1][1]), (255, 255, 0), 2)
            # cv2.rectangle(img, (self.right_eye_bb[0][0], self.right_eye_bb[0][1]),
            #               (self.right_eye_bb[1][0], self.right_eye_bb[1][1]), (255, 255, 0), 2)

    def run_controller(self):
        self.frame_rate = 10
        self.prev = 0
        self.initial_time = time.time()
        cap = cv2.VideoCapture(0)
        while True:

            time_elapsed = time.time() - self.prev
            res, frame = cap.read()

            # if time_elapsed > 1./self.frame_rate:
            print(time_elapsed)
            self.prev = time.time()
            self.detect_faces(frame)
            # print("Looking %s\n" % self.get_gaze_dir())
            
            cv2.putText(frame, self.gaze_dir, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
            cv2.putText(frame, "Left pupil:  " + str(self.offset_l), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            cv2.putText(frame, "Right pupil: " + str(self.offset_r), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

        return


def main():
    controller = EyeController()
    controller.run_controller()
    return


if __name__ == "__main__":
    main()
    # img = cv2.imread('../image.jpg')
    # detect_faces(img)