from google.cloud import vision
import cv2
import numpy as np
import time

client = vision.ImageAnnotatorClient()

class EyeController:
    """Class for handling eye tracking and returning outputs for direction
    of gaze and state of head nod.
    """
    def __init__(self):
        # Set default values for all internal variables
        self.left_eye_bb = None
        self.right_eye_bb = None
        self.left_pupil = None
        self.right_pupil = None
        self.gaze_dir = None

        self.offset_l = None
        self.offset_r = None


    def calc_gaze_dir(self):
        """Calculate direction of gaze for eyes.
        """
        # Generate center position of each eye
        center_l = self.get_center_pt(self.left_eye_bb)
        center_r = self.get_center_pt(self.right_eye_bb)

        # Generate offset of pupils to centers of eyes
        self.offset_l = self.left_pupil[0] - center_l[0]
        self.offset_r = self.right_pupil[0] - center_r[0]

        # Check which if pupils are more on the left side of the face,
        # right side of the face, or center of the face
        if self.offset_l < -1 and self.offset_r < 1:
            dir = "right"
        elif self.offset_l > 1 and self.offset_r > 1:
            dir = "left"
        else:
            dir = "center"
        
        self.gaze_dir = dir


    def get_gaze_dir(self):
        """Retrieve direction of eyes.
        """
        return self.gaze_dir


    def get_bbox(self, l, r, t, b):
        """Retrieve bounding box of landmark.
        """
        bbox = [[0, 0], [0, 0]]
        bbox[0][0] = int(l[0])
        bbox[0][1] = int(t[1])
        bbox[1][0] = int(r[0])
        bbox[1][1] = int(b[1])
        return bbox


    def get_center_pt(self, bbox):
        """Retrieve center point of bounding box.
        """
        cx = int((bbox[1][0] + bbox[0][0])/2)
        cy = int((bbox[1][1] + bbox[0][1])/2)
        return [cx, cy]


    def get_nod(self):
        """Evaluate if head tilt passes threshold to become nod.
        """
        if self.tilt_angle < -10:
            self.gaze_dir = "Nod"
        pass


    def detect_faces(self, img):
        """Generate values for eye landmarks and head tilt.
        """
        image = vision.types.Image(content=cv2.imencode('.jpg', img)[1].tostring())
        response = client.face_detection(image=image)
        faces = response.face_annotations

        for face in faces:
            # Set all variables assosciated with landmarks to zero
            l_eye_top = l_eye_r_corner = l_eye_l_corner = l_eye_bottom = r_eye_bottom = r_eye_l_corner = r_eye_r_corner\
                = r_eye_top = 0

            # Establish head tilt angle
            self.tilt_angle = face.tilt_angle

            # Assign variables that match landmark features, specifically eye landmarks
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
            #  Form bounding boxes for left eye and right eye
            self.left_eye_bb = self.get_bbox(l_eye_l_corner, l_eye_r_corner, l_eye_top, l_eye_bottom)
            self.right_eye_bb = self.get_bbox(r_eye_l_corner, r_eye_r_corner, r_eye_top, r_eye_bottom)
            
            # Identify direction of gaze
            self.calc_gaze_dir()

            # Identify if head is in nodding position
            self.get_nod()
            

    def run_controller(self):
        """Grabs camera images and detects eye movement/head nodding
        """
        cap = cv2.VideoCapture(0)
        while True:
            res, frame = cap.read()
            self.detect_faces(frame)
            
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