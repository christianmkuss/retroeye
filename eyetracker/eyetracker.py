from google.cloud import vision
import cv2
from flask import Flask, request
from flask_restful import Resource, Api
from flask.ext.jsonpify import jsonify
import threading

app = Flask(__name__)
api = Api(app)

client = vision.ImageAnnotatorClient()


class EyeController(Resource):
    def get(self):
        result = {'direction': self.get_gaze_dir()}
        return jsonify(result)

    def __init__(self):
        self.left_eye_bb = None
        self.right_eye_bb = None
        self.left_pupil = None
        self.right_pupil = None
        self.gaze_dir = None

    def calc_gaze_dir(self):
        dir = "center"
        center_l = self.get_center_pt(self.left_eye_bb)
        center_r = self.get_center_pt(self.right_eye_bb)
        if (center_l[0] - self.left_pupil[0]) < 0 or (center_r[0] - self.right_pupil[0]) < 0:
            dir = "right"
        if (center_l[0] - self.left_pupil[0]) > 0 or (center_r[0] - self.right_pupil[0]) > 0:
            dir = "left"

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
        cx = int(bbox[1][0] - bbox[0][0]/2)
        cy = int(bbox[1][1] - bbox[0][1]/2)
        return [cx, cy]

    def detect_faces(self, img):

        image = vision.types.Image(content=cv2.imencode('.jpg', img)[1].tostring())
        response = client.face_detection(image=image)
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
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            self.detect_faces(frame)
            print("Looking %s\n" % self.get_gaze_dir())
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

        return


api.add_resource(EyeController, '/direction')


def main():
    controller = EyeController()
    controller.run_controller()
    return

def runFlask():
    app.run(port=5000, use_reloader=False)


if __name__ == "__main__":
    threading.Thread(target=runFlask).start()
    main()
