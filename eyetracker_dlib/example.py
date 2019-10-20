import cv2
from eyetracker import EyeTracker

eyetrack = EyeTracker()
webcam = cv2.VideoCapture(0)

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to eyetrackTracking to analyze it
    eyetrack.refresh(frame)

    frame = eyetrack.annotated_frame()
    text = ""

    if eyetrack.is_blinking():
        text = "Blinking"
    elif eyetrack.is_right():
        text = "Looking right"
    elif eyetrack.is_left():
        text = "Looking left"
    elif eyetrack.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    # left_pupil = eyetrack.pupil_left_coords()
    # right_pupil = eyetrack.pupil_right_coords()
    # cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    # cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break
