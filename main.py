import argparse
import cv2
from function import FaceDetector, GradZoom, PinZoom
from pygame import mixer
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--host", default=0, type=int, help="source of webcam")
args = vars(ap.parse_args())

# named a opencv's window
wdwName = 'TikTok Bagaikan Langit'
cv2.namedWindow(wdwName)

# load the camera and set the resolution and fps
vs = cv2.VideoCapture(int(args["host"]))
vs.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
vs.set(cv2.CAP_PROP_FPS, 30)

# load class FaceDetector
fd = FaceDetector(args)
# initial read camera, to check ret=True
ret, img_camera = vs.read()

# initial t parameter for GradZoom function
t = 1

# initialize mixer for backsound
mixer.init()
mixer.music.load('bagaikan_langit.wav')
mixer.music.play()

while ret:
    ret, img_camera = vs.read()
    # if you want potrait camera, uncomment this 2 lines section
    # img_camera = cv2.transpose(img_camera)
    # img_camera = cv2.flip(img_camera, 0)

    img_camera = cv2.flip(img_camera, 1)
    # detecting face
    faceboxes, confidences = fd.get_faceboxes(img_camera)

    # if any face detected
    if len(faceboxes) > 0:
        # the algorithm may detect more than 1 face, filter it into only one
        facebox = faceboxes[0]
        H, W = img_camera.shape[:2]
        # GradZoom function with parameter t that increase if we continously detect face and crop 10%
        img_camera, facebox = GradZoom(img_camera, facebox, t, crop=0.1)
        x1, y1, x2, y2 = facebox
        h, w = y2 - y1, x2 - x1
        # check if the facebox not exceed 60% of image size
        if h * w <= 0.6 * H * W:
            t += 1

    # if no face detected, reset t parameter to 1
    else:
        t = 1

    # showing using opencv
    cv2.imshow(wdwName, img_camera)

    # used for break the loop if 'q' pressed using your keyboard
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

# after break the loop, close opencv's window
cv2.destroyAllWindows()
