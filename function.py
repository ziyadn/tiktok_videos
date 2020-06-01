import imutils
import cv2
import os


class FaceDetector(object):

    def __init__(self, args={}):
        self.face_detector = cv2.dnn.readNetFromCaffe(os.path.join(
            "detector", "deploy.prototxt.txt"), os.path.join("detector", "res10_300x300_ssd_iter_140000.caffemodel"))
        self.threshold = 0.88
        self.color1 = (3, 252, 186)
        self.color2 = (0, 0, 0)

    def get_faceboxes(self, image):
        H, W = image.shape[:2]

        confidences = []
        faceboxes = []
        self.face_detector.setInput(cv2.dnn.blobFromImage(
            image, 1.0, (150, 150), (104.0, 177.0, 123.0), False, False))
        detections = self.face_detector.forward()

        for result in detections[0, 0, :, :]:
            confidence = result[2]
            if confidence > self.threshold:
                x_left_bottom = int(result[3] * W)
                y_left_bottom = int(result[4] * H)
                x_right_top = int(result[5] * W)
                y_right_top = int(result[6] * H)
                confidences.append(confidence)
                faceboxes.append(
                    [x_left_bottom, y_left_bottom, x_right_top, y_right_top])

        detection_result = [faceboxes, confidences]

        return detection_result

    def draw_boxes(self, image, faceboxes):
        for facebox in faceboxes:
            (x1, y1, x2, y2) = facebox

            # Draw border
            scala_x = (x2 - x1) // 3
            scala_y = (y2 - y1) // 3

            # duplicate-1
            self.color1 = (3, 252, 186)
            cv2.line(image, (x1, y1), (x1 + scala_x, y1), self.color1, 2)
            cv2.line(image, (x1, y1), (x1, y1 + scala_y), self.color1, 2)

            cv2.line(image, (x1, y2), (x1 + scala_x, y2), self.color1, 2)
            cv2.line(image, (x1, y2), (x1, y2 - scala_y), self.color1, 2)

            cv2.line(image, (x2, y1), (x2 - scala_x, y1), self.color1, 2)
            cv2.line(image, (x2, y1), (x2, y1 + scala_y), self.color1, 2)

            cv2.line(image, (x2, y2), (x2 - scala_x, y2), self.color1, 2)
            cv2.line(image, (x2, y2), (x2, y2 - scala_y), self.color1, 2)

            # duplicate-2

            cv2.line(image, (x1, y1), (x1 + scala_x, y1), self.color2, 1)
            cv2.line(image, (x1, y1), (x1, y1 + scala_y), self.color2, 1)

            cv2.line(image, (x1, y2), (x1 + scala_x, y2), self.color2, 1)
            cv2.line(image, (x1, y2), (x1, y2 - scala_y), self.color2, 1)

            cv2.line(image, (x2, y1), (x2 - scala_x, y1), self.color2, 1)
            cv2.line(image, (x2, y1), (x2, y1 + scala_y), self.color2, 1)

            cv2.line(image, (x2, y2), (x2 - scala_x, y2), self.color2, 1)
            cv2.line(image, (x2, y2), (x2, y2 - scala_y), self.color2, 1)

        return image


def PinZoom(image, facebox, crop=0.05):

    H, W = image.shape[:2]
    scale_h, scale_w = int(H * crop), int(W * crop)
    x1, y1, x2, y2 = facebox
    height, width = y2 - y1, x2 - x1
    centroid_im = [W//2, H//2]
    centroid_fb = [(x1+x2)//2, (y1+y2)//2]

    if centroid_fb[0] > centroid_im[0]:
        im_cropped = image[:, scale_w:].copy()
        centroid_fb = (centroid_fb[0] - scale_w, centroid_fb[1])

    else:
        im_cropped = image[:, :W-scale_w].copy()

    if centroid_fb[1] > centroid_im[1]:
        im_cropped = im_cropped[scale_h:, :]
        centroid_fb = (centroid_fb[0], centroid_fb[1] - scale_h)

    else:
        im_cropped = im_cropped[:H-scale_h, :]

    centroid_im = ((W-scale_w) // 2, (H-scale_h) // 2)

    im_cropped = imutils.resize(im_cropped, width=W)
    x1, y1, x2, y2 = int(centroid_fb[0] - (width//2)), int(centroid_fb[1] -
                                                           (height//2)), int(centroid_fb[0] + (width//2)), int(centroid_fb[1] + (height//2))
    new_facebox = [int(x * 1/(1 - crop)) for x in [x1, y1, x2, y2]]

    return im_cropped, new_facebox


def GradZoom(im_cropped, new_facebox, t, crop):

    count = 0
    while count < t:
        im_cropped_, new_facebox_ = PinZoom(
            im_cropped, new_facebox, crop=crop)
        count += 1
        im_cropped = im_cropped_.copy()
        new_facebox = new_facebox_.copy()

    return im_cropped, new_facebox
