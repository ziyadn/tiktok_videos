import cv2
import imutils
from utils import PinZoom, GradZoom
import time


def click(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)


t = 1
while True:
    img = cv2.imread("test.jpg")
    facebox = [109, 189, 352, 559]
    H, W = img.shape[:2]
    img, facebox = GradZoom(img, facebox, t)
    x1, y1, x2, y2 = facebox
    h, w = y2 - y1, x2 - x1

    if h*w <= 0.6 * H*W:
        t += 1

    cv2.setMouseCallback("test", click, [])
    cv2.imshow("test", img)
    # cv2.imshow("test2", im)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break


cv2.destroyAllWindows()
print(t, h, w, H, W)
