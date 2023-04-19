import numpy as np
import cv2
from PIL import Image
from pillow_heif import register_heif_opener


def __detect_stars(img: np.ndarray):
    '''detect all stars and draw a red circle'''
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=250, param2=.4, minRadius=3, maxRadius=6)
    return circles


# main runing function
def get_stars(image: np.ndarray, size: tuple):
    circles = __detect_stars(image)

    circles = np.uint16(np.around(circles))
    circles_cordinates = []
    for i in circles[0, :]:
        if i[0] < size[0] and i[1] < size[1]:
            circles_cordinates.append(
                (int(i[0]), int(i[1]) , i[2] + 5,  image[i[0], i[1]]))
    return circles_cordinates


# run main function
if __name__ == '__main__':
    register_heif_opener()

    path1 = './imgs/IMG_3062.HEIC'
    path2 = './imgs/IMG_3063.HEIC'
    size = (600, 600)

    img1 = np.array(Image.open(path1).resize(size))
    img2 = np.array(Image.open(path2).resize(size))

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    stars1 = get_stars(img1_gray, size)
    stars2 = get_stars(img2_gray, size)

    print(stars1)

    # draw the outer circle
    for i in stars1:
        img1 = cv2.circle(img1, (i[0], i[1]), i[2], (255, 0, 255), 1, 0)

    for i in stars2:
        img2 = cv2.circle(img2, (i[0], i[1]), i[2], (255, 0, 255), 1, 0)

    cv2.imshow('detected circles 1', img1)
    cv2.imshow('detected circles 2', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
