import numpy as np
from math import sqrt
import cv2
from PIL import Image
from pillow_heif import register_heif_opener



# def load_image(path, size= 600 ,  grayscale=False):
#     image = np.array(Image.open(path))

#     if grayscale:
#         image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


# def has_intersecting_cicle(circles, circle):
#     '''check if circle intersect with another circle'''

#     x1, y1, radius1 = circle
#     for x2, y2, radius2, b in circles:
#         dist = sqrt((x2 - x1)**2 + (y2 - y1)**2)
#         if dist < radius1 + radius2:
#             return True
#     return False


def detect_stars(img: np.ndarray):
    '''detect all stars and draw a red circle'''
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=250, param2=.4, minRadius=3, maxRadius=6)
    return circles


# main runing function
def get_stars(image, size):
    circles = detect_stars(image)

    circles = np.uint16(np.around(circles))
    circles_cordinates = []
    for i in circles[0, :]:
        if i[0] < size[0] and i[1] < size[1]:
            circles_cordinates.append(
                (i[0], i[1], i[2] + 5,  image[i[0], i[1]]))
    return circles_cordinates


# run main function
if __name__ == '__main__':
    register_heif_opener()

    path1 = './imgs/IMG_3066.JPG'
    path2 = './imgs/IMG_3067.JPG'
    size = (600,600)

    img1 = np.array(Image.open(path1).resize(size))
    img2 = np.array(Image.open(path2).resize(size))
    
    img1_gray= cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray= cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


    stars1 = get_stars(img1_gray, size)
    stars2 = get_stars(img2_gray, size)

    # draw the outer circle
    for i in stars1:
        img1 = cv2.circle(img1, (i[0], i[1]), i[2], (255, 0, 255), 1, 0)

    for i in stars2:
        img2 = cv2.circle(img2, (i[0], i[1]), i[2], (255, 0, 255), 1, 0)

    cv2.imshow('detected circles 1', img1)
    cv2.imshow('detected circles 2', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
