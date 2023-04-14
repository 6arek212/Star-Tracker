import numpy as np
from star_finder import detect_stars
import img_compare as cp
import cv2

image1 = cv2.imread('./imgs/1.JPG')
image2 = cv2.imread('./imgs/2.JPG')

image1_gray = cv2.imread('./imgs/1.JPG', cv2.IMREAD_GRAYSCALE)
image2_gray = cv2.imread('./imgs/2.JPG', cv2.IMREAD_GRAYSCALE)


image1_gray_resize = cv2.resize(image1_gray, (600, 600),
                                interpolation=cv2.INTER_LINEAR)
image2_gray_resize = cv2.resize(image2_gray, (600, 600),
                                interpolation=cv2.INTER_LINEAR)


image1_resize = cv2.resize(image1, (600, 600),
                           interpolation=cv2.INTER_LINEAR)
image2_resize = cv2.resize(image2, (600, 600),
                           interpolation=cv2.INTER_LINEAR)


def draw_circle(image, xy: tuple, radius, color=(0, 0, 0), text='', withText=False):
    x, y = xy
    img = cv2.circle(image, (int(x), int(y)), radius, color, 1)
    if withText:
        image = cv2.putText(image, text, (int(x + radius), int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX,
                            .5, color, 1, cv2.LINE_AA)
    return img


def draw_stars(stars, image, color=(255, 0, 255), withText=False):
    i = 0
    for star in stars:
        x = star[0]
        y = star[1]
        img = draw_circle(image, (x, y), 7, color, str(i), withText)
        i += 1
    return img


s1 = detect_stars(image1_gray_resize, intensity=0.22, max_radius=3)
s2 = detect_stars(image2_gray_resize, intensity=0.2, max_radius=3)
map_stars, v1, v2 = cp.map_stars(s1, s2)


image1_resize = draw_stars(s1, image1_resize, (0, 0, 255))
image2_resize = draw_stars(map_stars, image2_resize)

image1_resize = draw_stars(v1, image1_resize, (0, 255, 0), True)
image2_resize = draw_stars(v2, image2_resize, (0, 255, 0), True)

print(v1)
print(v2)

cv2.imshow('image1', image1_resize)
cv2.imshow('image2', image2_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()
