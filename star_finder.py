import numpy as np
from math import sqrt


def has_intersecting_cicle(circles, circle):
    '''check if circle intersect with another circle'''

    x1, y1, radius1 = circle
    for x2, y2, radius2, b in circles:
        dist = sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if dist < radius1 + radius2:
            return True
    return False




def detect_stars(img: np.ndarray, intensity: float, max_radius: float):
    '''detect all stars and draw a red circle'''

    arr = np.asarray(img)
    radius = 5
    rows, cols = arr.shape
    circles = []
    for x in range(0, rows):
        for y in range(0, cols):
            b = img[x, y]
            if b/255 > intensity:
                if not has_intersecting_cicle(circles, (y, x, radius)):
                    circles.append((y, x, radius, b))

    return circles
