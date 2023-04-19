# from transformation2 import Transformation2D
# from transformation import Transformation2D
import math
import random
from transformation import Transformation2D
import numpy as np
import cv2
from PIL import Image
from pillow_heif import register_heif_opener
from star_finder import get_stars


def get_xy(star):
    x, y, r, b = star
    return (x, y)


def find_point(stars, mapped_point, e):
    for star in stars:
        x, y, r, b = star
        point = np.array([x, y])
        dist = np.linalg.norm(point-mapped_point)
        if dist <= e:
            return True
    return False


def count_intermidate_points(stars1, stars2, t, e=4):
    cnt = 0
    for star in stars1:
        x, y, r, b = star

        if (find_point(stars2, t((x, y)), e)):
            cnt += 1
    return cnt


def distance_point_to_line(m, x0, y0, x1, y1):
    # Find the y-intercept of the line
    b = y0 - m*x0

    # Convert the line equation to the standard form
    a = -m
    c = m*x1 - y1 + b

    # Calculate the minimum distance between the point and the line
    d = abs(a*x1 + b*y1 + c) / math.sqrt(a*a + b*b)
    return d


def line_cor(a, b, c, xlim=(0, 600), ylim=(600, 600)):
    # Compute two points on the line
    x1 = xlim[0]
    y1 = (-a*x1 - c) / b
    x2 = xlim[1]
    y2 = (-a*x2 - c) / b
    return (int(x1), int(y1), int(x2), int(y2))


def ransac_line_fit(points, threshold=10, max_iterations=10000):
    """
    RANSAC line fitting algorithm.

    Arguments:
    - points: a list of 2D points in the form [(x1, y1), (x2, y2), ...]
    - threshold: the maximum distance allowed between a point and the fitted line
    - max_iterations: the maximum number of iterations to run the RANSAC algorithm

    Returns:
    - best_line: a tuple (m, b) representing the equation of the fitted line y = mx + b
    """

    # Convert the list of points to a numpy array for easier indexing

    best_line = None
    best_score = 0
    points_on_line = []
    for i in range(max_iterations):
        # Choose two random points from the list
        idx = random.sample(range(len(points)), 2)
        p1 = points[idx[0]]
        p2 = points[idx[1]]
        curr_points = []
        # Compute the equation of the line between the two points
        if p2[0] == p1[0]:
            continue  # avoid division by zero
        a = p2[1] - p1[1]
        b = p1[0] - p2[0]
        c = p2[0]*p1[1] - p1[0]*p2[1]

        # Compute the distance between each point and the fitted line
        for point in points:
            d = abs(a * point[0] + b*point[1] + c) / math.sqrt(a ** 2 + b ** 2)
            if (d <= threshold):
                curr_points.append(point)

        # Update the best line if this iteration has more inliers than the previous best
        if len(curr_points) > best_score:
            best_line = line_cor(a, b, c)
            best_score = len(curr_points)
            points_on_line = curr_points

    return best_line, points_on_line


def map_stars(stars1, stars2, do_for=1000):
    '''
        1- pick two identical stars in each list
        2- make transformation matrix from stars1 -> stars2
        3- return the transformation function
    '''
    line1, points_on_line_1 = ransac_line_fit(stars1)
    line2, points_on_line_2 = ransac_line_fit(stars2)

    print(len(points_on_line_1), len(points_on_line_2))
    best_t = None
    current = -1
    tried_seq = []
    pick_cnt = 3
    s1 = random.sample(points_on_line_1, pick_cnt)

    for i in range(do_for):
        s2 = random.sample(points_on_line_2, pick_cnt)

        tup = (s1, s2)
        if (tup not in tried_seq):
            tried_seq.append(tup)
        else:
            # print('tried this', len(tried_seq), tup)
            continue
        # make transformations function
        t = Transformation2D([get_xy(s) for s in s1], [get_xy(s)
                             for s in s2]).make_transform_function()

        # check how many intermediate points
        count = count_intermidate_points(stars1, stars2, t)
        if (count > current):
            current = count - pick_cnt
            best_t = t

        mapped_stars = []
        for star in stars1:
            x, y, r, b = star
            m_x, m_y = best_t((x, y))
            mapped_stars.append((int(m_x), int(m_y)))

    return (mapped_stars, s1, s2, line1, points_on_line_1, line2, points_on_line_2)


# run main function
if __name__ == '__main__':
    register_heif_opener()

    path1 = './imgs/fr1.jpg'
    path2 = './imgs/fr2.jpg'
    size = (600, 600)

    img1 = np.array(Image.open(path1).resize(size))
    img2 = np.array(Image.open(path2).resize(size))

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    stars1 = get_stars(img1_gray, size)
    stars2 = get_stars(img2_gray, size)

    mapped_stars, source_points, dest_points, line1, points_on_line_1, line2, points_on_line_2 = map_stars(
        stars1, stars2)

    print(line1)
    print(line2)

    img1 = cv2.line(
        img1, (line1[0], line1[1]), (line1[2], line1[3]), (255, 0, 255), 1, 0)

    img2 = cv2.line(img2, (line2[0], line2[1]),
                    (line2[2], line2[3]), (255, 0, 255), 1, 0)

    for i in stars1:
        img1 = cv2.circle(img1, (i[0], i[1]), i[2], (255, 0, 255), 1, 0)

    for i in stars2:
        img2 = cv2.circle(img2, (i[0], i[1]), i[2], (255, 0, 255), 1, 0)

    for i, p in enumerate(points_on_line_1):
        x, y, r, b = p
        img1 = cv2.circle(img1, (x, y), r, (255, 0, 0), 1, 0)

    # for i, p in enumerate(points_on_line_2):
    #     x, y, r, b = p
    #     img2 = cv2.circle(img2, (x, y), 5, (255, 0, 0), 1, 0)

    for i, p in enumerate(mapped_stars):
        x, y = p
        img2 = cv2.circle(img2, (x, y), r, (255, 255, 255), 1, 0)

    for i, p in enumerate(source_points):
        x, y, r, b = p
        img1 = cv2.putText(img1, str(i), (int(x + r), int(y-r)), cv2.FONT_HERSHEY_SIMPLEX,
                           .5, (0, 255, 255), 1, cv2.LINE_AA)

    for i, p in enumerate(dest_points):
        x, y, r, b = p
        img2 = cv2.putText(img2, str(i), (int(x + 5), int(y-5)), cv2.FONT_HERSHEY_SIMPLEX,
                           .5, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('detected circles 1', img1)
    cv2.imshow('detected circles 2', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
