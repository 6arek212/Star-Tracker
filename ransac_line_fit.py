import math
import random


def line_cor(a, b, c, xlim=(0, 600), ylim=(600, 600)):
    # Compute two points on the line
    x1 = xlim[0]
    y1 = (-a*x1 - c) / b
    x2 = xlim[1]
    y2 = (-a*x2 - c) / b
    return (int(x1), int(y1), int(x2), int(y2))


def ransac_line_fit(points, threshold=100, max_iterations=10000):
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
