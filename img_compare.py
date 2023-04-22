import random
import numpy as np
import cv2
from PIL import Image
from pillow_heif import register_heif_opener
from display_imgs import show_data
from ransac_line_fit import ransac_line_fit
from star_finder import get_stars


# create the transformation matrix based on two sets of points
def create_mapper(source, destination):
    source = np.float32(source)
    destination = np.float32(destination)
    M = cv2.getAffineTransform(source, destination)

    def mapper(point):
        transformed_pt = np.dot(M, np.array([point[0], point[1], 1]))
        return (int(transformed_pt[0]), int(transformed_pt[1]))
    return mapper


# find mapped star in stars
def find_point(stars, source_point, mapped_point, e):
    x2, y2, r2, b2 = mapped_point

    for star in stars:
        x1, y1, r1, b1 = star
        # point = np.array([x, y])
        # dist = np.linalg.norm(point-mapped_point)
        dist_from_source = np.sqrt(
            (source_point[0]-x2) ** 2 + (source_point[1]-y2) ** 2)
        dist_s = np.sqrt((x1-x2) ** 2 + (y1-y2) ** 2)
        if dist_s < e and abs(r1 - r2) < 6:
            return star
    return None


# count the number of good points that are mapped correctly by 'T' from star1 to stars2 with error margin 'e'
def count_inliers(stars1, stars2, T, e=16):
    cnt = 0
    matchings = {}

    for star in stars1:
        x, y, r, b = star
        mapped_point = T((x, y))
        matched_point = find_point(
            stars2, star,  (mapped_point[0], mapped_point[1], r, b), e)

        matchings[matched_point] = matchings.get(matched_point, 0) + 1

        if (matched_point is not None and matchings[matched_point] <= 1):
            cnt += 1
    return cnt


# def find_good_triangle(points, iterations=100, e=3):

#     for i in range(iterations):
#         # Pick three random points from the set
#         p1, p2, p3 = random.sample(points, 3)

#         # Check that the points are not collinear
#         x1, y1 = (p1[0], p1[1])
#         x2, y2 = (p2[0], p2[1])
#         x3, y3 = (p3[0], p3[1])

#         if (y2 - y1) * (x3 - x2) == (y3 - y2) * (x2 - x1):
#             continue

#         # Check that the angles formed by the three points are greater than zero
#         a = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#         b = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
#         c = math.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
#         cos_a = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
#         cos_b = (c ** 2 + a ** 2 - b ** 2) / (2 * c * a)
#         cos_c = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)

#         if cos_a <= 0 or cos_b <= 0 or cos_c <= 0:
#             continue
#         return (p1, p2, p3)


def matching_ratio(set_1, set_2, inliers_count):
    length = min(len(set_1), len(set_2))
    return inliers_count / length


# map stars with constant number of iterations
def map_stars(stars1, stars2, iteration=10000):
    '''
        1- pick two identical stars in each list
        2- make transformation matrix from stars1 -> stars2
        3- return the transformation function
    '''
    line1, points_on_line_1 = ransac_line_fit(stars1)
    line2, points_on_line_2 = ransac_line_fit(stars2)

    # print('-----', len(points_on_line_1), len(points_on_line_2))
    best_t = None
    inliers_count = -1
    tried_seq = []
    pick_cnt = 3
    source_1 = None
    source_2 = None

    if (len(points_on_line_1) > 16):
        points_on_line_1 = random.sample(points_on_line_1, 16)

    if (len(points_on_line_2) > 16):
        points_on_line_2 = random.sample(points_on_line_2, 16)

    # s1 = find_good_triangle(points_on_line_1)
    for i in range(iteration):
        s1 = random.sample(points_on_line_1, pick_cnt)
        s2 = random.sample(points_on_line_2, pick_cnt)

        tup = (s1, s2)
        if (tup not in tried_seq):
            tried_seq.append(tup)
        else:
            # print('tried this', len(tried_seq), s2)
            continue

        # make transformations function
        a = np.array([(p[0], p[1]) for p in s1])
        b = np.array([(p[0], p[1]) for p in s2])
        t = create_mapper(a, b)

        # check how many inliers points
        count = count_inliers(points_on_line_1, points_on_line_2, t)

        if (count > inliers_count):
            inliers_count = count
            best_t = t
            source_1 = s1
            source_2 = s2

    mapped_stars = []
    for i, star in enumerate(points_on_line_1):
        x, y = best_t((star[0], star[1]))
        mapped_stars.append([(star[0], star[1]), (x, y)])

    return (mapped_stars, source_1, source_2, line1, points_on_line_1, line2, points_on_line_2, matching_ratio(points_on_line_1, points_on_line_2, inliers_count))


def save_mapped_stars(output_path, mapped_stars,  size: tuple, ratio=None):
    with open(output_path, "w") as f:
        f.write(f"source dest\n")
        for s_d_star in mapped_stars:
            source_star, dest_star = s_d_star
            if dest_star[0] < size[1] and dest_star[1] < size[0] and dest_star[0] >= 0 and dest_star[1] >= 0:
                f.write(f"{source_star} , {dest_star}\n")
            else:
                f.write(f"{source_star} , no matching \n")
        if (matching_ratio is not None):
            f.write(f'\nmatching ratio: {ratio}')


# run main function
if __name__ == '__main__':
    register_heif_opener()

    path1 = './imgs/fr1.jpg'
    path2 = './imgs/fr2.jpg'
    # path1 = './imgs/1.jpg'
    # path2 = './imgs/2.jpg'
    path1 = './imgs/ST_db1.png'
    path2 = './imgs/ST_db2.png'
    # path1 = './imgs/IMG_3053.HEIC'
    # path2 = './imgs/IMG_3054.HEIC'
    size = (600, 600)

    img1 = np.array(Image.open(path1).resize(size))
    img2 = np.array(Image.open(path2).resize(size))

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    stars1 = get_stars(img1_gray, size)
    stars2 = get_stars(img2_gray, size)

    mapped_stars, source_points, dest_points, line1, points_on_line_1, line2, points_on_line_2, inliers_count = map_stars(
        stars1, stars2)

    # print(mapped_stars)
    show_data(source_points, dest_points,
              points_on_line_1, line1, points_on_line_2, line2,  mapped_stars, img1, img2)
