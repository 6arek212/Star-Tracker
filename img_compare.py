# from transformation2 import Transformation2D
# from transformation import Transformation2D
import random
from transformation import Transformation2D
import numpy as np


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


def map_stars(stars1, stars2, do_for=9000):
    '''
        1- pick two identical stars in each list
        2- make transformation matrix from stars1 -> stars2
        3- return the transformation function
    '''
    print(len(stars1), len(stars2))
    best_t = None
    current = -1
    tried_seq = []
    while (current <= 0):
        pick_cnt = 3
        s11, s12, s13 = random.sample(stars1, pick_cnt)
        s21, s22, s23 = random.sample(stars2, pick_cnt)

        tup = (s11, s12, s13, s21, s22, s23)
        if (tup not in tried_seq):
            tried_seq.append(tup)
        else:
            print('tried this', len(tried_seq), tup)
            continue
        # make transformations function
        t = Transformation2D([get_xy(s11), get_xy(s12), get_xy(s13)], [get_xy(
            s21), get_xy(s22), get_xy(s23)]).make_transform_function()

        # check how many intermediate points
        count = count_intermidate_points(stars1, stars2, t)
        print(count)
        if (count >= current):
            current = count - pick_cnt
            best_t = t

    print(current)
    mapped_stars = []
    for star in stars1:
        x, y, r, b = star
        mapped_stars.append(best_t((x, y)))

    return (mapped_stars, [s11, s12, s13], [s21, s22, s23])
