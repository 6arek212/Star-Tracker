# from transformation2 import Transformation2D
# from transformation import Transformation2D
from transformation import Transformation2D


def pick_brightes_stars(stars):
    max_1 = float('-inf')
    max_2 = float('-inf')
    max_3 = float('-inf')

    max_1_star = None
    max_2_star = None
    max_3_star = None

    for star in stars:
        x, y, r, b = star
        if (max_1 < b):
            max_3 = max_2
            max_3_star = max_2_star

            max_2 = max_1
            max_1 = b

            max_2_star = max_1_star
            max_1_star = star

        elif (max_2 < b):
            max_3 = max_2
            max_3_star = max_2_star
            max_2 = b
            max_2_star = star

        elif (max_3 < b):
            max_3 = b
            max_3_star = star

    return (max_1_star, max_2_star, max_3_star)


def get_xy(star):
    x, y, r, b = star
    return (x, y)


def map_stars(stars1, stars2):
    '''
        1- pick two identical stars in each list
        2- make transformation matrix from stars1 -> stars2
        3- return the transformation function
    '''
    s11, s12, s13 = pick_brightes_stars(stars1)
    s21, s22, s23 = pick_brightes_stars(stars2)

    t = Transformation2D([get_xy(s11), get_xy(s12), get_xy(s13)], [get_xy(
        s21), get_xy(s22), get_xy(s23)]).make_transform_function()

    mapped_stars = []
    for star in stars1:
        x, y, r, b = star
        mapped_stars.append(t((x, y)))

    return (mapped_stars, [s11, s12 , s13], [s21, s22 , s23])
