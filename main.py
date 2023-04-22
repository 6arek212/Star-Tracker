from display_imgs import show_data
import star_finder as finder
import img_compare as compare
import math
import random
from matplotlib.patches import ConnectionPatch
import numpy as np
import cv2
from PIL import Image
from pillow_heif import register_heif_opener
from ransac_line_fit import ransac_line_fit
import matplotlib.pyplot as plt


# support for HEIC images
register_heif_opener()


path1 = './imgs/fr1.jpg'
path2 = './imgs/fr2.jpg'
size = (600, 600)

# regular images
img1 = np.array(Image.open(path1).resize(size))
img2 = np.array(Image.open(path2).resize(size))

# gray images
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# get star coordinates
stars1 = finder.get_stars(img1_gray, size)
stars2 = finder.get_stars(img2_gray, size)


# write the coordinates to a file
finder.save_stars_coordinates('./fr1_results.txt', stars1)
finder.save_stars_coordinates('./fr2_results.txt', stars2)


# compare stars
mapped_stars, source_points, dest_points, line1, points_on_line_1, line2, points_on_line_2, matching_ratio = compare.map_stars(
    stars1, stars2)
compare.save_mapped_stars('./fr_mappings.txt', mapped_stars)

print('matching ratio:', matching_ratio)


show_data(source_points, dest_points,
          points_on_line_1, points_on_line_2,  mapped_stars, img1, img2)


