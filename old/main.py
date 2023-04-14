from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
from star_finder import detect_stars
from pillow_heif import register_heif_opener


register_heif_opener()

image1_arr = np.asarray(Image.open(os.path.join('./imgs', '1_1.HEIC')
                   ).convert("L").resize((300, 300), Image.LANCZOS))


fig1, ax1 = plt.subplots()


# detect starts
# pick two initial stars
# find matching initial stars
# make transformation function
# transform the starts coordinates to the other image


def draw_circle(xy: tuple, radius, ax):
    x, y = xy
    circle = patches.Circle((x, y), radius,
                            linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(circle)


circles = detect_stars(image1_arr, intensity=0.8)
for circle in circles:
    x, y, radius, b = circle
    draw_circle((x, y), radius, ax1)

print(f"found {len(circles)} starts in image 1")


plt.tight_layout()
ax1.imshow(image1_arr, cmap='gray', vmin=0, vmax=255)
ax1.set_axis_off()
plt.show()
