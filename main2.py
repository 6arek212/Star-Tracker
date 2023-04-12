from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
from star_finder import detect_stars
from pillow_heif import register_heif_opener
import img_compare as cp
register_heif_opener()


image1_arr = np.asarray(Image.open(os.path.join('./imgs', '1.JPG')
                                   ).convert("L").resize((300, 300), Image.LANCZOS))

image2_arr = np.asarray(Image.open(os.path.join('./imgs', '2.JPG')
                                   ).convert("L").resize((300, 300), Image.LANCZOS))

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()


def draw_circle(xy: tuple, radius, ax, color='r'):
    x, y = xy
    circle = patches.Circle((x, y), radius,
                            linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(circle)


def draw_stars(stars, ax, color='r'):
    for star in stars:
        x = star[0]
        y = star[1]
        draw_circle((x, y), 4, ax, color)


s1 = detect_stars(image1_arr, 0.1, 3)
s2 = detect_stars(image2_arr, 0.06, 3)


draw_stars(s1, ax1)


map_stars, v1, v2 = cp.map_stars(s1, s2)
draw_stars(map_stars, ax2)

draw_stars(v1, ax1, color='g')
draw_stars(v2, ax2, color='g')
print(map_stars)
print(v1)
print(v2)

plt.tight_layout()
ax1.imshow(image1_arr, cmap='gray', vmin=0, vmax=255)
ax2.imshow(image2_arr, cmap='gray', vmin=0, vmax=255)
ax2.set_axis_off()
ax1.set_axis_off()
plt.show()
