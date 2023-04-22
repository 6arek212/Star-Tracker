

# project the images side by side with matching lines
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch


def show_data(source_points, dest_points, points_on_line_1, points_on_line_2,  mapped_stars, img1, img2):

    for p in points_on_line_1:
        img1 = cv2.circle(img1, (p[0], p[1]), p[2], (255, 0, 255), 1, 0)

    for p in points_on_line_2:
        img2 = cv2.circle(img2, (p[0], p[1]), p[2], (255, 0, 255), 1, 0)

    for i, p in enumerate(source_points):
        img1 = cv2.circle(img1, (int(p[0]), int(p[1])), 10, (0, 255, 0), 0)
        img1 = cv2.putText(img1, str(i), (int(p[0] + 5), int(p[1]-5)), cv2.FONT_HERSHEY_SIMPLEX,
                           .5, (0, 255, 255), 1, cv2.LINE_AA)

    for i, p in enumerate(dest_points):
        img2 = cv2.circle(img2, (int(p[0]), int(p[1])), p[2]-2, (0, 255, 0), 0)
        img2 = cv2.putText(img2, str(i), (int(p[0] + 5), int(p[1]-5)), cv2.FONT_HERSHEY_SIMPLEX,
                           .5, (0, 255, 255), 1, cv2.LINE_AA)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    for i, p in enumerate(mapped_stars):
        con = ConnectionPatch(xyA=p[0], xyB=p[1], coordsA="data", coordsB="data",
                              axesA=ax1, axesB=ax2, color="red")
        ax2.add_artist(con)

    ax1.imshow(img1)
    ax2.imshow(img2)
    plt.show()
