'''
this is a manual mapping test
'''

import math
from matplotlib.patches import ConnectionPatch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from ransac_line_fit import ransac_line_fit



path1 = './imgs/fr1.jpg'
path2 = './imgs/fr2.jpg'
size = (600, 600)

img1 = np.array(Image.open(path1).resize(size))
img2 = np.array(Image.open(path2).resize(size))

img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


def __detect_stars(img: np.ndarray):
    '''detect all stars and draw a red circle'''
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 2, 20,
                               param1=250, param2=1, minRadius=2, maxRadius=6)
    return circles


# main runing function
def get_stars(image: np.ndarray, size: tuple):
    circles = __detect_stars(image)
    circles = np.uint16(np.around(circles))
    circles_cordinates = []
    for i in circles[0, :]:
        if i[0] < size[0] and i[1] < size[1]:
            circles_cordinates.append(
                (int(i[0]), int(i[1]), i[2] + 5,  int(image[i[0], i[1]])))
    return circles_cordinates


stars1 = get_stars(img1_gray, size)
stars2 = get_stars(img2_gray, size)

line1, points_on_line_1 = ransac_line_fit(stars1)
line2, points_on_line_2 = ransac_line_fit(stars2)


fig, (ax1, ax2) = plt.subplots(1, 2)


# [[570, 474], [537, 436], [508, 360], [388, 357]   , [407  , 86] , [141 , 530] , [338 , 566]]
# [[563, 480], [483, 450], [441, 340], [326, 330]  , [378 , 56] , [63 , 488] , [258,528]]


src = [[570, 474], [537, 436], [538, 257]]
dest = [[483, 450], [457, 414], [479, 247]]
pts_src = np.array(np.float32([(p[0], p[1]) for p in src]))
pts_dst = np.array(np.float32([(p[0], p[1]) for p in dest]))

# h, status = cv2.findHomography(pts_src, pts_dst)
# M = cv2.estimateAffinePartial2D(np.float32(pts_src),np.float32( pts_dst))

M = cv2.getAffineTransform(pts_src, pts_dst)


def mapper(point):
    transformed_pt = np.dot(M, np.array([point[0], point[1], 1]))
    return (int(transformed_pt[0]), int(transformed_pt[1]))


# def mapper(vec):
#     point = np.array([[vec[0]], [vec[1]], [1]], dtype=np.float32)
#     transformed_point = np.dot(M, point)
#     print(point , transformed_point)
#     print('---')
#     return transformed_point[0][0], transformed_point[1][0]

# print(h)


def find_point(stars, s_point, mapped_point, e):
    x2, y2, r2, b2 = mapped_point

    for star in stars:
        x1, y1, r1, b1 = star
        # point = np.array([x, y])
        # dist = np.linalg.norm(point-mapped_point)

        dist_from_source = math.sqrt(
            (s_point[0]-x2) ** 2 + (s_point[1]-y2) ** 2)
        dist_s = math.sqrt((x1-x2) ** 2 + (y1-y2) ** 2)

        print(dist_from_source)
        if dist_s < e and abs(r1 - r2) < 4 and dist_from_source < 100:
            return True
    return False


def count_intermidate_points(stars1, stars2, t, e=10):
    cnt = 0
    for star in stars1:
        x, y, r, b = star
        if (find_point(stars2, star, (*t((x, y)), r, b), e)):
            cnt += 1
    return cnt


print(count_intermidate_points(stars1, stars2, mapper))
# current = 0
# best_t = None
# best_h = None

# for i in range(0, 10000):

#     src = random.sample(stars1, 5)
#     dest = random.sample(stars2, 5)
#     pts_src = np.array([(p[0], p[1]) for p in src])
#     pts_dst = np.array([(p[0], p[1]) for p in dest])
#     h, status = cv2.findHomography(
#         pts_src, pts_dst, cv2.LMEDS, confidence=.8, maxIters=100000)
#     if h is None:
#         continue

#     def t(p):
#         x, y, z = np.matmul(h, np.array([p[0], p[1], 1]))
#         return (x/z, y/z)

#     cnt = count_intermidate_points(stars1, stars2, t)

#     if (cnt > current):
#         print(h)
#         current = cnt
#         best_t = t
#         best_h = h
# print(current)

# draw img1 stars
for p in points_on_line_1:
    x, y, r, b = p
    img1 = cv2.circle(img1, (int(p[0]), int(p[1])), p[2], (255, 255, 255), 0)

# draw img2 stars
for p in points_on_line_2:
    img2 = cv2.circle(img2, (int(p[0]), int(p[1])), p[2], (255, 255, 255), 0)


for i, p in enumerate(pts_src):
    img1 = cv2.circle(img1, (int(p[0]), int(p[1])), 10, (0, 255, 0), 0)
    img1 = cv2.putText(img1, str(i), (int(p[0] + 5), int(p[1]-5)), cv2.FONT_HERSHEY_SIMPLEX,
                       .5, (0, 255, 255), 1, cv2.LINE_AA)

for i, p in enumerate(pts_dst):
    img2 = cv2.circle(img2, (int(p[0]), int(p[1])), 10, (0, 255, 0), 0)
    img2 = cv2.putText(img2, str(i), (int(p[0] + 5), int(p[1]-5)), cv2.FONT_HERSHEY_SIMPLEX,
                       .5, (0, 255, 255), 1, cv2.LINE_AA)


# draw mapped points
# for p in points_on_line_1:
#     x, y = mapper(p)
#     # x, y, z = np.matmul(h, np.array([p[0], p[1], 1]))
#     img2 = cv2.circle(img2, (int(x), int(y)),
#                       10, (255, 255, 255), 0)

# p = [761,  704, 1]
# print(np.matmul(h, p))
# cv2.imshow('detected circles 1', img1)
# cv2.drawMatches(img1, points_on_line_1, img2, [
#                 mapper(p) for p in points_on_line_1] , )

mapped_points = [mapper(p) for p in points_on_line_1]


for i, p in enumerate(points_on_line_1):
    con = ConnectionPatch(xyA=(p[0], p[1]), xyB=(mapped_points[i][0], mapped_points[i][1]), coordsA="data", coordsB="data",
                          axesA=ax1, axesB=ax2, color="red")
    ax2.add_artist(con)
    # ax1.plot(p[0], p[1], 'ro', markersize=10)
    # ax2.plot(mapped_points[i][0], mapped_points[i][1], 'ro', markersize=10)


ax1.imshow(img1)
ax2.imshow(img2)
plt.show()
