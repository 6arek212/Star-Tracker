import numpy as np
import cv2
from PIL import Image
from pillow_heif import register_heif_opener


# main runing function
def get_stars(image: np.ndarray, size: tuple, threshold=127):
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 2, 20,
                               param1=250, param2=1, minRadius=2, maxRadius=6)

    # no need for more that 20 star, try to reduce the number
    desired_size = 20
    length = len(circles[0, :])
    if length > desired_size:
        for t in range(1, 6):
            circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 2, 20,
                                       param1=250, param2=t, minRadius=2, maxRadius=6)
            length = len(circles[0, :])
            if length < desired_size:
                break

    if circles is None:
        return []
    circles = np.uint16(np.around(circles))
    circles_cordinates = []
    for i in circles[0, :]:
        if i[0] < size[0] and i[1] < size[1]:
            circles_cordinates.append(
                (int(i[0]), int(i[1]), i[2] + 5,  int(image[i[0], i[1]])))

    return circles_cordinates


def save_stars_coordinates(output_path , stars):
    with open(output_path, "w") as f:
        f.write(f"x y r b\n")
        for star in stars:
            f.write(f"{star[0]} , {star[1]} , {star[2]} , {star[3]}\n")




# run main function
if __name__ == '__main__':
    register_heif_opener()

    path1 = './imgs/fr1.jpg'
    # path1 = './imgs/ST_db1.png'
    path2 = './imgs/ST_db2.png'
    # path1 = './imgs/1.jpg'
    # path2 = './imgs/2.jpg'
    size = (600, 600)

    img1 = np.array(Image.open(path1).resize(size))
    img2 = np.array(Image.open(path2).resize(size))

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    stars1 = get_stars(img1_gray, size)
    stars2 = get_stars(img2_gray, size)

    # draw the outer circle
    for i in stars1:
        img1 = cv2.circle(img1, (i[0], i[1]), i[2], (255, 0, 255), 1, 0)

    for i in stars2:
        img2 = cv2.circle(img2, (i[0], i[1]), i[2], (255, 0, 255), 1, 0)

    cv2.imshow('detected circles 1', img1)
    cv2.imshow('detected circles 2', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
