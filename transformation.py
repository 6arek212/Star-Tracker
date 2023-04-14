
import numpy as np
import cv2


class Transformation2D:

    def __init__(self, src_points, dst_points):
        self.src_points = np.float32(src_points)
        self.dst_points = np.float32(dst_points)
        self.M = cv2.getAffineTransform(self.src_points, self.dst_points)

    def calculate_mapped_vec(self, vec):
        x, y = vec
        point = np.array([[x], [y], [1]], dtype=np.float32)
        transformed_point = np.dot(self.M, point)
        return transformed_point[0][0], transformed_point[1][0]

    def make_transform_function(self):
        return lambda vec: self.calculate_mapped_vec(vec)

