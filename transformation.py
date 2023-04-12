
import numpy as np
import cv2


class Transformation2D:

    def __init__(self, base_vector1, base_vector2, tranform_vector1, tranform_vector2):
        self.base_vector1 = np.array(base_vector1)
        self.base_vector2 = np.array(base_vector2)
        self.tranform_vector1 = np.array(tranform_vector1)
        self.tranform_vector2 = np.array(tranform_vector2)
        self.mat = np.array(
            [self.base_vector1, self.base_vector2]).transpose()
        self.scale_factor = self.get_scale_factor()

    def find_alphas(self, vec):
        return np.linalg.solve(self.mat, vec)

    def get_scale_factor(self):
        # Calculate the distance and angle between the two points in each image
        src_distance = np.sqrt(
            np.sum((self.base_vector2 - self.base_vector1)**2))
        src_angle = np.arctan2(
            self.base_vector2[1] - self.base_vector1[1], self.base_vector2[0] - self.base_vector1[0])

        dst_distance = np.sqrt(
            np.sum((self.tranform_vector2 - self.tranform_vector1)**2))
        dst_angle = np.arctan2(
            self.tranform_vector2[1] - self.tranform_vector1[1], self.tranform_vector2[0] - self.tranform_vector1[0])

        # Calculate the scaling factor
        scaling_factor = dst_distance / src_distance
        return scaling_factor

    def calculate_mapped_vec(self, vec):
        vec_py = np.array(vec)
        a1, a2 = self.find_alphas(vec_py)
        mapped = np.dot(a1, self.tranform_vector1) + \
            np.dot(a2, self.tranform_vector2)
        return mapped

    def make_transform_function(self):
        return lambda vec: self.calculate_mapped_vec(vec)


# def get_transform_function(src_points, dst_points):
#     """
#     Calculates the transformation function that maps points from img1 to img2
#     based on the corresponding src_points and dst_points.
#     """
#     src_points = np.vstack((src_points, [1, 1]))
#     dst_points = np.vstack((dst_points, [1, 1]))

#     # Calculate the transformation matrix
#     src_matrix = np.array(src_points, np.float32)
#     dst_matrix = np.array(dst_points, np.float32)


#     print(src_matrix)
#     print(dst_matrix)
#     transform_matrix = cv2.getAffineTransform(src_matrix, dst_matrix)

#     # Define the transformation function
#     def transform_function(x, y):
#         """
#         Maps a point (x, y) from img1 to img2 based on the transformation matrix.
#         """
#         point = np.array([[x, y, 1]], dtype=np.float32).T
#         transformed_point = np.dot(transform_matrix, point)
#         return transformed_point[0, 0], transformed_point[1, 0]

#     return transform_function


# t = Transformation2D((1, 0), (1, 1), (1, 2), (1, 4))
# trans = t.make_transform_function()

# x = trans((3,4))

# print(x)
