
import numpy as np


class Transformation2D:

    def __init__(self, src_points, dst_points):
        self.src_points = np.array(src_points)
        self.dst_points = np.array(dst_points)

        delta_src = self.src_points[1] - self.src_points[0]
        delta_dst = self.dst_points[1] - self.dst_points[0]

        theta = np.arctan2(delta_dst[1], delta_dst[0]) - \
            np.arctan2(delta_src[1], delta_src[0])
        
        scale = np.linalg.norm(delta_dst) / np.linalg.norm(delta_src)

        R = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
        S = np.array([[scale, 0], [0, scale]])
        T = self.dst_points[0] - np.dot(S, np.dot(R, self.src_points[0]))

        self.M = np.vstack((np.column_stack((S, T)), [0, 0, 1]))



    def calculate_mapped_vec(self, vec):
        x, y = vec
        point = np.array([x, y, 1])
        transformed_point = np.dot(self.M, point)
        return (transformed_point[0], transformed_point[1])
    
      
    def make_transform_function(self):
        return lambda vec: self.calculate_mapped_vec(vec)


