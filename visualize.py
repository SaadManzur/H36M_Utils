import numpy as np
from copy import copy
from constants import PARENTS
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.interpolation import rotate

class Visualize(object):

    def __init__(self, num_of_cameras):
        self.num_of_cameras = num_of_cameras
        self.cam_coords = np.zeros((self.num_of_cameras, 3))
        self.cam_rotations = np.zeros((self.num_of_cameras, 3, 3))

    def plot_3d(self, joints_3d, plot_cameras=False, plot_from_camera=None):
        coords = joints_3d.copy()

        fig = plt.figure(figsize=(16, 16))
        subplot = fig.add_subplot(111, projection='3d')

        if plot_from_camera is not None:
            view_matrix = self.get_view_matrix_of(plot_from_camera)
            coords = np.hstack((coords, np.ones((coords.shape[0], 1))))
            coords = np.matmul(view_matrix, coords.T).T
            self._plot_transformed_point(subplot, self.cam_coords[plot_from_camera, :], axes=self.cam_rotations[plot_from_camera, :, :], transformation_matrix=view_matrix)
        
        subplot.scatter(coords[:, 0], coords[:, 1], coords[:, 2])

        for i in range(coords.shape[0]):
            parent = PARENTS[i]

            if parent < 0:
                continue

            subplot.plot([coords[i, 0], coords[parent, 0]],
                        [coords[i, 1], coords[parent, 1]],
                        [coords[i, 2], coords[parent, 2]])

        if plot_cameras:
            self._plot_cameras(subplot)

        subplot.set_xlim(-3000, 3000)
        subplot.set_ylim(-3000, 3000)
        subplot.set_zlim(-3000, 3000)

        subplot.set_xlabel("X")
        subplot.set_ylabel("Y")
        subplot.set_zlabel("Z")
        plt.show()

    def plot_2d(self, joints_2d):
        fig = plt.figure(figsize=(16, 16))
        subplot = fig.add_subplot(111)
        
        subplot.scatter(joints_2d[:, 0], joints_2d[:, 1])
        subplot.set_aspect('equal')

        for i in range(joints_2d.shape[0]):
            parent = PARENTS[i]

            subplot.annotate(str(i), [joints_2d[i, 0], joints_2d[i, 1]])

            if parent < 0:
                continue

            subplot.plot([joints_2d[i, 0], joints_2d[parent, 0]],
                         [joints_2d[i, 1], joints_2d[parent, 1]])


        subplot.set_xlabel("X")
        subplot.set_ylabel("Y")
        plt.show()

    def place_camera_circular(self, height, radius, center):

        angular_interval = (2*np.pi) / self.num_of_cameras

        for i in range(self.num_of_cameras):
            self.cam_coords[i, :] = radius*np.cos(i * angular_interval) + center[0], height + center[1], radius*np.sin(i * angular_interval) + center[2]
            
            look_at = center - self.cam_coords[i, :]
            look_at = look_at / np.linalg.norm(look_at)
            up = [0, -1, 0]
            right = np.cross(up, look_at)
            up = np.cross(look_at, right)

            self.cam_rotations[i, :, 0] = right
            self.cam_rotations[i, :, 1] = up
            self.cam_rotations[i, :, 2] = look_at

    def place_random_cameras(self, cam_count, radius_range, center):
        assert cam_count <= self.num_of_cameras, "Camera count exceeds predefined camera count"

        radii = np.random.uniform(low=radius_range[0], high=radius_range[1], size=(cam_count, ))
        theta = np.random.uniform(low=-np.pi, high=0, size=(cam_count, ))
        phi = np.random.uniform(low=0, high=np.pi, size=(cam_count, ))

        for i in range(cam_count):
            self.place_camera_at(i, radii[i], theta[i], phi[i], center)

    def place_camera_at(self, index, radius, theta, phi, center):
            self.cam_coords[index, :] = radius*np.cos(theta)*np.sin(phi) + center[0], radius*np.sin(theta)*np.sin(phi) + center[1], radius*np.cos(phi) + center[2]

            look_at, up, right = self._get_camera_vectors(center - self.cam_coords[index, :])

            self.cam_rotations[index, :, 0] = right
            self.cam_rotations[index, :, 1] = up
            self.cam_rotations[index, :, 2] = look_at


    def _get_camera_vectors(self, look_at):
        look_at = look_at / np.linalg.norm(look_at)
        up = [0, -1, 0]
        right = np.cross(up, look_at)
        up = np.cross(look_at, right)

        return look_at, up, right
    

    def _plot_transformed_point(self, subplot, point, axes=None, transformation_matrix=None):
        if transformation_matrix is not None and axes is not None:
        
            ones = np.ones((4, 1))
            ones[:3, 0] = point
            ones = np.matmul(transformation_matrix, ones)

            zeros = np.zeros((4, 4))
            zeros[:3, :3] = axes
            zeros = np.matmul(transformation_matrix, zeros)

            zeros[:3, 0] /= np.linalg.norm(zeros[:3, 0])
            zeros[:3, 1] /= np.linalg.norm(zeros[:3, 1])
            zeros[:3, 2] /= np.linalg.norm(zeros[:3, 2])

            zeros[:3, 0] = np.cross(zeros[:3, 1], zeros[:3, 2])
            zeros[:3, 1] = np.cross(zeros[:3, 2], zeros[:3, 0])
            zeros[:3, 2] = np.cross(zeros[:3, 0], zeros[:3, 1])


        self._plot_point_with_axes(subplot, ones[:3, 0], zeros[:3, :3])

    def _plot_point_with_axes(self, subplot, point, axes):
        subplot.scatter(point[0], point[1], point[2])

        x = 400*axes[:, 0]
        y = 400*axes[:, 1]    
        z = 400*axes[:, 2]

        translated_x = x + point
        translated_y = y + point
        translated_z = z + point

        subplot.plot(
            [point[0], translated_x[0]],
            [point[1], translated_x[1]],
            [point[2], translated_x[2]],
            color='r'
        )

        subplot.plot(
            [point[0], translated_y[0]],
            [point[1], translated_y[1]],
            [point[2], translated_y[2]],
            color='g'
        )

        subplot.plot(
            [point[0], translated_z[0]],
            [point[1], translated_z[1]],
            [point[2], translated_z[2]],
            color='b'
        )


    def _plot_cameras(self, subplot):

        subplot.scatter(self.cam_coords[:, 0], self.cam_coords[:, 1], self.cam_coords[:, 2], color='k')

        for i in range(self.num_of_cameras):
            x = 400*self.cam_rotations[i, :, 0]
            y = 400*self.cam_rotations[i, :, 1]    
            z = 400*self.cam_rotations[i, :, 2]

            translated_x = x + self.cam_coords[i]
            translated_y = y + self.cam_coords[i]
            translated_z = z + self.cam_coords[i]

            subplot.plot(
                [self.cam_coords[i, 0], translated_x[0]],
                [self.cam_coords[i, 1], translated_x[1]],
                [self.cam_coords[i, 2], translated_x[2]],
                color='r'
            )

            subplot.plot(
                [self.cam_coords[i, 0], translated_y[0]],
                [self.cam_coords[i, 1], translated_y[1]],
                [self.cam_coords[i, 2], translated_y[2]],
                color='g'
            )

            subplot.plot(
                [self.cam_coords[i, 0], translated_z[0]],
                [self.cam_coords[i, 1], translated_z[1]],
                [self.cam_coords[i, 2], translated_z[2]],
                color='b'
            )

    def get_view_matrix_of(self, i):
        assert i < self.num_of_cameras, "Camera index should be less than camera count."

        transformation_matrix = np.zeros((4, 4))
        rotation_matrix = self.cam_rotations[i, :, :]

        translation = self.cam_coords[i, :].reshape((3, 1))

        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = - np.matmul(rotation_matrix, translation)[:, 0]
        transformation_matrix[3, 3] = 1

        return transformation_matrix


    def get_projection(self, joints_3d, plot_from_camera, f, c):
        coords = joints_3d.copy()

        view_matrix = self.get_view_matrix_of(plot_from_camera)
        coords = np.hstack((coords, np.ones((coords.shape[0], 1))))
        coords = np.matmul(view_matrix, coords.T).T

        coords_2d = np.zeros((coords.shape[0], 2))

        for i in range(coords.shape[0]):
            coords_2d[i, 0] = coords[i, 0] / coords[i, 2] * f[0] + c[0]
            coords_2d[i, 1] = coords[i, 1] / coords[i, 2] * f[1] + c[1]

        return coords_2d

    def get_camspace_coord(self, joints_3d, plot_from_camera):
        coords = joints_3d.copy()

        view_matrix = self.get_view_matrix_of(plot_from_camera)
        coords = np.hstack((coords, np.ones((coords.shape[0], 1))))
        coords = np.matmul(view_matrix, coords.T).T

        return coords[:, :3]

    def get_rotation_and_translation(self, cam_idx):
        return self.cam_rotations[cam_idx, :, :], self.cam_coords[cam_idx, :]