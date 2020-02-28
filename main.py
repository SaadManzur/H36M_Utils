from __future__ import print_function

from joint_set import JointSet
import constants as jnt
from utils import parse_metadata, read_h5, read_npz
from conversion import convert_json_to_npz, convert_h5_directory_to_augmented, convert_h5_to_projected
from visualize import Visualize
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":

    viz = Visualize(50)

    data = read_h5("annot.h5")

    viz.place_random_cameras(50, [3000, 3500], data['pose/3d-univ'][()][0, 0, :])

    data_2d = np.zeros((0, 32, 2))
    data_3d = np.zeros((0, 32, 3))

    point_2d = viz.get_projection(data['pose/3d-univ'][()][0, :, :], 32, jnt.CAMERAS[0]['focal_length'], jnt.CAMERAS[0]['center'])

    #viz.plot_3d(data['pose/3d-univ'][()][0, :, :], True)

    #viz.plot_2d(point_2d)

    #convert_h5_directory_to_augmented("../H36M_H5_Annotations/**/**/*.h5", 15)

    convert_h5_to_projected('annot.h5', 'h36m_s1_reprojected')
