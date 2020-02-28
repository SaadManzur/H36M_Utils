"""
Convert the data to different formats. Structure:
{
    Width
    Height
    Depth [ignore if 2d data]
    Cameras: [
        Camera {
            Id
            Center
            Focus
            Translation
            Rotation
        }
    ]
    Train: {
        3d: [
            [x, y, z]
        ],
        2d: [
            [x, y]
        ],
        2d_c: [
           [x, y] 
        ]
    },
    Test: {
        3d: [
            [x, y, z]
        ],
        2d: [
            [x, y]
        ],
        2d_c: [
            [x, y]
        ]
    } 
}
"""
from __future__ import print_function

import numpy as np
from tqdm import tqdm
import ijson
from visualize import Visualize
import constants as jnt
from utils import parse_metadata, read_h5, read_npz
from semgcn.camera import world_to_camera, normalize_screen_coordinates, project_to_2d, image_coordinates
from semgcn.utils import wrap
import glob
import copy

TOTAL_DATAPOINTS = 3640788

def read_data(filename, field, shape, data_points=TOTAL_DATAPOINTS):

    data = np.zeros(shape, dtype=np.float32)

    with open(filename, 'r') as file:

        print("Gathering ", field)
        json_data = ijson.items(file, field + ".item", shape)

        print("Compiling ", field)
        for entry in tqdm(json_data, total=data_points):
            joint = np.array(entry, dtype=np.float32).reshape((1, shape[1], shape[2]))
            data = np.vstack((data, joint))

        file.close()

    return data

def convert_json_to_npz(dataset_path, filename):

    data_2d_train = read_data(dataset_path, "h36m_train_2d_origin", (0, 17, 2), 311951)
    data_2d_c_train = read_data(dataset_path, "h36m_train_2d_crop", (0, 17, 2), 311951)
    data_3d_train = read_data(dataset_path, "h36m_train_3d", (0, 17, 3), 311951)
    
    data_2d_test = read_data(dataset_path, "h36m_test_2d_origin", (0, 17, 2), 109764)
    data_2d_c_test = read_data(dataset_path, "h36m_test_2d_crop", (0, 17, 2), 109764)
    data_3d_test = read_data(dataset_path, "h36m_test_3d", (0, 17, 3), 109764)

    data = {
        "width": 1000,
        "height": 1002,
        "width_cropped": 256,
        "height_cropped": 256,
        "train": {
            "2d": data_2d_train,
            "2d_c": data_2d_c_train,
            "3d": data_3d_train
        },
        "test": {
            "2d": data_2d_test,
            "2d_c": data_2d_c_test,
            "3d": data_3d_test
        }
    }

    np.savez_compressed(filename, data=data)

def convert_world_to_cam(data_3d, center, focus):

    data_2d = np.zeros((data_3d.shape[0], data_3d.shape[1], 2))

    for i in tqdm(range(data_3d.shape[0])):
        viz = Visualize(1)
        viz.place_camera_circular(0, 2000, data_3d[0, 0, :])

        data_2d[i, :, :] = viz.get_projection(data_3d[i, :, :], 0, focus, center)

    return data_2d

def convert_h5_directory_to_augmented(directory_path, cam_count):
    
    files = glob.glob(directory_path)

    data = {}

    viz = Visualize(cam_count)

    prev_subject = ""

    for file in files:

        meta = file.split("/")
        subject = meta[2]

        """
        if subject != prev_subject:
            if data is not None:
                np.savez_compressed("output/h36m_augment_" + subject + "_cam_count_" + str(cam_count), data=data, cameras=jnt.CAMERAS)
                print("Dumped data for", subject)
            data = {}
            prev_subject = subject
        """

        action_name = meta[3].split("-")[0]
        
        if subject not in data:
            data[subject] = {}
        
        if action_name not in data[subject]:
            data[subject][action_name] = {'2d': np.empty((0, 32, 2), dtype=np.float32), '3d': np.empty((0, 32, 3), dtype=np.float32)}

        h5_data = read_h5(file)

        print("Processing", subject, "for", action_name)

        if action_name != "SittingDown":
            continue
        
        for i in tqdm(range(h5_data['pose/3d'][()].shape[0])):
            viz.place_random_cameras(cam_count, [3000, 3500], h5_data['pose/3d'][i, 0, :])
            for j in range(4):
                for k in range(cam_count):
                    point_2d = viz.get_projection(h5_data['pose/3d-univ'][()][i, :, :], k, jnt.CAMERAS[j]['focal_length'], jnt.CAMERAS[j]['center']).reshape(1, 32, 2)
                    point_3d = viz.get_camspace_coord(h5_data['pose/3d-univ'][()][i, :, :], k).reshape(1, 32, 3)
                    R, t = viz.get_rotation_and_translation(k)

                    data[subject][action_name]['2d'] = np.vstack((data[subject][action_name]['2d'], point_2d))
                    data[subject][action_name]['3d'] = np.vstack((data[subject][action_name]['3d'], point_3d))
                    data[subject][action_name]['cam_id'] = jnt.CAMERAS[j]['id']
                    data[subject][action_name]['R'] = R
                    data[subject][action_name]['t'] = t

    np.savez_compressed("output/h36m_augment_sitting_down_cam_count"+str(cam_count), data=data, cameras=jnt.CAMERAS)


def convert_h5_to_projected(dataset_path, save_filename):

    data = read_h5(dataset_path)

    data_ = {
        'train': {
            '2d': np.zeros((data['pose/3d-univ'][()].shape[0], 32, 2), dtype=np.float32),
            '3d': np.zeros((data['pose/3d-univ'][()].shape[0], 32, 3), dtype=np.float32),
            'w': np.zeros((data['pose/3d-univ'][()].shape[0]), dtype=np.float32),
            'h': np.zeros((data['pose/3d-univ'][()].shape[0]), dtype=np.float32)
        }
    }

    for i in tqdm(range(data['pose/3d'][()].shape[0])):
        camera_id = data['camera'][()][i]
        camera_index = [k for k in range(len(jnt.CAMERAS)) if jnt.CAMERAS[k]['id'] == str(camera_id)][0]

        camera = copy.deepcopy([cam for cam in jnt.CAMERAS if cam['id'] == str(camera_id)][0])
        camera['translation'] = np.array(jnt.EXTRINSIC_PARAMS['S1'][camera_index]['translation'])/1000
        camera['orientation'] = np.array(jnt.EXTRINSIC_PARAMS['S1'][camera_index]['orientation'])
        camera['center'] = normalize_screen_coordinates(np.array(camera['center']), w=camera['res_w'], h=camera['res_h'])
        camera['focal_length'] = np.array(camera['focal_length']) / camera['res_w'] * 2.0
        camera['intrinsic'] = np.concatenate((camera['focal_length'], camera['center'], camera['radial_distortion'], camera['tangential_distortion']))
    
        pos_3d_world = data['pose/3d'][()][i, :, :]/1000
        pos_3d_cam = data['pose/3d-univ'][()][i, :, :]/1000 #world_to_camera(pos_3d_world, R=camera['orientation'], t=camera['translation'])
        pos_2d = wrap(project_to_2d, True, pos_3d_cam, camera['intrinsic'])
        pos_2d_pixel_space = image_coordinates(pos_2d, w=camera['res_w'], h=camera['res_h'])

        data_['train']['2d'][i, :, :] = pos_2d_pixel_space[:, :]
        data_['train']['3d'][i, :, :] = pos_3d_cam[:, :]
        data_['train']['w'][i] = camera['res_w']
        data_['train']['h'][i] = camera['res_h']

    np.savez_compressed(save_filename, data=data_)