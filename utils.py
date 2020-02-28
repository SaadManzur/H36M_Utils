"""
Utility module for conducting different operations on the dataset
"""
import untangle
import h5py
import numpy as np

def normalize_2d(data, w, h):
    return (2.0*data/w) - [1, (h*1.0)/w]

def parse_metadata(metadata_path):
    xml_data = untangle.parse(metadata_path)

    joint_names = [child.name.cdata for child in xml_data.a.skel_pos.tree.item]
    parents = [int(child.parent.cdata) for child in xml_data.a.skel_pos.tree.item]
    
    children = {}

    for i in range(len(parents)):
        if parents[i] not in children:
            children[parents[i]] = []
        
        children[parents[i]].append(i)
    
    right = []
    left = []

    for idx, name in enumerate(joint_names):
        if name.startswith("Left"):
            left.append(idx)
        elif name.startswith("Right"):
            right.append(idx)

    return {
        "names": joint_names,
        "parents": parents,
        "children": children,
        "joints_left": left,
        "joints_right": right
    }

def read_h5(file_path):
    data = h5py.File(file_path, 'r')

    return data

def read_npz(file_path):
    data = np.load(file_path, allow_pickle=True)['data'].item()

    return data
