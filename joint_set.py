from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt

import constants

class JointSet(object):

    def __init__(self, debug=False):
        self._joints_left = []
        self._joints_right = []
        self._joints = []
        self._parents = []
        self._debug = debug
        self._adjacency = constants.ADJACENCY_LIST
        self._all_parents = np.zeros(len(constants.JOINTS_NAME), dtype=np.int8)
        self._all_parents -= 1
        self.compute_parents()
        self._selected_adjacency = {}

    def compute_parents(self):
        """
        Compute parents for the dataset
        """
        for parent in constants.ADJACENCY_LIST:
            for child in constants.ADJACENCY_LIST[parent]:
                self._all_parents[child] = parent

    def get_parent(self, index, joints_choice):
        """
        Get parent recursively for a selected indices
        """
        if constants.PARENTS[index] < 0:
            return -1

        print("Resolving parent for ", constants.JOINTS_NAME[index])
        while(constants.PARENTS[index] not in joints_choice):
            #print("Checking ", constants.JOINTS_NAME[all_parents[index]])
            index = constants.PARENTS[index]

        return constants.PARENTS[index]

    def build(self, joints_choice=list(range(34))):
        """
        Build skeleton information from joint set
        
        @param joints_choice: Takes the choice list indices
        """

        self._joints = joints_choice

        for i in range(len(joints_choice)):
            if joints_choice[i] in constants.LEFT_JOINTS:
                self._joints_left.append(joints_choice[i])

            elif joints_choice[i] in constants.RIGHT_JOINTS:
                self._joints_right.append(joints_choice[i])

            current_parent = self.get_parent(joints_choice[i], joints_choice)
            self._parents.append(current_parent)
            
            if current_parent < 0:
                continue

            if current_parent in self._selected_adjacency:
                self._selected_adjacency[current_parent].append(joints_choice[i])
            else:
                self._selected_adjacency[current_parent] = []
                self._selected_adjacency[current_parent].append(joints_choice[i])

    def get_skeleton(self):
        """
        Get skeleton information after building a joint set.
        """
        joints_to_select = self._joints.copy()
        joints_left = [self._joints.index(i) for i in self._joints_left]
        joints_right = [self._joints.index(i) for i in self._joints_right]
        joints_parents = [self._joints.index(i) if i > 0 else -1 for i in self._parents]
        joint_names = [constants.JOINTS_NAME[i] for i in joints_to_select]

        return {
            "names": joint_names,
            "parents": joints_parents,
            "joints": self._joints,
            "left": joints_left,
            "right": joints_right,
            "adjacency": self._selected_adjacency
        }

    def draw_skeleton(self):
        """
        Draws the skeleton after setting the initial joint set
        """
        print("Skeleton Information")
        print("Total joints selected: ", len(self._joints))
        print("Names: ", [constants.JOINTS_NAME[i] for i in self._joints])
        print("Left: ", [constants.JOINTS_NAME[i] for i in self._joints_left])
        print("Right: ", [constants.JOINTS_NAME[i] for i in self._joints_right])
        print("Tree: [parent: children]")

        for parent in self._selected_adjacency:
            print("\t", constants.JOINTS_NAME[parent], end=": ")
            for child in self._selected_adjacency[parent]:
                print(constants.JOINTS_NAME[child], end=" ")

            print()

        fig = plt.figure(figsize=(16, 16))
        subplot = fig.add_subplot(111)
        for idx, item in enumerate(self._joints):
            subplot.scatter(
                constants.DUMMY_LOCATIONS[item][0],
                constants.DUMMY_LOCATIONS[item][1])

            annotation_offset_x = 0.1
            annotation_offset_y = 0.2
            if np.sign(constants.DUMMY_LOCATIONS[item][0]) < 0:
                annotation_offset_x = -0.9

            subplot.annotate(
                constants.JOINTS_NAME[item] + " ("+ str(idx) + ") ",
                [constants.DUMMY_LOCATIONS[item][0] + annotation_offset_x,
                 constants.DUMMY_LOCATIONS[item][1] + annotation_offset_y]
            )

        for idx, item in enumerate(self._joints):
            parent = self._parents[idx]
            
            if parent < 0:
                continue

            subplot.plot(
                [constants.DUMMY_LOCATIONS[item][0],
                 constants.DUMMY_LOCATIONS[parent][0]],
                [constants.DUMMY_LOCATIONS[item][1],
                 constants.DUMMY_LOCATIONS[parent][1]],
                color='k'
            )

        subplot.set_ylim(-6, 6)
        subplot.set_xlim(-6, 6)
        plt.show()
            