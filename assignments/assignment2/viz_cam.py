from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class CameraPoseVisualizer:
    def __init__(self, xlim=None, ylim=None, zlim=None):
        self.fig = plt.figure(figsize=(18, 7))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_box_aspect([1, 1, 1])
        self.poses = []


    def extrinsic2pyramid(self, c2w, focal_length_scaled = .1, w_scaled = .2, h_scaled = .2, color='r'):
        vertex_cam_coord = np.array([[0, 0, 0, 1], # camera center
                                    [w_scaled/2, h_scaled/2, focal_length_scaled, 1], # top right
                                    [w_scaled/2, -h_scaled/2, focal_length_scaled, 1], # bottom right
                                    [-w_scaled/2, -h_scaled/2, focal_length_scaled, 1], # bottom left
                                    [-w_scaled/2, h_scaled/2, focal_length_scaled, 1]]) # top left
        vertex_world_coord = np.dot(c2w, vertex_cam_coord.T).T
        
        # Build triangle meshes
        meshes = [[vertex_world_coord[0, :-1], vertex_world_coord[1, :-1], vertex_world_coord[2, :-1]],
                  [vertex_world_coord[0, :-1], vertex_world_coord[2, :-1], vertex_world_coord[3, :-1]],
                  [vertex_world_coord[0, :-1], vertex_world_coord[3, :-1], vertex_world_coord[4, :-1]],
                  [vertex_world_coord[0, :-1], vertex_world_coord[4, :-1], vertex_world_coord[1, :-1]],
                  [vertex_world_coord[1, :-1], vertex_world_coord[2, :-1], vertex_world_coord[3, :-1], vertex_world_coord[4, :-1]]]

        # Add meshes to the plot
        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))
        camera_position = c2w[:3, 3]
        self.poses.append(camera_position)



    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label='Frame Number')

    def show(self, xlim=None, ylim=None, zlim=None):
        alpha = 1.5
        # if xlim is None:
        #     xlim = (alpha*min([pose[0] for pose in self.poses]), alpha*max([pose[0] for pose in self.poses]))
        # self.ax.set_xlim(xlim)
        
        # if ylim is None:
        #     ylim = (alpha*min([pose[1] for pose in self.poses]), alpha*max([pose[1] for pose in self.poses]))
        # self.ax.set_ylim(ylim)
        
        # if zlim is None:
        #     zlim = (alpha*min([pose[2] for pose in self.poses]), alpha*max([pose[2] for pose in self.poses]))
        # self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        
        print('initialize camera pose visualizer')
        plt.title('Extrinsic Parameters')
        plt.show()


def read_transformations(file_path):
    transformations = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 3):
            name = lines[i].strip()
            R = [float(x) for x in lines[i + 1].split()]
            t = [float(x) for x in lines[i + 2].split()]
            R_matrix = np.array([R[0:3], R[3:6], R[6:9]])
            t_matrix = np.array(t)
            transformations[name] = {'R': R_matrix, 't': t_matrix}
    return transformations

directory = Path('/mnt/d/code/CSCI-599-3DCV/assets/assignment2/results')
folders = [f for f in directory.iterdir() if f.is_dir()]

for folder in folders:

    visa = CameraPoseVisualizer()
    file_path = folder / 'camera_poses.txt'
    transformations = read_transformations(file_path)
    # Create the root element of the XML

    # Iterate over the cameras and add them to the XML
    for cam_id, cam_params in transformations.items():
        # Create the camera element
        R, t = cam_params['R'], cam_params['t']
        w2c_matrix = np.concatenate((R, t.reshape(3, 1)), axis=1)
        w2c_matrix = np.vstack((w2c_matrix, np.array([0, 0, 0, 1])))
        # Add the c2w matrix to the XML
        c2w_matrix = np.linalg.inv(w2c_matrix)
        visa.extrinsic2pyramid(c2w_matrix, color='r')
    print(folder)
    visa.show()