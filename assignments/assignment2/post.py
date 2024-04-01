from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors
import plyfile

def remove_outliers(point_cloud, k=10, threshold_factor=3):
    """
    Remove outliers from a point cloud using k-nearest neighbors.

    Parameters:
    - point_cloud: numpy array of shape (N, 6), representing the point cloud with XYZRGB format.
    - k: number of nearest neighbors to consider for each point.
    - threshold_factor: factor to determine the threshold for outlier removal. Points with a mean distance to their k nearest neighbors greater than (threshold_factor * median distance) will be considered outliers.

    Returns:
    - filtered_point_cloud: numpy array of shape (M, 6), representing the point cloud with outliers removed.
    """
    # Compute the k-nearest neighbors for each point based on XYZ coordinates only
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(point_cloud[:, :3])
    distances, _ = nbrs.kneighbors(point_cloud[:, :3])

    # Compute the mean distance to the k nearest neighbors for each point
    mean_distances = np.mean(distances[:, 1:], axis=1)

    # Determine the threshold for outlier removal
    threshold = threshold_factor * np.median(mean_distances)

    # Filter out points with a mean distance greater than the threshold
    filtered_point_cloud = point_cloud[mean_distances <= threshold]
    return filtered_point_cloud

def read_ply(filename):
    """
    Read a PLY file and return the point cloud as a numpy array.
    """
    with open(filename, 'rb') as f:
        plydata = plyfile.PlyData.read(f)
        points = np.array([[vertex['x'], vertex['y'], vertex['z'], vertex['red'], vertex['green'], vertex['blue']] for vertex in plydata['vertex']])
    return points

def write_ply(filename, points):
    """
    Write a point cloud to a PLY file.
    """
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    # Convert the points to a structured array with the correct dtype
    vertex = np.array([tuple(point) for point in points], dtype=dtype)
    
    # Describe the vertex element and write to file
    el = plyfile.PlyElement.describe(vertex, 'vertex')
    plyfile.PlyData([el]).write(filename)


directory = Path('/mnt/d/code/CSCI-599-3DCV/assets/assignment2/results')
folders = [f for f in directory.iterdir() if f.is_dir()]
for folder in folders:
    print(f'Processing {folder}')
    name = folder.stem
    ply_files = list(folder.glob('point-clouds/*.ply'))
    ply_files.sort(key=lambda f: int(f.stem.split("_")[1]), reverse=True)
    ply_raw = ply_files[0]
    points = read_ply(ply_raw)
    print(f'Original point count: {len(points)}')
    filtered_points = remove_outliers(points)
    print(f'Filtered point count: {len(filtered_points)}')
    write_ply(folder / f'{name}.ply', filtered_points)
    