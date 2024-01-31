import trimesh
def split_edges(mesh):
    """
    Splits each edge of the mesh into two halves by adding a new vertex at the midpoint of each edge.

    :param mesh: The input mesh.
    :return: A tuple containing the updated mesh with new vertices and a mapping of old to new edge indices.
    """
    # Implementation goes here
    return updated_mesh, edge_mapping

def update_vertex_positions(mesh, edge_mapping):
    """
    Updates the positions of the original vertices based on local averaging with neighboring vertex positions.

    :param mesh: The mesh with newly inserted vertices from split_edges.
    :param edge_mapping: A mapping of old to new edge indices, as returned by split_edges.
    :return: The mesh with updated vertex positions.
    """
    # Implementation goes here
    return updated_mesh
def split_faces(mesh, edge_mapping):
    """
    Updates the face indices to reflect the new vertices and divides each original triangle into four new triangles.

    :param mesh: The mesh with updated vertices from update_vertex_positions.
    :param edge_mapping: A mapping of old to new edge indices, as returned by split_edges.
    :return: The mesh with updated faces.
    """
    # Implementation goes here
    return updated_mesh


def subdivision_loop(mesh, iterations=1):
    """
    Apply Loop subdivision to the input mesh for the specified number of iterations.
    :param mesh: input mesh
    :param iterations: number of iterations
    :return: mesh after subdivision
    """
    for _ in range(iterations):
        edge_mapping = split_edges(mesh)
        mesh = update_vertex_positions(mesh, edge_mapping)
        mesh = split_faces(mesh, edge_mapping)
        return mesh

def simplify_quadric_error(mesh, face_count=1):
    """
    Apply quadratic error mesh decimation to the input mesh until the target face count is reached.
    :param mesh: input mesh
    :param face_count: number of faces desired in the resulting mesh.
    :return: mesh after decimation
    """
    return mesh

if __name__ == '__main__':
    # Load mesh and print information
    mesh = trimesh.load_mesh('assets/cube.obj')
    print(f'Mesh Info: {mesh}')
    
    # apply loop subdivision over the loaded mesh
    mesh_subdivided = mesh.subdivide_loop(iterations=1)
    
    # TODO: implement your own loop subdivision here
    # mesh_subdivided = subdivision_loop(mesh, iterations=1)
    
    # print the new mesh information and save the mesh
    print(f'Subdivided Mesh Info: {mesh_subdivided}')
    mesh_subdivided.export('assets/assignment1/cube_subdivided.obj')
    
    # quadratic error mesh decimation
    mesh_decimated = mesh.simplify_quadric_decimation(4)
    
    # TODO: implement your own quadratic error mesh decimation here
    # mesh_decimated = simplify_quadric_error(mesh, face_count=1)
    
    # print the new mesh information and save the mesh
    print(f'Decimated Mesh Info: {mesh_decimated}')
    mesh_decimated.export('assets/assignment1/cube_decimated.obj')