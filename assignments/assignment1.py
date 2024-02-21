import trimesh
from trimesh import graph, grouping
from trimesh.geometry import faces_to_edges
import numpy as np
from itertools import zip_longest
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


def draw_arrows(arrow_start_list, arrow_end_list, point_list):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for start, end in zip(arrow_start_list, arrow_end_list):
        # Calculate the difference between the start and end points
        arrow = end - start
        # Plot the arrow
        ax.quiver(start[0], start[1], start[2], arrow[0], arrow[1], arrow[2], arrow_length_ratio=0.1)

    # Plot the points in red and annotate them with their index
    for i, point in enumerate(point_list):
        ax.scatter(point[0], point[1], point[2], c='r')
        ax.text(point[0], point[1], point[2], f'{i}', color='blue',size=30)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

class HalfEdge:
    def __init__(self, index):
        self.index = index
        self.twin = None
        self.next = None
        self.prev = None
        self.origin = None
        self.face = None

    def set_twin(self, twin):
        self.twin = twin
        twin.twin = self

    def set_next(self, next):
        self.next = next
        next.prev = self

    def set_prev(self, prev):
        self.prev = prev
        prev.next = self

    def set_origin(self, origin):
        self.origin = origin

    def set_face(self, face):
        self.face = face
    
    @property
    def is_boundary(self):
        return self.twin is None
    
    def __eq__(self, other):
        if isinstance(other, HalfEdge):
            return self.index == other.index
        return False
    
    def __str__(self):
        return f"HalfEdge(Index: {self.index}, Origin: {self.origin}, Next: {self.next.index if self.next else None}, Prev: {self.prev.index if self.prev else None}, Twin: {self.twin.index if self.twin else None}, Face: {self.face})"
    def __repr__(self):
        return str(self)

def build_topology(vertices, faces):
    new_vertices = vertices.tolist()
    new_faces = []
    edge_midpoint_index = {}  
    is_odd_vertex = [False] * len(vertices)

    def get_midpoint_index(v1, v2):
        edge = tuple(sorted([v1, v2]))
        if edge not in edge_midpoint_index:
            midpoint = (vertices[v1] + vertices[v2]) / 2
            edge_midpoint_index[edge] = len(new_vertices)
            new_vertices.append(midpoint)
            is_odd_vertex.append(True)
        return edge_midpoint_index[edge]

    for face in faces:
        midpoints = [get_midpoint_index(face[i], face[(i + 1) % 3]) for i in range(3)]
        new_faces.append([face[0], midpoints[0], midpoints[2]])
        new_faces.append([face[1], midpoints[1], midpoints[0]])
        new_faces.append([face[2], midpoints[2], midpoints[1]])
        new_faces.append(midpoints)

    return np.array(new_vertices), np.array(new_faces), np.array(is_odd_vertex)

def build_half_edges(vertices, faces):
    half_edges = {}
    half_edge_index = 0
    vertices_he, faces_he = [None] * len(vertices), [None] * len(faces)
    for face_index, face in enumerate(faces):
        face_half_edge = []
        for i in range(len(face)):
            origin_vertex_index = face[i]

            half_edge = HalfEdge(index=half_edge_index)
            half_edge.set_origin(origin_vertex_index)
            half_edge.set_face(face_index)
            vertices_he[origin_vertex_index] = half_edge
            half_edges[half_edge_index] = half_edge
            face_half_edge.append(half_edge)

            half_edge_index += 1
        # Link the half-edges of the current face
        for i, this_half_edge in enumerate(face_half_edge):
            next_half_edge = face_half_edge[(i + 1) % len(face_half_edge)]
            this_half_edge.set_next(next_half_edge)
            next_half_edge.set_prev(this_half_edge)
        faces_he[face_index] = half_edge
    # Link the twin half-edges
    for half_edge in half_edges.values():
        for potential_twin in half_edges.values():
            if (half_edge.origin == potential_twin.next.origin) and (half_edge.next.origin == potential_twin.origin):
                half_edge.set_twin(potential_twin)
                break

    return half_edges, vertices_he, faces_he

def get_boundary(vertices_he):
    is_boundary_vertex = []
    for i in range(len(vertices_he)):
        flag = False
        start_he = vertices_he[i]
        he = start_he
        while(he != start_he):
            if he.twin == None:
                flag = True
            he = he.prev.twin
        is_boundary_vertex.append(flag)
    return np.array(is_boundary_vertex)

def subdivision_loop_halfedge(mesh):
    # prepare geometry for the loop subdivision
    vertices, faces = mesh.vertices, mesh.faces # [N_vertices, 3] [N_faces, 3]
    temp_vertices, temp_faces, is_odd_vertex = build_topology(vertices, faces)
    half_edges, temp_vertices_he, temp_faces_he = build_half_edges(temp_vertices, temp_faces)
    is_boundary_vertex = get_boundary(temp_vertices_he)
    new_vertices = []
    for i in range(len(temp_vertices)):
        new_vertex_calc_dict = {}
        if is_odd_vertex[i] and is_boundary_vertex[i]:
            # Case 1: odd vertex and boundary
            neighbor_1d = []
            start_he = temp_vertices_he[i]
            he = start_he
            while True:
                neighbor_1d.append(he.origin)
                he = he.prev.twin
                if he == start_he:
                    break
            for n_1d in neighbors_1d:
                if not is_odd_vertex[n_1d]:
                    new_vertex_calc_dict[n_1d] = 1 / 2
            new_vertex = np.zeros_like(temp_vertices[0])
            for v, coeff in new_vertex_calc_dict.items():
                new_vertex += coeff * temp_vertices[v]
            new_vertices.append(new_vertex)

        elif is_odd_vertex[i] and not is_boundary_vertex[i]:
            # Case 2: odd vertex and not boundary
            neighbors_1d, neighbors_2d = [], []
            start_he = temp_vertices_he[i]
            he = start_he
            # Find all 1d neighbor
            while True:
                neighbors_1d.append(he.next.origin)
                he = he.prev.twin
                if he == start_he:
                    break
            # Find all 2d neighbor by expanding 2 steps
            for n_1d in neighbors_1d:
                start_he = temp_vertices_he[n_1d]
                he = start_he
                while True:
                    if he.next.origin not in neighbors_2d:
                        neighbors_2d.append(he.next.origin)
                    he = he.prev.twin
                    if he == start_he:
                        break
            for n_1d in neighbors_1d:
                if not is_odd_vertex[n_1d]:
                    new_vertex_calc_dict[n_1d] = 3 / 8

            for n_2d in neighbors_2d:
                if not is_odd_vertex[n_2d] and n_2d not in neighbors_1d and n_2d != i:
                    new_vertex_calc_dict[n_2d] = 1 / 8

            new_vertex = np.zeros_like(temp_vertices[0])
            for vertex_index, coeff in new_vertex_calc_dict.items():
                new_vertex += coeff * temp_vertices[vertex_index]
            new_vertices.append(new_vertex)

        elif not is_odd_vertex[i] and is_boundary_vertex[i]:
            # Case 3: even vertex and boundary
            neighbors_1d, neighbors_2d = [], []
            start_he = temp_vertices_he[i]
            he = start_he
            # Find all 1d neighbor
            while True:
                neighbors_1d.append(he.next.origin)
                he = he.prev.twin
                if he == start_he:
                    break
            # Find all 2d neighbor by expanding 2 steps
            for n_1d in neighbors_1d:
                start_he = temp_vertices_he[n_1d]
                he = start_he
                while True:
                    if he.next.origin not in neighbors_2d:
                        neighbors_2d.append(he.next.origin)
                    he = he.prev.twin
                    if he == start_he:
                        break

            for n_2d in neighbors_2d:
                if not is_odd_vertex[n_2d] and is_boundary_vertex[n_2d] and n_2d != i:
                    new_vertex_calc_dict[n_1d] = 1 / 8
        
            new_vertex = np.zeros_like(temp_vertices[0])
            for v, coeff in new_vertex_calc_dict.items():
                new_vertex += coeff * temp_vertices[v]
            k = len(new_vertex_calc_dict)
            new_vertex += 3 / 4 *temp_vertices[i]
            new_vertices.append(new_vertex)
        
        elif not is_odd_vertex[i] and not is_boundary_vertex[i]:
            beta = 1
            # Case 4: even vertex and not boundary
            neighbors_1d, neighbors_2d = [], []
            start_he = temp_vertices_he[i]
            he = start_he
            # Find all 1d neighbor
            while True:
                neighbors_1d.append(he.next.origin)
                he = he.prev.twin
                if he == start_he:
                    break
            # Find all 2d neighbor by expanding 2 steps
            for n_1d in neighbors_1d:
                start_he = temp_vertices_he[n_1d]
                he = start_he
                while True:
                    if he.next.origin not in neighbors_2d:
                        print(f"For {i}, Add {n_1d}'s neighbor {he.next.origin}")
                        neighbors_2d.append(he.next.origin)
                    he = he.prev.twin
                    if he == start_he:
                        break
            for n_2d in neighbors_2d:
                if not is_odd_vertex[n_2d] and  n_2d != i:
                    new_vertex_calc_dict[n_2d] = beta
            print(f"{i}, Len: {len(new_vertex_calc_dict), new_vertex_calc_dict.keys()}")
            new_vertex = np.zeros_like(temp_vertices[0])
            k = len(new_vertex_calc_dict)
            beta = (40.0 - (2.0 * np.cos(2 * np.pi / k) + 3) ** 2) / (64 * k)
            for v, _ in new_vertex_calc_dict.items():
                new_vertex += beta * temp_vertices[v]
            new_vertex += (1-k*beta)*temp_vertices[i]
            new_vertices.append(new_vertex)

    mesh.vertices = np.array(new_vertices)
    mesh.faces = np.array(temp_faces)
    return mesh

def subdivision_loop(mesh):
    """
    Apply Loop subdivision to the input mesh for the specified number of iterations.
    :param mesh: input mesh
    :param iterations: number of iterations
    :return: mesh after subdivision
    
    Overall process:
    Reference: https://github.com/mikedh/trimesh/blob/main/trimesh/remesh.py#L207
    1. Calculate odd vertices.
      Assign a new odd vertex on each edge and
      calculate the value for the boundary case and the interior case.
      The value is calculated as follows.
          v2
        / f0 \\        0
      v0--e--v1      /   \\
        \\f1 /     v0--e--v1
          v3
      - interior case : 3:1 ratio of mean(v0,v1) and mean(v2,v3)
      - boundary case : mean(v0,v1)
    2. Calculate even vertices.
      The new even vertices are calculated with the existing
      vertices and their adjacent vertices.
        1---2
       / \\/ \\      0---1
      0---v---3     / \\/ \\
       \\ /\\/    b0---v---b1
        k...4
      - interior case : (1-kB):B ratio of v and k adjacencies
      - boundary case : 3:1 ratio of v and mean(b0,b1)
    3. Compose new faces with new vertices.
    
    # The following implementation considers only the interior cases
    # You should also consider the boundary cases and more iterations in your submission
    """

    # prepare geometry for the loop subdivision
    vertices, faces = mesh.vertices, mesh.faces # [N_vertices, 3] [N_faces, 3]
    edges, edges_face = faces_to_edges(faces, return_index=True) # [N_edges, 2], [N_edges]
    edges.sort(axis=1)
    unique, inverse = grouping.unique_rows(edges)
    
    # split edges to interior edges and boundary edges
    edge_inter = np.sort(grouping.group_rows(edges, require_count=2), axis=1)
    edge_bound = grouping.group_rows(edges, require_count=1)
    
    # set also the mask for interior edges and boundary edges
    edge_bound_mask = np.zeros(len(edges), dtype=bool)
    edge_bound_mask[edge_bound] = True
    edge_bound_mask = edge_bound_mask[unique]
    edge_inter_mask = ~edge_bound_mask
    
    ###########
    # Step 1: #
    ###########
    # Calculate odd vertices to the middle of each edge.
    odd = vertices[edges[unique]].mean(axis=1) # [N_oddvertices, 3]
    
    # connect the odd vertices with even vertices
    # however, the odd vertices need further updates over it's position
    # we therefore complete this step later afterwards.
    
    ###########
    # Step 2: #
    ###########
    # find v0, v1, v2, v3 and each odd vertex
    # v0 and v1 are at the end of the edge where the generated odd vertex on
    # locate the edge first
    e = edges[unique[edge_inter_mask]]
    # locate the endpoints for each edge
    e_v0 = vertices[e][:, 0]
    e_v1 = vertices[e][:, 1]
    
    # v2 and v3 are at the farmost position of the two triangle
    # locate the two triangle face
    edge_pair = np.zeros(len(edges)).astype(int)
    edge_pair[edge_inter[:, 0]] = edge_inter[:, 1]
    edge_pair[edge_inter[:, 1]] = edge_inter[:, 0]
    opposite_face1 = edges_face[unique]
    opposite_face2 = edges_face[edge_pair[unique]]
    # locate the corresponding edge
    e_f0 = faces[opposite_face1[edge_inter_mask]]
    e_f1 = faces[opposite_face2[edge_inter_mask]]
    # locate the vertex index and vertex location
    e_v2_idx = e_f0[~(e_f0[:, :, None] == e[:, None, :]).any(-1)]
    e_v3_idx = e_f1[~(e_f1[:, :, None] == e[:, None, :]).any(-1)]
    e_v2 = vertices[e_v2_idx]
    e_v3 = vertices[e_v3_idx]
    
    # update the odd vertices based the v0, v1, v2, v3, based the following:
    # 3 / 8 * (e_v0 + e_v1) + 1 / 8 * (e_v2 + e_v3)
    odd[edge_inter_mask] = 0.375 * e_v0 + 0.375 * e_v1 + e_v2 / 8.0 + e_v3 / 8.0
    
    ###########
    # Step 3: #
    ###########
    # find vertex neightbors for even vertices and update accordingly
    neighbors = graph.neighbors(edges=edges[unique], max_index=len(vertices))
    # convert list type of array into a fixed-shaped numpy array (set -1 to empties)
    neighbors = np.array(list(zip_longest(*neighbors, fillvalue=-1))).T
    # if the neighbor has -1 index, its point is (0, 0, 0), so that it is not included in the summation of neighbors when calculating the even
    vertices_ = np.vstack([vertices, [0.0, 0.0, 0.0]])
    # number of neighbors
    k = (neighbors + 1).astype(bool).sum(axis=1)
    
    # calculate even vertices for the interior case
    beta = (40.0 - (2.0 * np.cos(2 * np.pi / k) + 3) ** 2) / (64 * k)
    even = (
        beta[:, None] * vertices_[neighbors].sum(1)
        + (1 - k[:, None] * beta[:, None]) * vertices
    )
    
    ############
    # Step 1+: #
    ############
    # complete the subdivision by updating the vertex list and face list
    
    # the new faces with odd vertices
    odd_idx = inverse.reshape((-1, 3)) + len(vertices)
    new_faces = np.column_stack(
        [
            faces[:, 0],
            odd_idx[:, 0],
            odd_idx[:, 2],
            odd_idx[:, 0],
            faces[:, 1],
            odd_idx[:, 1],
            odd_idx[:, 2],
            odd_idx[:, 1],
            faces[:, 2],
            odd_idx[:, 0],
            odd_idx[:, 1],
            odd_idx[:, 2],
        ]
    ).reshape((-1, 3)) # [N_face*4, 3]

    # stack the new even vertices and odd vertices
    new_vertices = np.vstack((even, odd)) # [N_vertex+N_edge, 3]
    
    return trimesh.Trimesh(new_vertices, new_faces)

class Vertex:
    def __init__(self, index, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.index = index
    def set_Q(self, Q):
        self.Q = Q
    def set_pos(self,pos):
        self.x, self.y, self.z = pos
    @property
    def pos(self):
        return np.array([self.x,self.y,self.z]) 
    def __str__(self):
        return f"Vertex {self.index}: ({self.x}, {self.y}, {self.z})"
    def __repr__(self):
        return str(self)
    def __eq__(self, other):
        if isinstance(other, Vertex):
            return self.index == other.index
        return False

class Edge:
    def __init__(self, index, v1, v2):
        self.v1 = v1
        self.v2 = v2
        self.index = index
    def set_Q(self, Q):
        self.Q = Q
    def update_Q(self):
        self.Q = self.v1.Q + self.v2.Q
        self.v_optimal = compute_optimal_vertex(self.Q)
        self.error = np.dot(self.v_optimal, np.dot(self.Q, self.v_optimal))

    def __str__(self):
        return f"Edge {self.index}: ({self.v1.index}, {self.v2.index})"
    def __repr__(self):
        return str(self)
    
class Face:
    def __init__(self, index, v1, v2, v3):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.index = index
    def set_Q(self, Q):
        self.Q = Q
    def update_vertex_Q(self):
        self.v1.set_Q(self.v1.Q + self.Q)
        self.v2.set_Q(self.v2.Q + self.Q)
        self.v3.set_Q(self.v3.Q + self.Q)
    @property
    def vertices(self):
        return [self.v1, self.v2, self.v3]
    @property
    def vertices_index(self):
        return [self.v1.index, self.v2.index, self.v3.index]   
    
    def __str__(self):
        return f"Face {self.index}: ({self.v1.index}, {self.v2.index}, {self.v3.index})"
    def __repr__(self):
        return str(self) 

def compute_plane_equation(v1, v2, v3):
    normal = np.cross(v2 - v1, v3 - v1)
    normal = normal / np.linalg.norm(normal)
    d = -np.dot(normal, v1)
    return np.append(normal, d)

def init_error_quadric(vertices, faces):
    for vertex in vertices.values():
        vertex.set_Q(np.zeros((4, 4)))
    for face in faces.values():
        plane_eq = compute_plane_equation(face.v1.pos, face.v2.pos, face.v3.pos)
        Q_face = np.outer(plane_eq, plane_eq)
        face.set_Q(Q_face)
        face.update_vertex_Q()

def compute_optimal_vertex(Q):
    Q_reduced = Q[:-1, :-1]
    v, _, _, _ = np.linalg.lstsq(Q_reduced, -Q[:-1, 3], rcond=None)
    v = np.append(v, 1)
    return v

def build_edges(vertices, faces):
    edges = {}
    for face in faces.values():
        face_sort = sorted([face.v1.index, face.v2.index, face.v3.index])
        for i,j in [(0,1),(1,2),(0,2)]:
            edge_index = (face_sort[i], face_sort[j])
            if edge_index not in edges:
                edge = Edge(edge_index,vertices[edge_index[0]], vertices[edge_index[1]])
                edge.update_Q() 
                edges[edge_index] = edge
    return edges

def update_edges(edges, edge_to_collapse):
    for (v1_index, v2_index), edge in list(edges.items()):
        if v1_index == edge_to_collapse.v2.index:
            new_key = (edge_to_collapse.v1.index, v2_index)
            if new_key[0] != new_key[1]:
                edge.v1 = edge_to_collapse.v1
                edge.index = new_key
                edge.update_Q()
                edges[new_key] = edge
            del edges[(v1_index, v2_index)]
        elif v2_index == edge_to_collapse.v2.index:
            new_key = (v1_index, edge_to_collapse.v1.index)
            if new_key[0] != new_key[1]:
                edge.v2 = edge_to_collapse.v1
                edge.index = new_key
                edge.update_Q()
                edges[new_key] = edge
            del edges[(v1_index, v2_index)]

def update_faces(faces, edge_to_collapse):
    for face_index, face in list(faces.items()):  
        is_v1_in = edge_to_collapse.v1 in face.vertices
        is_v2_in = edge_to_collapse.v2 in face.vertices

        if is_v1_in and is_v2_in:
            del faces[face_index]
        elif not is_v1_in and is_v2_in:
            if face.v1 == edge_to_collapse.v2:
                face.v1 = edge_to_collapse.v1
            elif face.v2 == edge_to_collapse.v2:
                face.v2 = edge_to_collapse.v1
            elif face.v3 == edge_to_collapse.v2:
                face.v3 = edge_to_collapse.v1


def get_neighbor_vertices(base_vertices, edges):
    neighbor_vertices = []
    for vertex in base_vertices:
        for (v1_i,v2_i), edge in edges.values():
            if v1_i == vertex.index and edge.v2 not in neighbor_vertices:
                neighbor_vertices.append(edge.v2)
            if v2_i == vertex.index and edge.v1 not in neighbor_vertices:
                neighbor_vertices.append(edge.v1)
    return neighbor_vertices


def simplify_quadric_error(mesh, face_count=1):
    vertices = {i:Vertex(i, mesh.vertices[i][0], mesh.vertices[i][1], mesh.vertices[i][2]) for i in range(mesh.vertices.shape[0])}
    faces = {i:Face(i, vertices[mesh.faces[i][0]], vertices[mesh.faces[i][1]], vertices[mesh.faces[i][2]]) for i in range(mesh.faces.shape[0])}
    init_error_quadric(vertices, faces)
    edges = build_edges(vertices, faces)
    start_time = time.time()
    iteration = 0
    while len(faces) > face_count:
        edge_to_collapse = min(edges.values(), key=lambda edge: edge.error)
        # update v1
        edge_to_collapse.v1.set_pos(edge_to_collapse.v_optimal[:3])
        edge_to_collapse.v1.set_Q(edge_to_collapse.v1.Q + edge_to_collapse.v2.Q)
        
        # update topology
        update_faces(faces, edge_to_collapse)
        update_edges(edges, edge_to_collapse)
        iteration += 1
        if iteration % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"\rIteration {iteration}: {len(faces)} faces remaining, elapsed time: {elapsed_time:.2f} seconds", end='', flush=True)
    print('\n')
    mesh.vertices = np.array([vertex.pos for vertex in vertices.values()])
    mesh.faces = np.array([face.vertices_index for face in faces.values()])
    return mesh


if __name__ == '__main__':
    # Load mesh and print information
    mesh = trimesh.load_mesh('assets/bunny.obj')
    # mesh = trimesh.creation.box(extents=[1, 1, 1])
    print(f'Mesh Info: {mesh}')
    
    # apply loop subdivision over the loaded mesh
    # mesh_subdivided = mesh.subdivide_loop(iterations=1)
    
    # TODO: implement your own loop subdivision here
    # mesh_subdivided = subdivision_loop_halfedge(mesh)
    
    # print the new mesh information and save the mesh
    # print(f'Subdivided Mesh Info: {mesh_subdivided}')
    # mesh_subdivided.export('assets/assignment1/cube_subdivided.obj')
    
    # # quadratic error mesh decimation
    # mesh_decimated = mesh.simplify_quadric_decimation(6)
    # print("vertices:", mesh_decimated.vertices)
    # print("faces:", mesh_decimated.faces)
    # # TODO: implement your own quadratic error mesh decimation here
    mesh_decimated = simplify_quadric_error(mesh, face_count=500)
    # # print the new mesh information and save the mesh
    # print(f'Decimated Mesh Info: {mesh_decimated}')
    mesh_decimated.export('assets/assignment1/bunny_decimated_500.obj')