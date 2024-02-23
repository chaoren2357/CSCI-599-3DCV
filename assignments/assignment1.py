import trimesh
from trimesh import graph, grouping
from trimesh.geometry import faces_to_edges
import numpy as np
from itertools import zip_longest
import time


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


# given the orginal vertices and faces, it will return the subdivided vertex and face, and a flag for odd vertex
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

# given vertices and faces, return a half edge structure
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

# judge whether a half-edge is a boundary edge
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

def subdivision_loop_halfedge(mesh, iteration = 1):
    for _ in range(iteration):
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
                    if he.prev.twin is not None:
                        he = he.prev.twin  
                    else:
                        # Handle the boundary case, find the next boundary half-edge in the loop
                        while he.next != start_he and he.next.twin is not None:
                            he = he.next.twin.prev
                        if he.next == start_he:
                            break  
                        he = he.next  
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
                    if he.prev.twin is not None:
                        he = he.prev.twin  
                    else:
                        # Handle the boundary case, find the next boundary half-edge in the loop
                        while he.next != start_he and he.next.twin is not None:
                            he = he.next.twin.prev
                        if he.next == start_he:
                            break  
                        he = he.next  
                    if he == start_he:
                        break  
                # Find all 2d neighbor by expanding 2 steps
                for n_1d in neighbors_1d:
                    start_he = temp_vertices_he[n_1d]
                    he = start_he
                    while True:
                        if he.next.origin not in neighbors_2d:
                            neighbors_2d.append(he.next.origin)
                        if he.prev.twin is not None:
                            he = he.prev.twin  
                        else:
                            # Handle the boundary case, find the next boundary half-edge in the loop
                            while he.next != start_he and he.next.twin is not None:
                                he = he.next.twin.prev
                            if he.next == start_he:
                                break  
                            he = he.next  
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
                    if he.prev.twin is not None:
                        he = he.prev.twin  
                    else:
                        # Handle the boundary case, find the next boundary half-edge in the loop
                        while he.next != start_he and he.next.twin is not None:
                            he = he.next.twin.prev
                        if he.next == start_he:
                            break  
                        he = he.next  
                    if he == start_he:
                        break  
                # Find all 2d neighbor by expanding 2 steps
                for n_1d in neighbors_1d:
                    start_he = temp_vertices_he[n_1d]
                    he = start_he
                    while True:
                        if he.next.origin not in neighbors_2d:
                            neighbors_2d.append(he.next.origin)
                        if he.prev.twin is not None:
                            he = he.prev.twin  
                        else:
                            # Handle the boundary case, find the next boundary half-edge in the loop
                            while he.next != start_he and he.next.twin is not None:
                                he = he.next.twin.prev
                            if he.next == start_he:
                                break  
                            he = he.next  
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
                    if he.prev.twin is not None:
                        he = he.prev.twin  
                    else:
                        # Handle the boundary case, find the next boundary half-edge in the loop
                        while he.next != start_he and he.next.twin is not None:
                            he = he.next.twin.prev
                        if he.next == start_he:
                            break  
                        he = he.next  
                    if he == start_he:
                        break  
                # Find all 2d neighbor by expanding 2 steps
                for n_1d in neighbors_1d:
                    start_he = temp_vertices_he[n_1d]
                    he = start_he
                    while True:
                        if he.next.origin not in neighbors_2d:
                            neighbors_2d.append(he.next.origin)
                        if he.prev.twin is not None:
                            he = he.prev.twin  
                        else:
                            # Handle the boundary case, find the next boundary half-edge in the loop
                            while he.next != start_he and he.next.twin is not None:
                                he = he.next.twin.prev
                            if he.next == start_he:
                                break  
                            he = he.next  
                        if he == start_he:
                            break  
                for n_2d in neighbors_2d:
                    if not is_odd_vertex[n_2d] and  n_2d != i:
                        new_vertex_calc_dict[n_2d] = beta
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

def subdivision_loop(mesh, iterations = 1):
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
    for _ in range(iterations):
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
        mesh = trimesh.Trimesh(new_vertices, new_faces)
    
    return mesh

class Vertex:
    def __init__(self, index, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.index = index

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
    def __init__(self, index, v1_index, v2_index, Q = None):
        self.v1_index = v1_index
        self.v2_index = v2_index
        self.index = index

    def is_in(self, vertex_index):
        return vertex_index in [self.v1_index, self.v2_index]
    
    def get_replace_index(self, vertex_index, new_index):
        if vertex_index == self.v1_index:
            edge_index = tuple(sorted((self.v2_index, new_index)))
            edge = Edge(edge_index, edge_index[0], edge_index[1])
            return edge
        elif vertex_index == self.v2_index:
            edge_index = tuple(sorted((self.v1_index, new_index)))
            edge = Edge(edge_index, edge_index[0], edge_index[1])
            return edge
        else:
            return None

    def __str__(self):
        return f"Edge {self.index}: ({self.v1_index}, {self.v2_index})"
    def __repr__(self):
        return str(self)
    
class Face:
    def __init__(self, index, v1_index, v2_index, v3_index, Q=None):
        self.v1_index = v1_index
        self.v2_index = v2_index
        self.v3_index = v3_index
        self.index = index

    def is_in(self, vertex_index):
        return vertex_index in [self.v1_index, self.v2_index, self.v3_index]
    def get_replace_index(self, index, vertex_index, new_index):
        if vertex_index == self.v1_index:
            return Face(index, new_index, self.v2_index, self.v3_index)
        elif vertex_index == self.v2_index:
            return Face(index, self.v1_index, new_index, self.v3_index)
        elif vertex_index == self.v3_index:
            return Face(index, self.v1_index, self.v2_index, new_index)
        else:
            return None
    @property
    def vertices_index(self):
        return [self.v1_index, self.v2_index, self.v3_index]
    
    def __str__(self):
        return f"Face {self.index}: ({self.v1_index}, {self.v2_index}, {self.v3_index})"
    def __repr__(self):
        return str(self) 

def update_vertex_Q_by_face(vertices_Q, face_Q, face):
    vertices_Q[face.v1_index] += face_Q
    vertices_Q[face.v2_index] += face_Q
    vertices_Q[face.v3_index] += face_Q
    return vertices_Q

def update_edge_Q(vertices_Q, edges_Q, edge_index):
    v1_index, v2_index = edge_index
    edges_Q.update({edge_index:vertices_Q[v1_index] + vertices_Q[v2_index]})
    return edges_Q

def update_edge_info(edges_info, edge_Q, edge_index):
    v_optimal, error = get_optimV_error(edge_Q)
    edges_info.update({edge_index:(v_optimal, error)})
    return edges_info

def get_optimV_error(edge_Q):
    v_optimal = compute_optimal_vertex(edge_Q)
    error = np.dot(v_optimal, np.dot(edge_Q, v_optimal))
    return v_optimal, error

def get_new_index(dictionary):
    return max(dictionary.keys()) + 1

def compute_plane_equation(v1, v2, v3):
    normal = np.cross(v2 - v1, v3 - v1)
    normal = normal / np.linalg.norm(normal)
    d = -np.dot(normal, v1)
    return np.append(normal, d)

# comupte optimal vertex position from given Q matrix
def compute_optimal_vertex(Q):
    Q_reduced = Q[:-1, :-1]
    v, _, _, _ = np.linalg.lstsq(Q_reduced, -Q[:-1, 3], rcond=None)
    v = np.append(v, 1)
    return v

## given the collapse edge, get the optimal vertex position and update vertices
def update_vertices(vertices, vertices_Q, edge_to_collapse, edges_info):
    x_opt, y_opt, z_opt = edges_info.get(edge_to_collapse.index)[0][:3]
    new_index = get_new_index(vertices)
    new_vertex = Vertex(new_index, x_opt, y_opt, z_opt)
    v1_Q, v2_Q = vertices_Q[edge_to_collapse.v1_index], vertices_Q[edge_to_collapse.v2_index]
    vertices_Q.update({new_index: v1_Q + v2_Q})
    vertices.update({new_index: new_vertex})
    del vertices_Q[edge_to_collapse.v1_index]
    del vertices_Q[edge_to_collapse.v2_index]
    del vertices[edge_to_collapse.v1_index]
    del vertices[edge_to_collapse.v2_index]
    return vertices, vertices_Q, new_vertex

# given a new vertex and the collapse edge, update edge topology and compute the new Q value for new edges
def update_edges(edges, edges_Q, edges_info, vertices_Q, new_vertex, edge_to_collapse):
    new_edges = {}
    
    for (v1_index, v2_index), edge in edges.items():
        is_v1_in = edge.is_in(edge_to_collapse.v1_index)
        is_v2_in = edge.is_in(edge_to_collapse.v2_index)
        if is_v1_in and is_v2_in:
            del edges_Q[(v1_index, v2_index)]
            del edges_info[(v1_index, v2_index)]
        elif is_v1_in and not is_v2_in:
            new_edge = edge.get_replace_index(edge_to_collapse.v1_index, new_vertex.index)
            new_index = new_edge.index
            del edges_Q[(v1_index, v2_index)]
            del edges_info[(v1_index, v2_index)]
            edges_Q = update_edge_Q(vertices_Q, edges_Q, new_index)   
            edges_info = update_edge_info(edges_info, edges_Q.get(new_index), new_index)  
            new_edges.update({new_index:new_edge})

        elif not is_v1_in and is_v2_in:
            new_edge = edge.get_replace_index(edge_to_collapse.v2_index, new_vertex.index)
            new_index = new_edge.index
            del edges_Q[(v1_index, v2_index)]
            del edges_info[(v1_index, v2_index)]
            edges_Q = update_edge_Q(vertices_Q, edges_Q, new_index)  
            edges_info = update_edge_info(edges_info, edges_Q.get(new_index), new_index)
            new_edges.update({new_index:new_edge})
        else:
            new_edges.update({(v1_index, v2_index):edge})

    return new_edges, edges_Q, edges_info

# given a new vertex and the collapse edge, update face topology
def update_faces(faces, new_vertex, edge_to_collapse):
    new_faces = {}
    max_face_index = get_new_index(faces) - 1
    for face_index, face in faces.items():  
        is_v1_in = face.is_in(edge_to_collapse.v1_index)
        is_v2_in = face.is_in(edge_to_collapse.v2_index)
        if is_v1_in and is_v2_in:
            continue
        elif is_v1_in and not is_v2_in:
            new_index = max_face_index + 1
            new_face = face.get_replace_index(new_index, edge_to_collapse.v1_index, new_vertex.index)
            new_faces.update({new_index:new_face})
            max_face_index += 1
        elif not is_v1_in and is_v2_in:
            new_index = max_face_index + 1
            new_face = face.get_replace_index(new_index, edge_to_collapse.v2_index, new_vertex.index)
            new_faces.update({new_index:new_face})
            max_face_index += 1
        else:
            new_faces.update({face_index:face})
    return new_faces

# convert my class data to numpy data 
def convert_to_numpy(vertices, faces):
    old2new = {}
    new_idx = 0
    vertices_np, faces_np = [],[]
    for idx, vertex in vertices.items():
        old2new.update({idx:new_idx})
        new_idx += 1
        vertices_np.append(vertex.pos)
    for face in faces.values():
        faces_np.append([old2new[face.v1_index], old2new[face.v2_index], old2new[face.v3_index]])
    return np.array(vertices_np), np.array(faces_np)

def simplify_quadric_error(mesh, face_count=1):
    # initialize vertices
    vertices = {}
    vertices_Q = {}
    for i in range(mesh.vertices.shape[0]):
        new_vertex = Vertex(i, mesh.vertices[i][0], mesh.vertices[i][1], mesh.vertices[i][2])
        vertices.update({i:new_vertex})
        vertices_Q.update({i:np.zeros((4, 4))})
    # initialize faces
    faces = {}
    faces_Q = {}
    for i in range(mesh.faces.shape[0]):
        new_face = Face(i, mesh.faces[i][0], mesh.faces[i][1], mesh.faces[i][2])
        v1, v2, v3 = vertices[new_face.v1_index], vertices[new_face.v2_index], vertices[new_face.v3_index]
        plane_eq = compute_plane_equation(v1.pos, v2.pos, v3.pos)
        Q_face = np.outer(plane_eq, plane_eq)
        faces_Q.update({i:Q_face})
        vertices_Q = update_vertex_Q_by_face(vertices_Q, faces_Q[i], new_face)       
        faces.update({i: new_face})
    
    # initialize edges
    edges = {}
    edges_Q = {}
    for face in faces.values():
        face_sort = sorted([face.v1_index, face.v2_index, face.v3_index])
        for i,j in [(0,1),(1,2),(0,2)]:
            edge_index = (face_sort[i], face_sort[j])
            if edge_index not in edges:
                edge = Edge(edge_index, edge_index[0], edge_index[1])
                edges_Q = update_edge_Q(vertices_Q, edges_Q, edge_index)
                edges[edge_index] = edge
    
    # initialize (optimal position, error) info
    edges_info = {}
    for edge_index, edge_Q in edges_Q.items():
        edges_info = update_edge_info(edges_info, edge_Q, edge_index)
    # main loop
    start_time = time.time()
    iteration = 0
    while len(faces) > face_count:

        edge_to_collapse = edges[min(edges_info, key=lambda x: edges_info[x][1])]

        # update topology
        vertices, vertices_Q, new_vertex = update_vertices(vertices, vertices_Q, edge_to_collapse, edges_info)
        faces = update_faces(faces, new_vertex, edge_to_collapse)
        edges, edges_Q, edges_info = update_edges(edges, edges_Q, edges_info, vertices_Q, new_vertex, edge_to_collapse)
        
        # estimate time
        iteration += 1
        if iteration % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"\rIteration {iteration}: {len(faces)} faces remaining, {len(edges)} edges remaining,elapsed time: {elapsed_time:.2f} seconds", end='', flush=True)
    print('\n')
    mesh.vertices, mesh.faces = convert_to_numpy(vertices, faces)
    return mesh


if __name__ == '__main__':
    # Load mesh and print information
    # mesh = trimesh.load_mesh('assets/bunny.obj')
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    print(f'Mesh Info: {mesh}')

    iter = 4
    # apply loop subdivision over the loaded mesh
    # start_time = time.time()
    # mesh_subdivided = mesh.subdivide_loop(iterations=iter)
    mesh_subdivided = subdivision_loop(mesh, iterations=iter)
    # end_time = time.time()
    # print(end_time - start_time)
    mesh_subdivided.export(f'assets/assignment1/cube_subdivided_{iter}_gt.obj')
    # TODO: implement your own loop subdivision here
    
    # start_time = time.time()
    # mesh_subdivided = subdivision_loop_halfedge(mesh, iter)
    # end_time = time.time()
    # print(end_time - start_time)
    # mesh_subdivided.export(f'assets/assignment1/cube_subdivided_{iter}_he.obj')
    # print the new mesh information and save the mesh
    # print(f'Subdivided Mesh Info: {mesh_subdivided}')
    # mesh_subdivided.export('assets/assignment1/cube_subdivided.obj')
    
    # # quadratic error mesh decimation
    # mesh_decimated = mesh.simplify_quadric_decimation(2000)
    # mesh_decimated.export('assets/assignment1/face_decimated_2000_gt.obj')
    # print("vertices:", mesh_decimated.vertices)
    # print("faces:", mesh_decimated.faces)
    # # TODO: implement your own quadratic error mesh decimation here
    # mesh_decimated = simplify_quadric_error(mesh, face_count=10000)
    # # print the new mesh information and save the mesh
    # print(f'Decimated Mesh Info: {mesh_decimated}')
    # mesh_decimated.export('assets/assignment1/face_decimated_10000.obj')