import numpy as np
import numpy.linalg as LA

#TODO: update docstrings?

def get_edge_face_neighbors (shape) -> np.ndarray:
    '''
    Gets the indices of the faces that are adjacent to each edge.

    Args:
        shape (Polyhedron): the polyhedron that is being looked at (can be convex or concave)

    Returns:
        np.ndarray: the indices of the nearest faces for each edge [shape = (n_edges, 2)]
    '''
    faces_len = shape.num_faces
    num_edges = shape.num_edges

    #appending the vertex list of each face and a -1 to the end of each list of vertices, then flattening the awkward array (the -1 indicates a change in face)
    faces_flat = []

    for face in shape.faces:
        face_extra = np.array([face[0], -1])

        faces_flat = np.append(faces_flat, face)
        faces_flat = np.append(faces_flat, face_extra)

    faces_flat = np.asarray(faces_flat)

    #creating a matrix where each row corresponds to an edge that contains the indices of its two corresponding vertices (a -1 index indicates a change in face)
    list_len = len(faces_flat)
    face_edge_mat = np.block([np.expand_dims(faces_flat[:-1],axis=1), np.expand_dims(faces_flat[1:],axis=1)]) 

    #finding the number of edges associated with each face
    fe_mat_inds = np.arange(0,list_len-1,1)
    find_num_edges = fe_mat_inds[(fe_mat_inds==0) + (np.any(face_edge_mat==-1, axis=1))]
    find_num_edges[:][0] = -1
    find_num_edges = find_num_edges.reshape(faces_len,2)
    face_num_edges = find_num_edges[:,1] - find_num_edges[:,0] -1

    #repeating each face index for the number of edges that are associated with it; length equals num_edges * 2
    face_correspond_inds = np.repeat(np.arange(0,faces_len,1), face_num_edges)

    #shape.edges lists the indices of the edge vertices lowest to highest. edges1_reshape lists the indices of the edge vertices highest to lowest
    edges1_reshape = np.hstack((np.expand_dims(shape.edges[:,1], axis=1), np.expand_dims(shape.edges[:,0], axis=1)))

    #For the new_edge_ind_bool: rows correspond with the face_correspond_inds and columns correpond with the edge index; finding the neighboring faces for each edge
    true_face_edge_mat = np.tile(np.expand_dims(face_edge_mat[np.all(face_edge_mat!=-1, axis=1)],axis=1), (1, num_edges,1))
    new_edge_ind_bool0 = np.all(true_face_edge_mat == np.expand_dims(shape.edges, axis=0), axis=2).astype(int) #faces to the LEFT of edges if edges are oriented pointing upwards
    new_edge_ind_bool1 = np.all(true_face_edge_mat == np.expand_dims(edges1_reshape, axis=0), axis=2).astype(int) #faces to the RIGHT of edges if edges are oriented pointing upwards

    #tiling face_correspond_inds so it can be multiplied to the new_edge_ind_bool0 and new_edge_ind_bool1
    new_face_corr_inds = np.tile(np.expand_dims(face_correspond_inds, axis=1), (1,num_edges))

    #getting the face indices and completing the edge-face neighbors
    ef_neighbor0 = np.expand_dims(np.sum(new_face_corr_inds*new_edge_ind_bool0, axis=0), axis=1) #faces to the LEFT
    ef_neighbor1 = np.expand_dims(np.sum(new_face_corr_inds*new_edge_ind_bool1, axis=0), axis=1) #faces to the RIGHT
    ef_neighbor = np.hstack((ef_neighbor0, ef_neighbor1))

    return ef_neighbor

def point_to_edge_distance (point: np.ndarray, vert: np.ndarray, edge_vector: np.ndarray) -> np.ndarray:
    '''
    Calculates the distances between several points and several varying lines.

    n is the total number of distance calculations that are being made. For example, let's say 
    we have points: A, B, C & D, and edges: U, V & W, and we want to calculate the distances between:
    - A & U, A & W, B & U, C & V, C & W, D & U, D & V, and D & W
    n = 8 for this example, and point = [A,A,B,C,C,D,D,D] and edge_vector = [U,W,U,V,W,U,V,W]

    Args:
        point (np.ndarray): the positions of the points [shape = (n, 3)] 
        vert (np.ndarray): positions of the points that lie on each corresponding line [shape = (n, 3)]
        edge_vector (np.ndarray): the vectors that describe each line [shape = (n, 3)]

    Returns:
        np.ndarray: distances [shape = (n,)]
    '''
    edge_unit = edge_vector / np.expand_dims(LA.norm(edge_vector, axis=1), axis=1) #unit vectors of the edges

    dist = LA.norm(((vert - point) - (np.expand_dims(np.sum((vert-point)*edge_unit, axis=1),axis=1) *edge_unit)), axis=1) #distances
    return dist

def point_to_edge_displacement (point: np.ndarray, vert: np.ndarray, edge_vector: np.ndarray) -> np.ndarray:
    '''
    Calculates the displacements between several points and several varying lines.

    n is the total number of displacement calculations that are being made. For example, let's say 
    we have points: A, B, C & D, and edges: U, V & W, and we want to calculate the displacements between:
    - A & U, A & W, B & U, C & V, C & W, D & U, D & V, and D & W
    n = 8 for this example, and point = [A,A,B,C,C,D,D,D] and edge_vector = [U,W,U,V,W,U,V,W]

    Args:
        point (np.ndarray): the positions of the points [shape = (n, 3)] 
        vert (np.ndarray): positions of the points that lie on each corresponding line [shape = (n, 3)]
        edge_vector (np.ndarray): the vectors that describe each line [shape = (n, 3)]

    Returns:
        np.ndarray: displacements [shape = (n, 3)]
    '''
    edge_unit = edge_vector / np.expand_dims(LA.norm(edge_vector, axis=1), axis=1) #unit vectors of the edges

    disp = ((vert - point) - (np.expand_dims(np.sum((vert-point)*edge_unit, axis=1),axis=1) *edge_unit)) #displacements
    return disp

def point_to_face_distance(point: np.ndarray, vert: np.ndarray, face_normal: np.ndarray) -> np.ndarray:
    '''
    Calculates the distances between several points and several varying planes.

    n is the total number of distance calculations that are being made. For example, let's say 
    we have points: A, B, C & D, and faces: P, Q & R, and we want to calculate the distances between:
    - A & P, A & R, B & P, C & Q, C & R, D & P, D & Q, and D & R
    n = 8 for this example, and point = [A,A,B,C,C,D,D,D] and edge_vector = [P,R,P,Q,R,P,Q.R]

    Args:
        point (np.ndarray): the positions of the points [shape = (n, 3)]
        vert (np.ndarray): points that lie on each corresponding plane [shape = (n, 3)]
        face_normal (np.ndarray): the normals that describe each plane [shape = (n, 3)]

    Returns:
        np.ndarray: distances [shape = (n,)]
    '''
    vert_point_vect = vert - point #displacements between the points and relevent vertices
    face_unit = face_normal / np.expand_dims(LA.norm(face_normal, axis=1), axis=1) #unit vectors of the normals of the faces

    dist = np.sum(vert_point_vect*face_unit, axis=1) * (-1) #distances

    return dist

def point_to_face_displacement(point: np.ndarray, vert: np.ndarray, face_normal: np.ndarray) -> np.ndarray:
    '''
    Calculates the displacements between several points and several varying planes.

    n is the total number of displacement calculations that are being made. For example, let's say 
    we have points: A, B, C & D, and faces: P, Q & R, and we want to calculate the displacements between:
    - A & P, A & R, B & P, C & Q, C & R, D & P, D & Q, and D & R
    n = 8 for this example, and point = [A,A,B,C,C,D,D,D] and edge_vector = [P,R,P,Q,R,P,Q.R]

    Args:
        point (np.ndarray): the positions of the points [shape = (n, 3)]
        vert (np.ndarray): points that lie on each corresponding plane [shape = (n, 3)]
        face_normal (np.ndarray): the normals that describe each plane [shape = (n, 3)]

    Returns:
        np.ndarray: displacements [shape = (n, 3)]
    '''
    vert_point_vect = vert - point #displacements between the points and relevent vertices
    face_units = face_normal / np.expand_dims(LA.norm(face_normal, axis=1), axis=1) #unit vectors of the normals of the faces

    disp = np.expand_dims(np.sum(vert_point_vect*face_units, axis=1), axis=1) * face_units #*(-1) #displacements

    return disp

def get_vert_zones (shape):
    '''
    Gets the constraints and bounds needed to partition the volume surrounding a polyhedron into zones 
    where the shortest distance from any point that is within a vertex zone is the distance between the 
    point and the corresponding vertex.

    Args:
        shape (Polyhedron): the polyhedron that is being looked at (can be convex or concave)

    Returns:
        dict: "constraint": np.ndarray [shape = (n_verts, n_edges, 3)], "bounds": np.ndarray [shape = (n_verts, n_edges)]
    '''
    #For a generalized shape, we cannot assume that every vertex has the same number of edges connected to it 
    #(EX:vertices in a cube have 3 connected edges each, and for an icosahedron, vertices have 5 conncected edges).
    #This would result in a ragged list for the constraint and bounds, which is not ideal.



    #v--- This `for`` loop is used to build and fill that ragged list with zeros, so that it makes an easy to work with array. ---v
    for v_i in range(len(shape.vertices)):
        pos_adj_edges = shape.edge_vectors[shape.edges[:,0] == v_i] #edges that point away from v_i
        neg_adj_edges = (-1)*shape.edge_vectors[shape.edges[:,1] == v_i] #edges that point towards v_i, so have to multiply by -1
        adjacent_edges = np.append(pos_adj_edges, neg_adj_edges, axis=0)

        if v_i == 0: #initial vertex, start of building the constraint
            vert_constraint = np.asarray([adjacent_edges])
        
        else:
            difference = len(adjacent_edges) - vert_constraint.shape[1] 
            #^---difference between the # of edges v_i has and the max # of edges from a previous vertex

            if difference < 0: #adjacent_edges needs to be filled with zeros to match the length of axis=1 for vert_constraint
                adjacent_edges = np.append(adjacent_edges, np.zeros((abs(difference), 3)), axis=0)

            if difference > 0: #vert_constraint needs to be filled with zeros to match the # of edges for v_i
                vert_constraint = np.append(vert_constraint, np.zeros((len(vert_constraint), difference, 3)), axis=1)

            vert_constraint = np.append(vert_constraint, np.asarray([adjacent_edges]), axis=0)

    vert_bounds = np.sum(vert_constraint * np.expand_dims(shape.vertices, axis=1), axis=2)

    return {"constraint":vert_constraint, "bounds":vert_bounds}

def get_edge_zones (shape):
    '''
    Gets the constraints and bounds needed to partition the volume surrounding a polyhedron into zones 
    where the shortest distance from any point that is within an edge zone is the distance between the 
    point and the corresponding edge.

    Args:
        shape (Polyhedron): the polyhedron that is being looked at (can be convex or concave)

    Returns:
        dict: "constraint": np.ndarray [shape = (n_edges, 4, 3)], "bounds": np.ndarray [shape = (n_edges, 4)]
    '''
    #Set up
    edge_constraint = np.zeros((shape.num_edges, 4, 3))
    edge_bounds = np.zeros((shape.num_edges, 4))

    #Calculating the normals of the plane boundaries
    edge_constraint[:,0] = shape.edge_vectors
    edge_constraint[:,1] = -1*shape.edge_vectors
    edge_constraint[:,2] = np.cross(shape.edge_vectors, shape.normals[shape.edge_face_neighbors[:,1]])
    edge_constraint[:,3] = -1*np.cross(shape.edge_vectors, shape.normals[shape.edge_face_neighbors[:,0]])
    #Constraint shape = (n_edges, 4, 3)

    #Bounds [shape = (n_edges, 4)]
    edge_verts = np.zeros((shape.num_edges, 2, 3))
    edge_verts[:,0] = shape.vertices[shape.edges[:,0]]
    edge_verts[:,1] = shape.vertices[shape.edges[:,1]]

    edge_bounds[:,0] = np.sum(edge_constraint[:,0] *(edge_verts[:,1]), axis=1)
    edge_bounds[:,1] = np.sum(edge_constraint[:,1] *(edge_verts[:,0]), axis=1)
    edge_bounds[:,2] = np.sum(edge_constraint[:,2] *(edge_verts[:,0]), axis=1)
    edge_bounds[:,3] = np.sum(edge_constraint[:,3] *(edge_verts[:,0]), axis=1)

    return {"constraint":edge_constraint, "bounds":edge_bounds}

def get_face_zones (shape):
    '''
    Gets the constraints and bounds needed to partition the volume surrounding a polyhedron into zones 
    where the shortest distance from any point that is within a triangulated face zone is the distance between the 
    point and the corresponding triangulated face.

    Args:
        shape (Polyhedron): the polyhedron that is being looked at (can be convex or concave)

    Returns:
        dict: "constraint": np.ndarray [shape = (n_tri_faces, 3, 3)], "bounds": np.ndarray [shape = (n_tri_faces, 3)],
            "face_points": np.ndarray [shape= (n_tri_faces, 3)], "normals": np.ndarray [shape=(n_tri_faces, 3)]
    '''
    #----- Triangulating the surface of the shape -----
    try:
        #checking to see if faces are already triangulated
        something = np.asarray(shape.faces).reshape(shape.num_faces,3)

    except:
        #triangulating faces
        triangle_verts = []
        for triangle in shape._surface_triangulation():
            triangle_verts.append(list(triangle))

        triangle_verts = np.asarray(triangle_verts) #vertices of the triangulated faces
        tri_edges = np.append(triangle_verts[:,1:], triangle_verts[:,0].reshape(len(triangle_verts),1,3), axis=1) - triangle_verts #edges point counterclockwise
        tri_face_normals = np.cross(tri_edges[:,0], tri_edges[:,1]) #normals of the triangulated faces

    else:
        triangle_verts = shape.vertices[shape.faces] #vertices of the triangulated faces
        tri_edges = np.append(triangle_verts[:,1:], triangle_verts[:,0].reshape(len(triangle_verts),1,3), axis=1) - triangle_verts #edges point counterclockwise
        tri_face_normals = shape.normals #normals of the triangulated faces


    face_constraint = np.cross(tri_edges, np.expand_dims(tri_face_normals, axis=1)) #shape = (n_tri_faces, 3, 3)
    face_bounds = np.sum(face_constraint*triangle_verts, axis=2) #shape = (n_tri_faces, 3)

    face_one_vertex = triangle_verts[:,0] #a point (vertex) that lies on each of the planes of the triangulated faces

    return {"constraint":face_constraint, "bounds":face_bounds, "face_points":face_one_vertex, "normals": tri_face_normals}

def get_edge_normals(shape) -> np.ndarray:
    '''
    Gets the analogous normals of the edges of the polyhedron. The normals point outwards from the polyhedron 
    and are used to determine whether an edge zone is outside or inside the polyhedron.

    Args:
        shape (Polyhedron): the polyhedron that is being looked at (can be convex or concave)

    Returns:
        np.ndarray: analogous edge normals [shape = (n_edges, 3)]
    '''
    face_unit = shape.normals / np.expand_dims(LA.norm(shape.normals, axis=1), axis=1) #unit vectors of the face normals
    face_unit1 = face_unit[shape.edge_face_neighbors[:,0]] 
    face_unit2 = face_unit[shape.edge_face_neighbors[:,1]] 

    edge_normals = face_unit1 + face_unit2 #sum of the adjacent face normals for each edge

    #returning the unit vectors of the edge normals
    return edge_normals / np.expand_dims(LA.norm(edge_normals, axis=1), axis=1)

def get_vert_normals(shape) -> np.ndarray:
    '''
    Gets the analogous normals of the vertices of the polyhedron. The normals point outwards from the polyhedron 
    and are used to determine whether a vertex zone is outside or inside the polyhedron.

    Args:
        shape (Polyhedron): the polyhedron that is being looked at (can be convex or concave)

    Returns:
        np.ndarray: analogous vertex normals [shape = (n_verts, 3)]
    '''
    n_edges = len(shape.edge_normals)
    n_verts = np.max(shape.edges) +1

    #Tiling for set up
    nverts_edge_vert0 = np.tile(shape.edges[:,0], (n_verts, 1))
    nverts_edge_vert1 = np.tile(shape.edges[:,1], (n_verts, 1))
    vert_inds = np.arange(0, n_verts, 1).reshape((n_verts, 1))
    nverts_tile_edges = np.tile(shape.edge_normals, (n_verts, 1)).reshape((n_verts, n_edges, 3))

    #Creating the bools needed to get the edges that correspond to each vertex
    evbool0 = (np.expand_dims(nverts_edge_vert0 == vert_inds, axis=2)).astype(int)
    evbool1 = (np.expand_dims(nverts_edge_vert1 == vert_inds, axis=2)).astype(int)

    #Applying the bools to find the corresponding edges
    vert_edges = nverts_tile_edges * evbool0 + nverts_tile_edges * evbool1

    vert_normals = np.sum(vert_edges, axis=1) #sum of the adjacent edge normals for each vertex

    #returning the unit vectors of the vertex normals
    return vert_normals / np.expand_dims(LA.norm(vert_normals, axis=1), axis=1)


def get_weighted_edge_normals(shape) -> np.ndarray:
    '''
    Gets the weighted normals of the edges of the polyhedron. The normals point outwards from the polyhedron.

    Args:
        shape (Polyhedron): the polyhedron that is being looked at (can be convex or concave)

    Returns:
        np.ndarray: analogous edge normals [shape = (n_edges, 3)]
    '''
    face_1 = shape.normals[shape.edge_face_neighbors[:,0]] 
    face_2 = shape.normals[shape.edge_face_neighbors[:,1]] 

    edge_normals = face_1 + face_2 #sum of the adjacent face normals for each edge

    return edge_normals 

def get_weighted_vert_normals(shape) -> np.ndarray:
    '''
    Gets the weighted normals of the vertices of the polyhedron. The normals point outwards from the polyhedron.

    Args:
        shape (Polyhedron): the polyhedron that is being looked at (can be convex or concave)

    Returns:
        np.ndarray: analogous vertex normals [shape = (n_verts, 3)]
    '''
    n_edges = len(shape.edge_normals)
    n_verts = np.max(shape.edges) +1

    #Tiling for set up
    nverts_edge_vert0 = np.tile(shape.edges[:,0], (n_verts, 1))
    nverts_edge_vert1 = np.tile(shape.edges[:,1], (n_verts, 1))
    vert_inds = np.arange(0, n_verts, 1).reshape((n_verts, 1))
    nverts_tile_edges = np.tile(shape.weighted_edge_normals, (n_verts, 1)).reshape((n_verts, n_edges, 3))

    #Creating the bools needed to get the edges that correspond to each vertex
    evbool0 = (np.expand_dims(nverts_edge_vert0 == vert_inds, axis=2)).astype(int)
    evbool1 = (np.expand_dims(nverts_edge_vert1 == vert_inds, axis=2)).astype(int)

    #Applying the bools to find the corresponding edges
    vert_edges = nverts_tile_edges * evbool0 + nverts_tile_edges * evbool1

    vert_normals = np.sum(vert_edges, axis=1) #sum of the adjacent weighted edge normals for each vertex

    return vert_normals



def shortest_distance_to_surface (
        shp,
        points: np.ndarray,
        translation_vector: np.ndarray,
) -> np.ndarray:
    '''
    Solves for the shortest distance between points and the surface of a polyhedron. 
    If the point lies inside the polyhedron, the distance is negative.

    This function calculates the shortest distance by partitioning the space around 
    a polyhedron into zones: vertex, edge, and face. Determining the zone(s) a 
    point lies in, determines the distance calculation(s) done. For a vertex zone,
    the distance is calculated between a point and the vertex. For an edge zone, the 
    distance is calculated between a point and the edge. For a face zone, the distance
    is calculated between a point and the face. Zones are allowed to overlap, and points
    can be in more than one zone. By taking the minimum of all the calculated distances,
    the shortest distances are found.

    Args:
        points (list or np.ndarray): positions of the points [shape = (n_points, 3)]
        translation_vector (list or np.ndarray): translation vector of the polyhedron [shape = (3,)]

    Returns:
        np.ndarray: shortest distances [shape = (n_points,)]
    '''
    points = np.asarray(points)
    translation_vector = np.asarray(translation_vector)

    if translation_vector.shape[0]!=3 or len(translation_vector.shape)>1:
        raise ValueError(f"Expected the shape of the polygon's position to be (3,), instead it got {translation_vector.shape}")

    if points.shape == (3,):
        points = points.reshape(1, 3)

    atol = 1e-8
    n_points = len(points) #number of inputted points
    n_verts = len(shp.vertices) #number of vertices = number of vertex zones
    n_edges = len(shp.edges) #number of edges = number of edge zones
    n_tri_faces = len(shp.face_zones["bounds"]) #number of triangulated faces = number of triangulated face zones

    #arrays consisting of 1 or -1, and used to determine if a point is inside the polyhedron
    vert_inside_mult = np.diag(np.all((shp.vertex_zones["constraint"] @ np.transpose(shp.vertex_normals+shp.vertices)) <= np.expand_dims(shp.vertex_zones["bounds"], axis=2), axis=1)).astype(int)*2 -1
    edge_inside_mult = np.diag(np.all((shp.edge_zones["constraint"] @ np.transpose(shp.edge_normals+0.5*(shp.vertices[shp.edges[:,0]]+shp.vertices[shp.edges[:,1]]))) <= np.expand_dims(shp.edge_zones["bounds"], axis=2), axis=1)).astype(int)*2 -1


    #Updating bounds with the position of the polyhedron
    vert_bounds = shp.vertex_zones["bounds"] + (shp.vertex_zones["constraint"] @ translation_vector)
    edge_bounds = shp.edge_zones["bounds"] + (shp.edge_zones["constraint"] @ translation_vector)
    face_bounds = shp.face_zones["bounds"] + (shp.face_zones["constraint"] @ translation_vector)

    points_trans = np.transpose(points) #Have to take the transpose so that 'constraint @ points_trans' returns the right shape and values
    max_value = 3*np.max(LA.norm(points - (translation_vector+shp.vertices[0]), axis=1)) #Placeholder value, it is large so that it is not chosen when taking the min of the distances

    #Calculating the distances

    # Solving for the distances between the points and any relevant vertices
    vert_dist=LA.norm(np.repeat(np.expand_dims(points, axis=1),n_verts, axis=1) - np.expand_dims(shp.vertices + translation_vector, axis=0), axis=2)*np.expand_dims(vert_inside_mult, axis=0) #Distances between two points

    #Taking the minimum of the distances for each point
    vert_dist_arg = np.expand_dims(np.argmin(abs(vert_dist), axis=1), axis=1)
    min_dist_arr = np.take_along_axis(vert_dist, vert_dist_arg, axis=1)

    #Solving for the distances between the points and any relevant edges
    edge_bool = np.all((shp.edge_zones["constraint"] @ points_trans) <= (np.expand_dims(edge_bounds, axis=2)+atol), axis=1) #<--- shape = (n_edges, n_points)
    # edge_bool = edge_bool + np.allclose((shp.edge_zones["constraint"] @ points_trans), np.expand_dims(edge_bounds, axis=2), atol=1e-6)
    if np.any(edge_bool):

        #v--- shape = (number of True in edge_bool,) ---v
        edge_used = np.transpose(np.tile(np.arange(0,n_edges,1), (n_points,1)))[edge_bool] #Contains the indices of the edges that hold True for edge_bool
        e_points_used = np.tile(np.arange(0,n_points,1), (n_edges,1))[edge_bool] #Contains the indices of the points that hold True for edge_bool

        vert_on_edge = shp.vertices[shp.edges[edge_used][:,0]] + translation_vector #Vertices that lie on the needed edges

        #Calculating the distances
        edge_dist = np.ones((n_edges,n_points))*max_value
        edge_dist[edge_bool]=point_to_edge_distance(points[e_points_used], vert_on_edge, shp.edge_vectors[edge_used])*edge_inside_mult[edge_used] #Distances between a point and a line
        edge_dist = np.transpose(edge_dist) #<--- shape = (n_points, n_edges)

        #Taking the minimum of the distances for each point
        edge_dist_arg = np.expand_dims(np.argmin(abs(edge_dist), axis=1), axis=1)
        edge_dist = np.take_along_axis(edge_dist, edge_dist_arg, axis=1)

        min_dist_arr = np.concatenate((min_dist_arr, edge_dist), axis=1)

    #Solving for the distances between the points and any relevant faces
    face_bool = np.all((shp.face_zones["constraint"] @ points_trans) <= (np.expand_dims(face_bounds, axis=2)+atol), axis=1) #<--- shape = (n_tri_faces, n_points)
    # face_bool = face_bool + np.allclose((shp.face_zones["constraint"] @ points_trans), np.expand_dims(face_bounds, axis=2), atol=1e-6)
    if np.any(face_bool):

        #v--- shape = (number of True in face_bool,) ---v
        face_used = np.transpose(np.tile(np.arange(0,n_tri_faces,1), (n_points,1)))[face_bool] #Contains the indices of the triangulated faces that hold True for face_bool
        f_points_used = np.tile(np.arange(0,n_points,1), (n_tri_faces,1))[face_bool] #Contains the indices of the points that hold True for face_bool

        vert_on_face = (shp.face_zones["face_points"][face_used]) + translation_vector #Vertices that lie on the needed faces

        #Calculating the distances
        face_dist = np.ones((n_tri_faces,n_points))*max_value
        face_dist[face_bool]=point_to_face_distance(points[f_points_used], vert_on_face, shp.face_zones["normals"][face_used]) #Distances between a point and a plane
        face_dist = np.transpose(face_dist) #<--- shape = (n_points, n_tri_faces)

        #Taking the minimum of the distances for each point
        face_dist_arg = np.expand_dims(np.argmin(abs(face_dist), axis=1), axis=1)
        face_dist = np.take_along_axis(face_dist, face_dist_arg, axis=1)

        min_dist_arr = np.concatenate((min_dist_arr, face_dist), axis=1)

    min_dist_arg = np.expand_dims(np.argmin(abs(min_dist_arr), axis=1), axis=1) #determining the distances that are the shortest
    true_min_dist = np.take_along_axis(min_dist_arr, min_dist_arg, axis=1).flatten()

    return true_min_dist

def shortest_displacement_to_surface (
        shp,
        points: np.ndarray,
        translation_vector: np.ndarray
) -> np.ndarray:
    '''
    Solves for the shortest displacement between points and the surface of a polyhedron.

    This function calculates the shortest displacement by partitioning the space around 
    a polyhedron into zones: vertex, edge, and face. Determining the zone(s) a 
    point lies in, determines the displacement calculation(s) done. For a vertex zone,
    the displacement is calculated between a point and the vertex. For an edge zone, the 
    displacement is calculated between a point and the edge. For a face zone, the 
    displacement is calculated between a point and the face. Zones are allowed to overlap, 
    and points can be in more than one zone. By taking the minimum of all the distances of
    the calculated displacements, the shortest displacements are found.

    Args:
        points (list or np.ndarray): positions of the points [shape = (n_points, 3)]
        translation_vector (list or np.ndarray): translation vector of the polyhedron [shape = (3,)]

    Returns:
        np.ndarray: shortest displacements [shape = (n_points, 3)]
    '''
    points = np.asarray(points)
    translation_vector = np.asarray(translation_vector)

    if translation_vector.shape[0]!=3 or len(translation_vector.shape)>1:
        raise ValueError(f"Expected the shape of the polygon's position to be (3,), instead it got {translation_vector.shape}")

    if points.shape == (3,):
        points = points.reshape(1, 3)

    atol = 1e-8
    n_points = len(points) #number of inputted points
    n_verts = len(shp.vertices) #number of vertices = number of vertex zones
    n_edges = len(shp.edges) #number of edges = number of edge zones
    n_tri_faces = len(shp.face_zones["bounds"]) #number of triangulated faces = number of triangulated face zones

    #Updating bounds with the position of the polyhedron
    vert_bounds = shp.vertex_zones["bounds"] + (shp.vertex_zones["constraint"] @ translation_vector)
    edge_bounds = shp.edge_zones["bounds"] + (shp.edge_zones["constraint"] @ translation_vector)
    face_bounds = shp.face_zones["bounds"] + (shp.face_zones["constraint"] @ translation_vector)

    coord_trans = np.transpose(points) #Have to take the transpose so that 'constraint @ coord_trans' returns the right shape and values
    max_value = 3*np.max(LA.norm(points - (translation_vector+shp.vertices[0]), axis=1)) #Placeholder value, it is large so that it is not chosen when taking the min of the distances

    #Calculating the displacements

    #Solving for the displacements between the points and any relevant vertices
    vert_disp=(-1*np.repeat(np.expand_dims(points, axis=1),n_verts, axis=1)) + np.expand_dims(shp.vertices + translation_vector, axis=0) #Displacements between two point
    
    #Taking the minimum of the displacements for each point
    vert_disp_min = np.expand_dims(np.argmin( LA.norm(vert_disp, axis=2), axis=1), axis=(1,2))
    min_disp_arr = np.take_along_axis(vert_disp, vert_disp_min, axis=1)

    #Solving for the displacements between the points and any relevant edges
    edge_bool = np.all((shp.edge_zones["constraint"] @ coord_trans) <= (np.expand_dims(edge_bounds, axis=2)+atol), axis=1) #<--- shape = (n_edges, n_points)
    if np.any(edge_bool):

        #v--- shape = (number of True in edge_bool,) ---v
        edge_used = np.transpose(np.tile(np.arange(0,n_edges,1), (n_points,1)))[edge_bool] #Contains the indices of the edges that hold True for edge_bool
        ecoords_used = np.tile(np.arange(0,n_points,1), (n_edges,1))[edge_bool] #Contains the indices of the points that hold True for edge_bool

        vert_on_edge = shp.vertices[shp.edges[edge_used][:,0]] + translation_vector #Vertices that lie on the needed edges

        #Calculating the displacements
        edge_disp = np.ones((n_edges,n_points,3))*max_value
        edge_disp[edge_bool]=point_to_edge_displacement(points[ecoords_used], vert_on_edge, shp.edge_vectors[edge_used]) #Displacements between a point and a line
        edge_disp = np.transpose(edge_disp, (1, 0, 2)) #<--- shape = (n_points, n_edges, 3)

        #Taking the minimum of the displacements for each point
        edge_disp_arg = np.expand_dims(np.argmin( LA.norm(edge_disp, axis=2), axis=1), axis=(1,2))
        edge_disp = np.take_along_axis(edge_disp, edge_disp_arg, axis=1)

        min_disp_arr = np.concatenate((min_disp_arr, edge_disp), axis=1)

    #Solving for the displacements between the points and any relevant faces
    face_bool = np.all((shp.face_zones["constraint"] @ coord_trans) <= (np.expand_dims(face_bounds, axis=2)+atol), axis=1) #<--- shape = (n_tri_faces, n_points)
    if np.any(face_bool):

        #v--- shape = (number of True in face_bool,) ---v
        face_used = np.transpose(np.tile(np.arange(0,n_tri_faces,1), (n_points,1)))[face_bool] #Contains the indices of the triangulated faces that hold True for face_bool
        fcoords_used = np.tile(np.arange(0,n_points,1), (n_tri_faces,1))[face_bool] #Contains the indices of the points that hold True for face_bool
        
        vert_on_face = (shp.face_zones["face_points"][face_used]) + translation_vector #Vertices that lie on the needed faces

        #Calculating the displacements
        face_disp = np.ones((n_tri_faces,n_points,3))*max_value
        face_disp[face_bool]=point_to_face_displacement(points[fcoords_used], vert_on_face, shp.face_zones["normals"][face_used]) #Displacements between a point and a plane
        face_disp = np.transpose(face_disp, (1, 0, 2)) #<--- shape = (n_points, n_tri_faces, 3)

        #Taking the minimum of the displacements for each point
        face_disp_arg = np.expand_dims(np.argmin(LA.norm(face_disp, axis=2), axis=1), axis=(1,2))
        face_disp = np.take_along_axis(face_disp, face_disp_arg, axis=1)

        min_disp_arr = np.concatenate((min_disp_arr, face_disp), axis=1)

    disp_arr_bool = np.expand_dims(np.argmin( (LA.norm(min_disp_arr, axis=2)), axis=1), axis=(1,2)) #determining the displacements that are shortest
    true_min_disp = np.squeeze(np.take_along_axis(min_disp_arr, disp_arr_bool, axis=1), axis=1)

    return true_min_disp

def spheropolyhedron_shortest_displacement_to_surface (
        shp,
        radius,
        points: np.ndarray,
        translation_vector: np.ndarray
) -> np.ndarray:
    '''
    Solves for the shortest displacement between points and the surface of a polyhedron.

    This function calculates the shortest displacement by partitioning the space around 
    a polyhedron into zones: vertex, edge, and face. Determining the zone(s) a 
    point lies in, determines the displacement calculation(s) done. For a vertex zone,
    the displacement is calculated between a point and the vertex. For an edge zone, the 
    displacement is calculated between a point and the edge. For a face zone, the 
    displacement is calculated between a point and the face. Zones are allowed to overlap, 
    and points can be in more than one zone. By taking the minimum of all the distances of
    the calculated displacements, the shortest displacements are found.

    Args:
        points (list or np.ndarray): positions of the points [shape = (n_points, 3)]
        translation_vector (list or np.ndarray): translation vector of the polyhedron [shape = (3,)]

    Returns:
        np.ndarray: shortest displacements [shape = (n_points, 3)]
    '''
    points = np.asarray(points)
    translation_vector = np.asarray(translation_vector)

    if translation_vector.shape[0]!=3 or len(translation_vector.shape)>1:
        raise ValueError(f"Expected the shape of the polygon's position to be (3,), instead it got {translation_vector.shape}")

    if points.shape == (3,):
        points = points.reshape(1, 3)

    atol = 1e-8
    n_points = len(points) #number of inputted points
    n_verts = len(shp.vertices) #number of vertices = number of vertex zones
    n_edges = len(shp.edges) #number of edges = number of edge zones
    n_tri_faces = len(shp.face_zones["bounds"]) #number of triangulated faces = number of triangulated face zones

    #Updating bounds with the position of the polyhedron
    vert_bounds = shp.vertex_zones["bounds"] + (shp.vertex_zones["constraint"] @ translation_vector)
    edge_bounds = shp.edge_zones["bounds"] + (shp.edge_zones["constraint"] @ translation_vector)
    face_bounds = shp.face_zones["bounds"] + (shp.face_zones["constraint"] @ translation_vector)

    coord_trans = np.transpose(points) #Have to take the transpose so that 'constraint @ coord_trans' returns the right shape and values
    max_value = 3*np.max(LA.norm(points - (translation_vector+shp.vertices[0]), axis=1)) #Placeholder value, it is large so that it is not chosen when taking the min of the distances
    min_disp_arr = np.ones((n_points,1, 3))*max_value #Initial min_disp_arr

    #Calculating the displacements

    #Solving for the displacements between the points and any relevant vertices
    vert_bool = np.all((shp.vertex_zones["constraint"] @ coord_trans) <= np.expand_dims(vert_bounds, axis=2), axis=1) #<--- shape = (n_verts, n_points)
    if np.any(vert_bool):

        #v--- shape = (number of True in vert_bool,) ---v
        vert_used = np.transpose(np.tile(np.arange(0,n_verts,1), (n_points,1)))[vert_bool] #Contains the indices of the vertices that hold True for vert_bool
        vcoords_used = np.tile(np.arange(0,n_points,1), (n_verts,1))[vert_bool] #Contains the indices of the points that hold True for vert_bool
        
        #Calculating the displacements
        vert_disp = np.ones((n_verts,n_points,3))*max_value
        vert_disp[vert_bool]=(shp.vertices[vert_used] + translation_vector) - points[vcoords_used] #Displacements between two points
        vert_disp = np.transpose(vert_disp, (1,0,2)) #<--- shape = (n_points, n_verts, 3)

        #TODO: subtract radius*unit_displacement -- unless displacement is zero, then subtract radius*vert_normal
        vert_zero_disp_bool = np.all(vert_disp == 0, axis=2)
        vert_disp[vert_zero_disp_bool] = vert_disp[vert_zero_disp_bool] + radius*(np.repeat(np.expand_dims(shp.vertex_normals,axis=0),n_points,axis=0)[vert_zero_disp_bool])
        vert_disp[np.invert(vert_zero_disp_bool)] = vert_disp[np.invert(vert_zero_disp_bool)] - radius*(vert_disp[np.invert(vert_zero_disp_bool)]/np.expand_dims(np.linalg.norm(vert_disp[np.invert(vert_zero_disp_bool)],axis=1),axis=1))
        
        #Taking the minimum of the displacements for each point
        vert_disp_min = np.expand_dims(np.argmin( LA.norm(vert_disp, axis=2), axis=1), axis=(1,2))
        vert_disp = np.take_along_axis(vert_disp, vert_disp_min, axis=1)

        min_disp_arr = np.concatenate((min_disp_arr, vert_disp), axis=1)

    #Solving for the displacements between the points and any relevant edges
    edge_bool = np.all((shp.edge_zones["constraint"] @ coord_trans) <= (np.expand_dims(edge_bounds, axis=2)+atol), axis=1) #<--- shape = (n_edges, n_points)
    if np.any(edge_bool):

        #v--- shape = (number of True in edge_bool,) ---v
        edge_used = np.transpose(np.tile(np.arange(0,n_edges,1), (n_points,1)))[edge_bool] #Contains the indices of the edges that hold True for edge_bool
        ecoords_used = np.tile(np.arange(0,n_points,1), (n_edges,1))[edge_bool] #Contains the indices of the points that hold True for edge_bool

        vert_on_edge = shp.vertices[shp.edges[edge_used][:,0]] + translation_vector #Vertices that lie on the needed edges

        #Calculating the displacements
        edge_disp = np.ones((n_edges,n_points,3))*max_value
        edge_disp[edge_bool]=point_to_edge_displacement(points[ecoords_used], vert_on_edge, shp.edge_vectors[edge_used]) #Displacements between a point and a line
        edge_disp = np.transpose(edge_disp, (1, 0, 2)) #<--- shape = (n_points, n_edges, 3)

        #TODO: subtract radius*unit_displacement -- unless displacement is zero, then subtract radius*vert_normal
        edge_zero_disp_bool = np.all(edge_disp == 0, axis=2)
        edge_disp[edge_zero_disp_bool] = edge_disp[edge_zero_disp_bool] + radius*(np.repeat(np.expand_dims(shp.edge_normals,axis=0),n_points,axis=0)[edge_zero_disp_bool])
        edge_disp[np.invert(edge_zero_disp_bool)] = edge_disp[np.invert(edge_zero_disp_bool)] - radius*(edge_disp[np.invert(edge_zero_disp_bool)]/np.expand_dims(np.linalg.norm(edge_disp[np.invert(edge_zero_disp_bool)],axis=1),axis=1))

        #Taking the minimum of the displacements for each point
        edge_disp_arg = np.expand_dims(np.argmin( LA.norm(edge_disp, axis=2), axis=1), axis=(1,2))
        edge_disp = np.take_along_axis(edge_disp, edge_disp_arg, axis=1)

        min_disp_arr = np.concatenate((min_disp_arr, edge_disp), axis=1)

    #Solving for the displacements between the points and any relevant faces
    face_bool = np.all((shp.face_zones["constraint"] @ coord_trans) <= (np.expand_dims(face_bounds, axis=2)+atol), axis=1) #<--- shape = (n_tri_faces, n_points)
    if np.any(face_bool):

        #v--- shape = (number of True in face_bool,) ---v
        face_used = np.transpose(np.tile(np.arange(0,n_tri_faces,1), (n_points,1)))[face_bool] #Contains the indices of the triangulated faces that hold True for face_bool
        fcoords_used = np.tile(np.arange(0,n_points,1), (n_tri_faces,1))[face_bool] #Contains the indices of the points that hold True for face_bool
        
        vert_on_face = (shp.face_zones["face_points"][face_used]) + translation_vector #Vertices that lie on the needed faces

        #Calculating the displacements
        face_disp = np.ones((n_tri_faces,n_points,3))*max_value
        face_disp[face_bool]=point_to_face_displacement(points[fcoords_used], vert_on_face, shp.face_zones["normals"][face_used]) #Displacements between a point and a plane

        #TODO: subtract radius*unit_displacement -- unless displacement is zero, then subtract radius*vert_normal
        #TODO: if point is inside, add radius*unit_displacement instead
        point_inside = (-1)*np.ones((n_tri_faces, n_points))
        point_inside[face_bool] = (point_to_face_distance(points[fcoords_used], vert_on_face, shp.face_zones["normals"][face_used]) < 0).astype(int)*2 -1 #(+1) outside, (-1) inside

        face_zero_disp_bool = np.all(face_disp == 0, axis=2)
        face_disp[face_zero_disp_bool] = face_disp[face_zero_disp_bool] + radius*(np.repeat(np.expand_dims((shp.face_zones["normals"]/np.expand_dims(np.linalg.norm(shp.face_zones["normals"],axis=1),axis=1)),axis=1),n_points,axis=1)[face_zero_disp_bool])
        face_disp[np.invert(face_zero_disp_bool)] = face_disp[np.invert(face_zero_disp_bool)] + radius*np.expand_dims(point_inside[np.invert(face_zero_disp_bool)],axis=1)*(face_disp[np.invert(face_zero_disp_bool)]/np.expand_dims(np.linalg.norm(face_disp[np.invert(face_zero_disp_bool)],axis=1),axis=1))

        face_disp = np.transpose(face_disp, (1, 0, 2)) #<--- shape = (n_points, n_tri_faces, 3)
        #Taking the minimum of the displacements for each point
        face_disp_arg = np.expand_dims(np.argmin(LA.norm(face_disp, axis=2), axis=1), axis=(1,2))
        face_disp = np.take_along_axis(face_disp, face_disp_arg, axis=1)

        min_disp_arr = np.concatenate((min_disp_arr, face_disp), axis=1)

    disp_arr_bool = np.expand_dims(np.argmin( (LA.norm(min_disp_arr, axis=2)), axis=1), axis=(1,2)) #determining the displacements that are shortest
    true_min_disp = np.squeeze(np.take_along_axis(min_disp_arr, disp_arr_bool, axis=1), axis=1)

    return true_min_disp
