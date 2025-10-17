import numpy as np
import numpy.linalg as LA

# --- "Hidden" Functions ---
#good?
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

#good?
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

#good?
def point_to_face_distance(point: np.ndarray, vert: np.ndarray, face_normal: np.ndarray) -> np.ndarray:
    '''
    Calculates the distances between a single point and the plane of the polygon.

    Args:
        point (np.ndarray): the positions of the points [shape=(n_points, 3)]
        vert (np.ndarray): a point that lies on the plane of the polygon [shape=(3,)]
        face_normal (np.ndarray): the normal that describes the plane of the polygon [shape=(3,)]

    Returns:
        np.ndarray: distances [shape = (n_points,)]
    '''
    
    vert_point_vect = -1*vert + point
    face_unit = face_normal / LA.norm(face_normal) #unit vector of the normal of the polygon
    dist = abs(vert_point_vect@np.transpose(face_unit))

    return dist

#good?
def point_to_face_displacement(point: np.ndarray, vert: np.ndarray, face_normal: np.ndarray) -> np.ndarray:
    '''
    Calculates the displacements between a single point and the plane of the polygon.

    Args:
        point (np.ndarray): the positions of the points (shape=(n_points, 3))
        vert (np.ndarray): a point that lies on the plane of the polygon (shape=(3,))
        face_normal (np.ndarray): the normal that describes the plane of the polygon (shape=(3,))

    Returns:
        np.ndarray: displacements (n_points, 3)
    '''
    vert_point_vect = -1*vert + point
    face_unit = face_normal / LA.norm(face_normal) #unit vector of the normal of the polygon
    disp = np.expand_dims(np.sum(vert_point_vect*face_unit, axis=1), axis=1) * face_unit *(-1)

    return disp

#good input
def get_vert_zones (shape):
    '''
    Gets the constraints and bounds needed to partition the volume surrounding a polygon into zones 
    where the shortest distance from any point that is within a vertex zone is the distance between the 
    point and the corresponding vertex.

    Args:
        #will be just `self` once added into coxeter

    Returns:
        dict: "constraint": np.ndarray [shape = (n_verts, 2, 3)], "bounds": np.ndarray [shape = (n_verts, 2)]
    '''
    vert_constraint = np.append( 
        np.expand_dims(shape.edge_vectors,axis=1), 
        -1*np.expand_dims(np.append(np.expand_dims(shape.edge_vectors[-1],axis=0), shape.edge_vectors[:-1], axis=0), axis=1), 
        axis=1)
    
    vert_bounds = np.sum(vert_constraint * np.expand_dims(shape.vertices, axis=1), axis=2)

    return {"constraint": vert_constraint, "bounds":vert_bounds}

#good input
def get_edge_zones (shape):
    '''
    Gets the constraints and bounds needed to partition the volume surrounding a polygon into zones 
    where the shortest distance from any point that is within an edge zone is the distance between the 
    point and the corresponding edge.

    Args:
        #will be just `self` once added into coxeter

    Returns:
        dict: "constraint": np.ndarray [shape = (n_edges, 3, 3)], "bounds": np.ndarray [shape = (n_edges, 3)]
    '''
    #vectors that are 90 degrees from the edges and point inwards
    edges_90 = -1*np.expand_dims(np.cross(shape.edge_vectors, shape.normal), axis=1) 

    #Calculating the constraint [shape = (n_edges, 3, 3)]
    edge_constraint = np.append( -1*np.expand_dims(shape.edge_vectors, axis=1) , np.expand_dims(shape.edge_vectors, axis=1), axis=1 )
    edge_constraint = np.append( edge_constraint, edges_90 , axis=1)

    #Bounds [shape = (n_edges, 3)]
    edge_bounds = np.zeros((shape.num_vertices, 3))
    edge_bounds[:,0] = np.sum(edge_constraint[:,0] *(shape.vertices), axis=1)
    edge_bounds[:,1] = np.sum(edge_constraint[:,1] *(np.append(shape.vertices[1:], np.expand_dims(shape.vertices[0],axis=0), axis=0)), axis=1)
    edge_bounds[:,2] = np.sum(edge_constraint[:,2] *(np.append(shape.vertices[1:], np.expand_dims(shape.vertices[0],axis=0), axis=0)), axis=1)

    return {"constraint":edge_constraint, "bounds":edge_bounds}

#good input
def get_face_zones (shape):
    '''
    Gets the constraints and bounds needed to partition the volume surrounding a polygon into zones 
    where the shortest distance from any point that is within a triangulated face zone is the distance between the 
    point and the corresponding triangulated face.

    Args:
        #will be just `self` once added into coxeter

    Returns:
        dict: "constraint": np.ndarray , "bounds": np.ndarray 
    '''
    face_constraint = np.cross(shape.edge_vectors, shape.normal) #only one face zone for a polygon | shape = (n_edges, 3)
    face_bounds = np.sum(face_constraint * shape.vertices, axis=1) #shape = (n_edges,)

    #Checking to see if all the vertices are in the face zone. If not, the polygon is nonconvex.
    if np.all(face_constraint @ np.transpose(shape.vertices) <= np.expand_dims(face_bounds, axis=1)+5e-6) == False: 
        #--Polygon is nonconvex and needs to be triangulated--
        triangle_verts =[]

        for tri in shape._triangulation():
            triangle_verts.append(list(tri))
        
        triangle_verts = np.asarray(triangle_verts)
        tri_edges = np.append(triangle_verts[:,1:], np.expand_dims(triangle_verts[:,0], axis=1), axis=1) - triangle_verts #edges point counterclockwise

        face_constraint = np.cross(tri_edges, shape.normal) #shape = (n_triangles, 3, 3)
        face_bounds = np.sum(face_constraint*triangle_verts, axis=2) #shape = (n_triangles, 3)

    else:
        #--Polygon is convex--
        face_constraint = np.expand_dims(face_constraint, axis=0) #shape = (1, n_edges, 3)
        face_bounds = np.expand_dims(face_bounds, axis=0) #shape = (1, n_edges)

    return {"constraint":face_constraint, "bounds":face_bounds}


# --- User Available Functions ---
#good input
def shortest_distance_to_surface (
        shape,
        points: np.ndarray,
        translation_vector: np.ndarray,

) -> np.ndarray:
    '''
    Solves for the shortest distance between points and the surface of a polygon. 
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
        points (list or np.ndarray): positions of the points
        translation_vector (list or np.ndarray): translation vector of the polyhedron [shape = (3,) or (2,)]

    Returns:
        np.ndarray: shortest distances [shape = (n_points,)]
    '''

    points = np.asarray(points)
    translation_vector = np.asarray(translation_vector)

    if len(points.shape) == 1:
        points = np.expand_dims(points, axis=0)

    n_points = len(points) #number of inputted points

    if points.shape[1] == 2:
        points = np.append(points, np.zeros((n_points,1)), axis=1) 

    if translation_vector.shape[0]>3 or len(translation_vector.shape)>1 or translation_vector.shape[0]<2:
        raise ValueError(f"Expected the shape of the polygon's position to be either (2,) or (3,), instead it got {translation_vector.shape}")

    if translation_vector.shape[0] == 2:
        translation_vector = np.append(translation_vector, [0])

    #Updating bounds with the position of the polyhedron
    vert_bounds = shape.vertex_zones["bounds"] + (shape.vertex_zones["constraint"] @ translation_vector)
    edge_bounds = shape.edge_zones["bounds"] + (shape.edge_zones["constraint"] @ translation_vector)
    face_bounds = shape.face_zones["bounds"] + (shape.face_zones["constraint"] @ translation_vector)


    points_trans = np.transpose(points)

    max_value = 3*np.max(LA.norm(points - (translation_vector+shape.vertices[0]), axis=1))

    min_dist_arr = np.ones((len(points),1))*max_value

    #Solving for the distances between the points and any relevant vertices
    vert_bool = np.all((shape.vertex_zones["constraint"] @ points_trans) <= np.expand_dims(vert_bounds, axis=2), axis=1) #<--- shape = (number_of_vertex_zones, number_of_points)
    if np.any(vert_bool):

         #v--- shape = (number of True in vert_bool,) ---v
        vert_used = np.transpose(np.tile(np.arange(0,shape.num_vertices,1), (n_points,1)))[vert_bool] #Contains the indices of the vertices that hold True for vert_bool
        v_points_used = np.tile(np.arange(0,n_points,1), (shape.num_vertices,1))[vert_bool] #Contains the indices of the points that hold True for vert_bool
        
        vert_dist = np.ones((shape.num_vertices,n_points))*max_value
        vert_dist[vert_bool]=LA.norm(points[v_points_used] - (shape.vertices[vert_used] + translation_vector), axis=1) #Distances between two points
        vert_dist = np.transpose(vert_dist) #<--- shape = (n_points, n_verts)

        vert_dist_arg = np.expand_dims(np.argmin(abs(vert_dist), axis=1), axis=1)
        vert_dist = np.take_along_axis(vert_dist, vert_dist_arg, axis=1)

        min_dist_arr = np.concatenate((min_dist_arr, vert_dist), axis=1)


#   Solving for the distances between the points and any relevant edges
    edge_bool = np.all((shape.edge_zones["constraint"] @ points_trans) <= np.expand_dims(edge_bounds, axis=2), axis=1) #<--- shape = (number_of_edge_zones, number_of_points)
    if np.any(edge_bool):

        #v--- shape = (number of True in edge_bool,) ---v
        edge_used = np.transpose(np.tile(np.arange(0,shape.num_vertices,1), (n_points,1)))[edge_bool] #Contains the indices of the edges that hold True for edge_bool
        e_points_used = np.tile(np.arange(0,n_points,1), (shape.num_vertices,1))[edge_bool] #Contains the indices of the points that hold True for edge_bool

        vert_on_edge = shape.vertices[shape.edges[edge_used][:,0]] + translation_vector #Vertices that lie on the needed edges
        edge_vectors = np.append(shape.vertices[1:], np.expand_dims(shape.vertices[0], axis=0), axis=0) - shape.vertices

        edge_dist = np.ones((shape.num_vertices,n_points))*max_value
        edge_dist[edge_bool]=point_to_edge_distance(points[e_points_used], vert_on_edge, edge_vectors[edge_used]) #Distances between a point and a line
        edge_dist = np.transpose(edge_dist) #<--- shape = (n_points, n_edges)

        edge_dist_arg = np.expand_dims(np.argmin(abs(edge_dist), axis=1), axis=1)
        edge_dist = np.take_along_axis(edge_dist, edge_dist_arg, axis=1)

        min_dist_arr = np.concatenate((min_dist_arr, edge_dist), axis=1)


    face_bool = np.all((shape.face_zones["constraint"] @ points_trans) <= np.expand_dims(face_bounds, axis=2), axis=1) #<--- shape = (number_of_face_zones, number_of_points)
    if np.any(face_bool):

        vert_on_face = shape.vertices[0] + translation_vector
        face_dist = point_to_face_distance(points, vert_on_face, shape.normal)
        face_dist = face_dist + max_value*(np.any(face_bool,axis=0) == False).astype(int)
        face_dist = np.expand_dims(face_dist, axis=1)

        min_dist_arr = np.concatenate((min_dist_arr, face_dist), axis=1)

    true_min_dist = np.min(min_dist_arr, axis=1)

    return true_min_dist

#good input
def shortest_displacement_to_surface (
        shape,
        points: np.ndarray,
        translation_vector: np.ndarray,
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
        points (list or np.ndarray): positions of the points
        translation_vector (list or np.ndarray): translation vector of the polyhedron [shape = (3,) or (2,)]

    Returns:
        np.ndarray: shortest displacements [shape = (n_points, 3)]
    '''
    points = np.asarray(points)
    translation_vector = np.asarray(translation_vector)

    if points.shape == (3,) or points.shape == (2,):
        points = np.expand_dims(points, axis=0)

    n_points = len(points) #number of inputted points
    n_verts = shape.num_vertices #number of vertices = number of vertex zones
    n_edges = n_verts #number of edges = number of edge zones

    if points.shape[1] == 2:
        points = np.append(points, np.zeros((n_points,1)), axis=1)

    if translation_vector.shape[0]>3 or len(translation_vector.shape)>1 or translation_vector.shape[0]<2:
        raise ValueError(f"Expected the shape of the polygon's position to be either (2,) or (3,), instead it got {translation_vector.shape}")

    if translation_vector.shape[0] == 2:
        translation_vector = np.append(translation_vector, [0])

    #Updating bounds with the position of the polyhedron
    vert_bounds = shape.vertex_zones["bounds"] + (shape.vertex_zones["constraint"] @ translation_vector)
    edge_bounds = shape.edge_zones["bounds"] + (shape.edge_zones["constraint"] @ translation_vector)
    face_bounds = shape.face_zones["bounds"] + (shape.face_zones["constraint"] @ translation_vector)


    points_trans = np.transpose(points)

    max_value = 3*np.max(LA.norm(points - (translation_vector+shape.vertices[0]), axis=1))

    min_disp_arr = np.ones((n_points,1, 3))*max_value

    #Solving for the distances between the points and any relevant vertices
    vert_bool = np.all((shape.vertex_zones["constraint"] @ points_trans) <= np.expand_dims(vert_bounds, axis=2), axis=1) #<--- shape = (number_of_vertex_zones, number_of_points)
    if np.any(vert_bool):

        #v--- shape = (number of True in vert_bool,) ---v
        vert_used = np.transpose(np.tile(np.arange(0,n_verts,1), (n_points,1)))[vert_bool] #Contains the indices of the vertices that hold True for vert_bool
        v_points_used = np.tile(np.arange(0,n_points,1), (n_verts,1))[vert_bool] #Contains the indices of the points that hold True for vert_bool
        
        vert_disp = np.ones((n_verts,n_points,3))*max_value
        vert_disp[vert_bool]=(shape.vertices[vert_used] + translation_vector) - points[v_points_used] #Displacements between two points
        vert_disp = np.transpose(vert_disp, (1,0,2)) #<--- shape = (n_points, n_verts, 3)
        
        vert_disp_min = np.expand_dims(np.argmin( LA.norm(vert_disp, axis=2), axis=1), axis=(1,2))
        vert_disp = np.take_along_axis(vert_disp, vert_disp_min, axis=1)

        min_disp_arr = np.concatenate((min_disp_arr, vert_disp), axis=1)


#   Solving for the distances between the points and any relevant edges
    edge_bool = np.all((shape.edge_zones["constraint"] @ points_trans) <= np.expand_dims(edge_bounds, axis=2), axis=1) #<--- shape = (number_of_edge_zones, number_of_points)
    if np.any(edge_bool):

        #v--- shape = (number of True in edge_bool,) ---v
        edge_used = np.transpose(np.tile(np.arange(0,n_edges,1), (n_points,1)))[edge_bool] #Contains the indices of the edges that hold True for edge_bool
        e_points_used = np.tile(np.arange(0,n_points,1), (n_edges,1))[edge_bool] #Contains the indices of the points that hold True for edge_bool

        vert_on_edge = shape.vertices[shape.edges[edge_used][:,0]] + translation_vector #Vertices that lie on the needed edges
        edge_vectors = np.append(shape.vertices[1:], np.expand_dims(shape.vertices[0], axis=0), axis=0) - shape.vertices

        edge_disp = np.ones((n_edges,n_points,3))*max_value
        edge_disp[edge_bool]=point_to_edge_displacement(points[e_points_used], vert_on_edge, edge_vectors[edge_used]) #Displacements between a point and a line
        edge_disp = np.transpose(edge_disp, (1, 0, 2)) #<--- shape = (n_points, n_edges, 3)

        edge_disp_arg = np.expand_dims(np.argmin( LA.norm(edge_disp, axis=2), axis=1), axis=(1,2))
        edge_disp = np.take_along_axis(edge_disp, edge_disp_arg, axis=1)

        min_disp_arr = np.concatenate((min_disp_arr, edge_disp), axis=1)


    face_bool = np.all((shape.face_zones["constraint"] @ points_trans) <= np.expand_dims(face_bounds, axis=2), axis=1) #<--- shape = (number_of_face_zones, number_of_points)
    if np.any(face_bool):

        face_disp = point_to_face_displacement(points, shape.vertices[0]+translation_vector, shape.normal) + np.repeat(np.expand_dims((max_value*(np.any(face_bool,axis=0) == False).astype(int)), axis=1), 3, axis=1)

        min_disp_arr = np.concatenate((min_disp_arr, np.expand_dims(face_disp, axis=1)), axis=1)

    disp_list_bool = np.argmin( (LA.norm(min_disp_arr, axis=2)), axis=1).reshape(n_points, 1, 1)
    true_min_disp = np.squeeze(np.take_along_axis(min_disp_arr, disp_list_bool, axis=1), axis=1)

    return true_min_disp


#think it is right/will work correctly?
def spheropolygon_shortest_displacement_to_surface (
        shape,
        points: np.ndarray,
        translation_vector: np.ndarray,
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
        points (list or np.ndarray): positions of the points
        translation_vector (list or np.ndarray): translation vector of the polyhedron [shape = (3,) or (2,)]

    Returns:
        np.ndarray: shortest displacements [shape = (n_points, 3)]
    '''
    points = np.asarray(points)
    translation_vector = np.asarray(translation_vector)

    if points.shape == (3,) or points.shape == (2,):
        points = np.expand_dims(points, axis=0)

    n_points = len(points) #number of inputted points
    n_verts = shape.num_vertices #number of vertices = number of vertex zones
    n_edges = n_verts #number of edges = number of edge zones

    if points.shape[1] == 2:
        points = np.append(points, np.zeros((n_points,1)), axis=1)

    if translation_vector.shape[0]>3 or len(translation_vector.shape)>1 or translation_vector.shape[0]<2:
        raise ValueError(f"Expected the shape of the polygon's position to be either (2,) or (3,), instead it got {translation_vector.shape}")

    if translation_vector.shape[0] == 2:
        translation_vector = np.append(translation_vector, [0])

    #Updating bounds with the position of the polyhedron
    vert_bounds = shape.vertex_zones["bounds"] + (shape.vertex_zones["constraint"] @ translation_vector)
    edge_bounds = shape.edge_zones["bounds"] + (shape.edge_zones["constraint"] @ translation_vector)
    face_bounds = shape.face_zones["bounds"] + (shape.face_zones["constraint"] @ translation_vector)


    points_trans = np.transpose(points)

    max_value = 3*np.max(LA.norm(points - (translation_vector+shape.vertices[0]), axis=1))

    min_disp_arr = np.ones((n_points,1, 3))*max_value

    #Solving for the distances between the points and any relevant vertices
    vert_bool = np.all((shape.vertex_zones["constraint"] @ points_trans) <= np.expand_dims(vert_bounds, axis=2), axis=1) #<--- shape = (number_of_vertex_zones, number_of_points)
    if np.any(vert_bool):

        #v--- shape = (number of True in vert_bool,) ---v
        vert_used = np.transpose(np.tile(np.arange(0,n_verts,1), (n_points,1)))[vert_bool] #Contains the indices of the vertices that hold True for vert_bool
        v_points_used = np.tile(np.arange(0,n_points,1), (n_verts,1))[vert_bool] #Contains the indices of the points that hold True for vert_bool
        
        vert_disp = np.ones((n_verts,n_points,3))*max_value
        vert_disp[vert_bool]=(shape.vertices[vert_used] + translation_vector) - points[v_points_used] #Displacements between two points
        vert_disp = np.transpose(vert_disp, (1,0,2)) #<--- shape = (n_points, n_verts, 3)
        
        vert_disp_min = np.expand_dims(np.argmin( LA.norm(vert_disp, axis=2), axis=1), axis=(1,2))
        vert_disp = np.take_along_axis(vert_disp, vert_disp_min, axis=1)

        #for spheropolygon
        vert_projection = vert_disp - np.expand_dims(vert_disp @ (shape.normal/np.linalg.norm(shape.normal)), axis=1) * (shape.normal/np.linalg.norm(shape.normal))
        v_projection_bool = np.linalg.norm(vert_projection, axis=1) > shape.radius
        vert_projection[v_projection_bool] = shape.radius * vert_projection[v_projection_bool]/np.linalg.norm(vert_projection[v_projection_bool], axis=1)
        vert_disp = vert_disp - vert_projection

        min_disp_arr = np.concatenate((min_disp_arr, vert_disp), axis=1)


#   Solving for the distances between the points and any relevant edges
    edge_bool = np.all((shape.edge_zones["constraint"] @ points_trans) <= np.expand_dims(edge_bounds, axis=2), axis=1) #<--- shape = (number_of_edge_zones, number_of_points)
    if np.any(edge_bool):

        #v--- shape = (number of True in edge_bool,) ---v
        edge_used = np.transpose(np.tile(np.arange(0,n_edges,1), (n_points,1)))[edge_bool] #Contains the indices of the edges that hold True for edge_bool
        e_points_used = np.tile(np.arange(0,n_points,1), (n_edges,1))[edge_bool] #Contains the indices of the points that hold True for edge_bool

        vert_on_edge = shape.vertices[shape.edges[edge_used][:,0]] + translation_vector #Vertices that lie on the needed edges
        edge_vectors = np.append(shape.vertices[1:], np.expand_dims(shape.vertices[0], axis=0), axis=0) - shape.vertices

        edge_disp = np.ones((n_edges,n_points,3))*max_value
        edge_disp[edge_bool]=point_to_edge_displacement(points[e_points_used], vert_on_edge, edge_vectors[edge_used]) #Displacements between a point and a line
        edge_disp = np.transpose(edge_disp, (1, 0, 2)) #<--- shape = (n_points, n_edges, 3)

        edge_disp_arg = np.expand_dims(np.argmin( LA.norm(edge_disp, axis=2), axis=1), axis=(1,2))
        edge_disp = np.take_along_axis(edge_disp, edge_disp_arg, axis=1)

        #for spheropolygon
        edge_projection = edge_disp - np.expand_dims(edge_disp @ (shape.normal/np.linalg.norm(shape.normal)), axis=1) * (shape.normal/np.linalg.norm(shape.normal))
        e_projection_bool = np.linalg.norm(edge_projection, axis=1) > shape.radius
        edge_projection[e_projection_bool] = shape.radius * edge_projection[e_projection_bool]/np.linalg.norm(edge_projection[e_projection_bool], axis=1)
        edge_disp = edge_disp - edge_projection

        min_disp_arr = np.concatenate((min_disp_arr, edge_disp), axis=1)


    face_bool = np.all((shape.face_zones["constraint"] @ points_trans) <= np.expand_dims(face_bounds, axis=2), axis=1) #<--- shape = (number_of_face_zones, number_of_points)
    if np.any(face_bool):

        face_disp = point_to_face_displacement(points, shape.vertices[0]+translation_vector, shape.normal) + np.repeat(np.expand_dims((max_value*(np.any(face_bool,axis=0) == False).astype(int)), axis=1), 3, axis=1)

        min_disp_arr = np.concatenate((min_disp_arr, np.expand_dims(face_disp, axis=1)), axis=1)

    disp_list_bool = np.argmin( (LA.norm(min_disp_arr, axis=2)), axis=1).reshape(n_points, 1, 1)
    true_min_disp = np.squeeze(np.take_along_axis(min_disp_arr, disp_list_bool, axis=1), axis=1)

    return true_min_disp
