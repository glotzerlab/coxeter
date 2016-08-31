# For calling up commonly used shapes, as well as some shape related tools

from . import np
from . import ConvexHull, Delaunay

# This has more functionality than the freud equivilent, so its not totally redundant
# Truncation ranges from 0 (octahedra) to 2/3 (truncated octahedra) to 1 (cuboctahedra)
def octahedron(side_length = 1, truncation = 0):
	pre = np.sqrt(2)/2
	if truncation:
				
		trunc_value = truncation*np.sqrt(2)*side_length/4
		
		return np.asarray([[pre*side_length-trunc_value,trunc_value,0],
                         [pre*side_length-trunc_value,-trunc_value,0],
                         [pre*side_length-trunc_value,0,trunc_value],
                         [pre*side_length-trunc_value,0,-trunc_value],
                         [-pre*side_length+trunc_value,trunc_value,0],
                         [-pre*side_length+trunc_value,-trunc_value,0],
                         [-pre*side_length+trunc_value,0,trunc_value],
                         [-pre*side_length+trunc_value,0,-trunc_value],
                         [0,pre*side_length-trunc_value,trunc_value],
                         [0,pre*side_length-trunc_value,-trunc_value],
                         [trunc_value,pre*side_length-trunc_value,0],
                         [-trunc_value,pre*side_length-trunc_value,0],
                         [0,-pre*side_length+trunc_value,trunc_value],
                         [0,-pre*side_length+trunc_value,-trunc_value],
                         [trunc_value,-pre*side_length+trunc_value,0],
                         [-trunc_value,-pre*side_length+trunc_value,0],
                         [0,trunc_value,pre*side_length-trunc_value],
                         [0,-trunc_value,pre*side_length-trunc_value],
                         [trunc_value,0,pre*side_length-trunc_value],
                         [-trunc_value,0,pre*side_length-trunc_value],
                         [0,trunc_value,-pre*side_length+trunc_value],
                         [0,-trunc_value,-pre*side_length+trunc_value],
                         [trunc_value,0,-pre*side_length+trunc_value],
                         [-trunc_value,0,-pre*side_length+trunc_value]])
	else:
		return np.asarray([[pre*side_length,0,0],[-pre*side_length,0,0],[0,pre*side_length,0],
                         [0,-pre*side_length,0],[0,0,pre*side_length],[0,0,-pre*side_length]])

def cube(side_length = 1):
	return np.asarray([[side_length/2,side_length/2,side_length/2],
		[-side_length/2,side_length/2,side_length/2],
		[side_length/2,-side_length/2,side_length/2],
		[side_length/2,side_length/2,-side_length/2],
		[-side_length/2,-side_length/2,side_length/2],
		[-side_length/2,side_length/2,-side_length/2],
		[side_length/2,-side_length/2,-side_length/2],
		[-side_length/2,-side_length/2,-side_length/2]])

def rectangle(a, b, c):
	return np.asarray([[a/2,b/2,c/2],
		[-a/2,b/2,c/2],
		[a/2,-b/2,c/2],
		[a/2,b/2,-c/2],
		[-a/2,-b/2,c/2],
		[-a/2,b/2,-c/2],
		[a/2,-b/2,-c/2],
		[-a/2,-b/2,-c/2]])

# This golden spiral code snippet is taken from http://www.softimageblog.com/archives/115
def poly_sphere(radius = 1, points = 100):
	stype = 'polyedral sphere'
	pts = []
	inc = np.pi*(3 - np.sqrt(5))
	off = 2/float(points)
	for k in range(0, int(points)):
		y = k*off - 1 + (off/2)
		r = np.sqrt(1 - y*y)
		phi = k*inc
		pts.append([np.cos(phi)*r, y, np.sin(phi)*r])
	return radius*np.array(pts) 

def rhombic_dodecahedron(a = 1):
	return np.array([[a,a,a],[-a,a,a],[a,-a,a],
                     [a,a,-a],[a,-a,-a],[-a,a,-a],
                     [-a,-a,a],[-a,-a,-a],[2.0*a,0.0,0.0],
                     [-2.0*a,0.0,0.0],[0.0,2.0*a,0.0],[0.0,-2.0*a,0.0],
                     [0.0,0.0,2.0*a],[0.0,0.0,-2.0*a]])

# Plots a frame of lines based on the simplicies of a set of verts. ax is a 
# matplotlib axis object, and color should be a string or vector interpretable by matplotlib
# as a plot color for the lines
def plot_skeleton(verts, color, ax):
    hull = ConvexHull(verts)
    for simplex in hull.simplices:
        ax.plot(*zip(hull.points[simplex[0]],hull.points[simplex[1]]), c=color)
        ax.plot(*zip(hull.points[simplex[1]],hull.points[simplex[2]]), c=color)
        ax.plot(*zip(hull.points[simplex[2]],hull.points[simplex[0]]), c=color)
    return None

# Finds the intersetion point of a plane and a line
# line given as [p0,p], plane as [n0,n]
def line_plane_intersection(line, plane):
    line = np.asarray(line)
    plane = np.asarray(plane)
    t = np.dot(plane[1],(plane[0]-line[0]))/np.dot(plane[1],(line[1]-line[0]))
    return line[0] + t*(line[1]-line[0])

# Returns the verts of a shape that is input verts sliced by the input plane
# Verts on the positive side of the plane are retained and negative are discarded
def shape_slice(verts, plane):
    # get all the lines of the shape
    lines = []
    hull = ConvexHull(verts)
    for simplex in hull.simplices:
        for point1 in simplex:
            for point2 in simplex:
            # if the points straddle the plane
                if(np.dot(plane[1],hull.points[point1]-plane[0])*np.dot(plane[1],hull.points[point2]-plane[0])<0):
                    lines.append([hull.points[point1],hull.points[point2]])
    intersections = []
    for line in lines:
        intersections.append(line_plane_intersection(line,plane))
    # add any points that are already on the plane
    for vert in verts:
        if np.abs(np.dot(plane[1],vert-plane[0]))<1e-8:
            intersections.append(vert)
    intersections = np.asarray(intersections)
    # remove nan's if some lines were parallel
    intersections = intersections[~np.any((np.isnan(intersections)), axis=1),:]
    # discard old points that are below the plane
    verts = verts[np.dot(verts, plane[1])>0,:]
    # append the new intersections
    verts = np.append(verts, intersections, axis=0)
    
    # remove duplicate points
    inds = np.ones(verts.shape[0]).astype(bool)
    for i in range(verts.shape[0]):
        if inds[i]:
            for j in range(i+1,verts.shape[0]):
                if np.linalg.norm(verts[i]-verts[j])<1e-8:
                    inds[j] = False
                                                                                        
    return verts[inds,:]

def simplical_content(verts):
    if len(verts)==4:
        edges=verts-verts[0,:]
        return np.abs(np.dot(edges[1],np.cross(edges[2],edges[3])))/6

    if len(verts)==3:
        dist=pdist(verts,'euclidean')
        s=(dist[0]+dist[1]+dist[2])/2
        return np.sqrt(s*(s-dist[0])*(s-dist[1])*(s-dist[2]))    

def convex_content(verts):
    trian = Delaunay(verts)
    total_vol = 0
    for simplex in trian.simplices:
        simp_verts = verts[simplex]
        total_vol += simplical_content(simp_verts)
    return total_vol

# Returns a list of tuples of cap angles and lengths (angle, length)
def cylinder_caps(verts):
    cap_list = []
    hull = ConvexHull(verts)
    for i in range(hull.simplices.shape[0]):
        neighbor_list = np.argwhere(hull.neighbors==i)
        for j in neighbor_list[:,0]:
            if j>i:
                eq1 = hull.equations[i][0:3]
                eq2 = hull.equations[j][0:3]
                # Angle of the cylindrical edge
                angle = np.arccos(np.dot(eq1, eq2)/(np.linalg.norm(eq1)*np.linalg.norm(eq2)))
                # Shared points of the neighboring simplices
                shared = np.intersect1d(hull.simplices[i], hull.simplices[j])
                # Length of the cylindrical edge
                length = np.linalg.norm(hull.points[shared[0]]-hull.points[shared[1]])
                if angle<2*np.pi:
                    cap_list.append((angle, length))
    return cap_list

# Returns the areas of all the simplices of a convex hull
def simplex_area(verts):
    areas = []
    hull = ConvexHull(verts)
    for simplex in hull.simplices:
        areas.append(simplical_content(hull.points[simplex]))
      
    return areas

def spheropolyhedra_volume(verts, R=1.0):
    assert(R>=0)
    hull = ConvexHull(verts)
    # Base volume
    convex_vol = hull.volume
    # Rectangular section volume
    rect_vol = hull.area*R
    # Cylinder cap volume
    cyl_vol = 0
    for (ang, length) in cylinder_caps(verts):
        cyl_vol += R**2*length*np.pi*(ang/(2*np.pi))
    
    assert((convex_vol>=0)*(rect_vol>=0)*(cyl_vol>=0))
    return convex_vol + rect_vol + cyl_vol + 4*np.pi*R**3/3
