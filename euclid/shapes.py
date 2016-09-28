# For calling up commonly used shapes
from . import np

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

