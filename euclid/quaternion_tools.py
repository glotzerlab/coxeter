# quaternion_tools.py
# Bryan VanSaders <bvansade@umich.edu>
# This is a collection of quaternion manipulation tools
# Conventions taken from www.lfd.uci.edu/~gohlke/code/transformations.py.html

import numpy as np

EPS = np.finfo(float).eps * 4  # A small number

# This function takes two vectors and finds the quaternion which will
# rotate the first into the second


def find_quat(v1, v2):
    v = np.array(v1)/np.linalg.norm(v1)
    t = np.array(v2)/np.linalg.norm(v2)

    axis = np.cross(v, t)
    if np.linalg.norm(axis) < EPS:
        axis = v
    norm = np.linalg.norm(v)
    if norm > EPS:
        acos = np.dot(v, t) / np.linalg.norm(v)
        if np.absolute(np.absolute(acos) - 1) < EPS:
            angle = 0
        else:
            angle = np.arccos(acos)
    else:
        angle = 0
    q = build_quat(axis, angle)
    return qinverse(q)

# This function produces a quaternion which re-orients vx along [100] and
# vy along [010]  (as projected along the [100] plane)


def find_xyquat(vx, vy):
    tx = np.array([1, 0, 0])
    ty = np.array([0, 1, 0])
    vx = np.array(vx)
    vy = np.array(vy)

    qx = find_quat(vx, tx)
    vy_rot = qrotate(qx, vy)
    y_inplane = vy_rot - np.dot(vy_rot, tx) * tx
    qy = find_quat(y_inplane, ty)

    return qproduct(qy, qx)

# This function takes a vector and an angle in radians and makes a
# quaternion of that rotation about that vector (positive is clockwise)


def build_quat(axis, angle):
    q = np.array([0, axis[0], axis[1], axis[2]])
    norm = np.linalg.norm(q)
    q = q * np.sin(angle / 2.0) / norm
    q[0] = np.cos(angle / 2.0)
    return q

# Old product function that I am phasing out
def _qproduct(q1, q2):
    a1, b1, c1, d1 = q1
    a2, b2, c2, d2 = q2

    return np.array([-b2 * b1 - c2 * c1 - d2 * d1 + a2 * a1,
                     b2 * a1 + c2 * d1 - d2 * c1 + a2 * b1,
                     -b2 * d1 + c2 * a1 + d2 * b1 + a2 * c1,
                     b2 * c1 - c2 * b1 + d2 * a1 + a2 * d1])

# New product function that can handle arrays
def qproduct(q1, q2):
    q1 = np.array(q1).reshape((-1,4))
    q2 = np.array(q2).reshape((-1,4))

    # They must have the same number of quaterions
    assert(q1.shape[0]==q2.shape[0])

    q3 = np.zeros(q1.shape)

    q3[:,0] = -q2[:,1]*q1[:,1] - q2[:,2]*q1[:,2] - q2[:,3]*q1[:,3] + q2[:,0]*q1[:,0]
    q3[:,1] = q2[:,1]*q1[:,0] + q2[:,2] * q1[:,3] - q2[:,3] * q1[:,2] + q2[:,0] * q1[:,1]
    q3[:,2] = -q2[:,1] * q1[:,3] + q2[:,2] * q1[:,0] + q2[:,3] * q1[:,1] + q2[:,0] * q1[:,2]
    q3[:,3] = q2[:,1] * q1[:,2] - q2[:,2] * q1[:,1] + q2[:,3] * q1[:,0] + q2[:,0] * q1[:,3]

    # Make sure it is normalized
    #assert(np.abs(np.linalg.norm(q3, axis=1).sum() - q3.shape[0]) <1e-6)

    # Make it a 1D vector if that's how it got passed in
    if q3.shape[0] == 1:
        q3 = q3[0]

    return q3


# new vectorized inverse function
def qinverse(q):
    qinv = np.array(q).reshape((-1,4))
    qinv[:,1:] = -qinv[:,1:]
    # Normalization
    qinv = qinv / np.tile((qinv*qinv).sum(axis=1).reshape((-1,1)), (1,4))

    if qinv.shape[0] == 1:
        qinv = qinv[0]
    return qinv

def qrotate(q, v):
    v = np.array(v).reshape((-1,3))
    qv = np.zeros((v.shape[0],4))
    qv[:,1:] = v

    q = np.array(q).reshape((-1,4))
    q = q/np.linalg.norm(q,axis=1)
    q = np.tile(q, (v.shape[0],1))

    v2 = qproduct(q, qv)
    vf = qproduct(v2, qinverse(q))

    # Handle the possibilites for array
    # dimensionality
    if len(vf.shape) > 1:
        vf = vf[:,1:]
        if vf.shape[0] == 1:
            vf = vf[0]
    else:
        vf = vf[1:]

    return vf

# A reflection quaternion has the form [0,nx,ny,nz],
# where [nx,ny,nz] is the normal of the mirror plane
def qreflect(q,v):
    v = np.array(v).reshape((-1,3))
    qv = np.zeros((v.shape[0],4))
    qv[:,1:] = v

    q = np.array(q).reshape((-1,4))
    q = q/np.linalg.norm(q)
    q = np.tile(q, (v.shape[0],1))

    v2 = qproduct(q, qv)
    vf = qproduct(v2, q)
    vf = vf[:,1:]
    if vf.shape[0] == 1:
        vf = vf[0]
    return vf

# Taken from here: http://planning.cs.uiuc.edu/node198.html
def qrandom(num=1):
    u = np.random.rand(num,3)
    q = np.zeros((num,4))
    q[:,0] = np.sqrt(1-u[:,0])*np.sin(2*np.pi*u[:,1])
    q[:,1] = np.sqrt(1-u[:,0])*np.cos(2*np.pi*u[:,1])
    q[:,2] = np.sqrt(u[:,0])*np.sin(2*np.pi*u[:,2])
    q[:,3] = np.sqrt(u[:,0])*np.cos(2*np.pi*u[:,2])

    if num == 1:
        q = q[0]

    return q

