euclid is meant to contain tools for dealing with shapes. It is pure python, and can be installed by

python setup.py install --user

It requires numpy and scipy. There are currently two susections to euclid:

1. Functions and shapes collected from several group members. These are in shapes.py, and utils.py. There is also a quaternion package in quaternion_tools.py. For example, call 

   from euclid import quaternion_tools as qt
   rand_q = qt.qrandom(10)

   to import quaternion_tools and make 10 random quaternions

2. A migration of freud.shape here. 


