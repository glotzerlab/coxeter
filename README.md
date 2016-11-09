# euclid - shape and geometry tools

## About

euclid is a package to help members of the Glotzer research group initialize and manipulate shapes. It includes prototypes for commonly used shapes, as well as a variety of helper functions
There are currently two susections to euclid:

1. Functions and shapes collected from several group members. These are in shapes.py, and utils.py. There is also a quaternion package in quaternion_tools.py. For example, call 

   from euclid import quaternion_tools as qt
   rand_q = qt.qrandom(10)

   to import quaternion_tools and make 10 random quaternions

2. A migration of freud.shape. functions and classes from version 0.6 of freud can be imported throug euclid.FreudShape:

   from euclid.FreudShape.Cube import shape


## Installation

The recommeded method of installation is using *setup.py*:

From within the euclid folder, run
    
    python setup.py install --user

This works while in a conda environment

## Other Packages

euclid requires numpy and scipy


