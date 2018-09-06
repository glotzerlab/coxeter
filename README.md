# euclid - shape and geometry tools

## About

`euclid` is a collection of tools to help initialize and manipulate shapes. It includes prototypes for commonly used shapes, as well as a variety of helper functions.

The package includes:

1. Functions and shapes collected from several group members. These are in shapes.py, and utils.py. There is also a quaternion package in quaternion\_tools.py. For example, call `from euclid import quaternion_tools as qt` and `rand_q = qt.qrandom(10)` to import quaternion\_tools and make 10 random quaternions

2. A migration of `freud.shape` functions and classes taken from version 0.6 of `freud` can be imported through `euclid.FreudShape`: `from euclid.FreudShape.Cube import shape`


## Installation

From within the euclid folder, run

```bash
pip install . --user
```
