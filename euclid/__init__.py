from . import shapes
from . import utils
from . import symmetry
from . import ft
from . import polygon
from . import polyhedron

import os
import importlib

module = __import__(__name__)
path = os.path.join(os.path.dirname(module.__file__),
                    'common_shapes')
for filename in os.listdir(path):
    if filename.startswith('_'):
        continue
    importlib.import_module('euclid.common_shapes.' + filename[:-3])
