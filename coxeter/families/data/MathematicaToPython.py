# Copyright (c) 2015-2024 The Regents of the University of Michigan.
# This file is from the coxeter project, released under the BSD 3-Clause License.

"""Convert Mathematica polyhedron vertices to Python.

The input stream is parsed ta a shape definition file. The script must be
called with a single argument, the name of the shape. This is used to generate
the data in the science1220869.json file from :cite:`Damasceno2012a`.

Mathematica data is obtained with calls like
ExportString[PolyhedronData["Cube", "VertexCoordinates"], "Table",
"FieldSeparators" -> ", "] or ExportString[PolyhedronData[{"Prism", 6},
"VertexCoordinates"], "Table", "FieldSeparators" -> ", "] and the results cut
and pasted to terminal input to this script.  Input is terminated with a
newline and end-of-file character, i.e. CTRL-D

Numeric interpretation of Mathematica data may be necessary where translation
to Python isn't as easy.  E.g.
ExportString[N[PolyhedronData["ObtuseGoldenRhombohedron",
"VertexCoordinates"]], "Table", "FieldSeparators" -> ", "]
"""


def main():
    """Convert the output from Mathematica to a Python script."""
    import sys

    name = sys.argv[1]
    outfile = open(name + ".py", "w")

    # Set up some boiler plate

    header = """from __future__ import division
    from numpy import sqrt
    import numpy
    from coxeter.common_shapes import ConvexPolyhedron

    # Example:
    """

    example = f"# from coxeter.common_shapes.{name} import shape\n"

    footer = """         ]

    shape = ConvexPolyhedron(numpy.array(points))
    """

    # Process the input

    pstrings = []
    instring = sys.stdin.read()
    # Strip out quotes
    instring = instring.replace('"', "")
    # merge wrapped lines
    instring = instring.replace("\\\n", "")
    # split input into list of lines
    lines = instring.splitlines()
    for line in lines:
        # Turn Mathematica syntax into Python syntax
        line = line.replace("Sqrt", "sqrt")
        line = line.replace("[", "(").replace("]", ")")
        line = line.replace("^", "**")
        # get string values of x,y,z
        x, y, z = line.split(", ")
        pstring = f"          ({x}, {y}, {z}),\n"
        pstrings.append(pstring)

    # Write the output

    outfile.write(header)
    outfile.write(example)
    outfile.write("points = [ \n")

    outfile.writelines(pstrings)

    outfile.write(footer)
    outfile.close()


if __name__ == "__main__":
    main()
