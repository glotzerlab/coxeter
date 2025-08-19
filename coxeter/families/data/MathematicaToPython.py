# Copyright (c) 2015-2025 The Regents of the University of Michigan.
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


<<<<<<< HEAD
UPDATE 08/2025: the following code is simpler and less brittle
ExportString[
    N[PolyhedronData["AcuteGoldenRhombohedron", "VertexCoordinates"], 32],
    "RawJSON", "Compact" -> False
]
"""
=======
def update_polyhedron_vertices_by_source(input_path):
    """Update the stored JSON shapes via wolframscript calls.

    Args:
        input_path (str): The path science1220869.json
    """
    with open(input_path, encoding="utf-8") as f:
        polyhedra_data = json.load(f)

    # Group polyhedra by their source file
    polyhedra_by_source = {}
    for polyhedron_info in polyhedra_data.values():
        source_file = polyhedron_info.get("source")
        if source_file:
            if source_file not in polyhedra_by_source:
                polyhedra_by_source[source_file] = []
            polyhedra_by_source[source_file].append(polyhedron_info)

    print(f"Found {len(polyhedra_data)} polyhedra in {len(polyhedra_by_source)} files.")

    for source_file, polyhedra_list in polyhedra_by_source.items():
        print(f"\n--- Processing source file: '{source_file}' ---")

        # Load the existing data from the current source file
        try:
            with open(source_file, encoding="utf-8") as f:
                source_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Source file '{source_file}' not found. Skipping.")
            continue
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{source_file}'. Skipping.")
            continue

        # Iterate through the polyhedra for this source file and update vertices
        for polyhedron_info in polyhedra_list:
            full_name = polyhedron_info.get("name")
            if not full_name:
                print("Warning: Skipping a polyhedron due to missing 'name' field.")
                continue

            # Remove spaces from the name for the WolframScript command
            wolfram_name = re.sub(r"\s+", "", full_name)

            # Construct the wolframscript command
            command = [
                "wolframscript",
                "-c",
                (
                    f'ExportString[N[PolyhedronData["{wolfram_name}"'
                    ', "VertexCoordinates"] / '
                    f'PolyhedronData["{wolfram_name}", "Volume"]^(1/3), 32], '
                    '"RawJSON", "Compact" -> False"]'
                ),
            ]

            print(f"-> Updating vertices for '{full_name}'...")

            # Execute the command and capture the output
            try:
                result = subprocess.run(
                    command, capture_output=True, text=True, check=True
                )
                vertices_json_string = result.stdout.strip()

                # Parse the JSON string to get a Python list of vertices
                new_vertices = json.loads(vertices_json_string)
                print(new_vertices)

                if full_name in source_data:
                    source_data[full_name]["vertices"] = new_vertices
                    print(f"   Successfully updated '{full_name}'.")
                else:
                    print(
                        f"Warning: Could not find '{full_name}' in '{source_file}'."
                    )

            except subprocess.CalledProcessError as e:
                print(
                    f"Error running wolframscript for '{full_name}': {e.stderr.strip()}"
                )
            except json.JSONDecodeError:
                print(
                    f"Error: Failed to parse JSON output for '{full_name}'."
                )
            except FileNotFoundError:
                print(
                    "Error: 'wolframscript' command not found."
                )
                return

        if source_data:
            try:
                with open(source_file, "w", encoding="utf-8") as f:
                    json.dump(source_data, f, indent=4)
                print(f"Successfully updated '{source_file}'.")
            except OSError as e:
                print(f"Error: Could not write to file '{source_file}'. Reason: {e}")
        else:
            print(f"No data to write for '{source_file}'. The file was not modified.")
>>>>>>> bdc49b8 (Clean up m2py)


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
