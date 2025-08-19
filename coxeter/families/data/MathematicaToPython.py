import json
import subprocess
import os
import re
import numpy as np
from coxeter.shapes import ConvexPolyhedron


def update_polyhedron_vertices_by_source(input_path):
    """
    Update the stored JSON shapes via wolframscript calls.

    Args:
        input_path (str): The path to the master JSON file.
    """
    try:
        with open(input_path, encoding="utf-8") as f:
            polyhedra_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The master file '{input_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_path}'")
        return

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

            # Construct the wolframscript command with volume normalization and centering
            command = [
                "wolframscript",
                "-c",
                f"""
                (*Must evaluate numerically, as real algebra is prohibitively slow.*)
                vertices = N[
                    PolyhedronData["{wolfram_name}", "VertexCoordinates"] /
                    CubeRoot[PolyhedronData["{wolfram_name}", "Volume"]],
                    36 (*Make sure future results are accurate to 32 decimals.*)
                ];
                (*Center the shape on the origin*)
                centroid = RegionCentroid @ ConvexHullRegion[vertices]; 
                vertices = Map[
                    # - centroid&, vertices
                ];

                ExportString[
                    (*Replace values close to 0 with 0*)
                    N[vertices/. x_ /; Abs[x] < 1*^-16 -> 0.0, 32]
                    ,"RawJSON"
                ]
                """,
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

                np.testing.assert_allclose(ConvexPolyhedron(new_vertices).volume, 1)
                np.testing.assert_allclose(
                    ConvexPolyhedron(new_vertices).centroid, 0, atol=1e-16
                )

                if full_name in source_data:
                    source_data[full_name]["vertices"] = new_vertices
                    print(f"   Successfully updated '{full_name}'.")
                else:
                    print(f"Warning: Could not find '{full_name}' in '{source_file}'.")

            except subprocess.CalledProcessError as e:
                print(
                    f"Error running wolframscript for '{full_name}': {e.stderr.strip()}"
                )
            except json.JSONDecodeError:
                print(f"Error: Failed to parse JSON output for '{full_name}'.")
            except FileNotFoundError:
                print("Error: 'wolframscript' command not found.")
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


if __name__ == "__main__":
    import sys

    update_polyhedron_vertices_by_source(sys.argv[1])
