# Copyright (c) 2015-2025 The Regents of the University of Michigan.
# This file is from the coxeter project, released under the BSD 3-Clause License.

import json
import re
import subprocess
import sys

import numpy as np

from coxeter.shapes import ConvexPolyhedron


def update_polyhedron_vertices_by_source(input_path):
    """
    Update the stored JSON shapes via wolframscript calls.

    Args:
        input_path (str): The path to the master JSON file.
    """
    # Load the data from the master JSON file
    try:
        with open(input_path, encoding="utf-8") as f:
            polyhedra_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The master file '{input_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_path}'")
        return

    # A dictionary to hold the updated data for each source file,
    # so we only load and write each source file once.
    updated_source_data = {}

    print(f"Found {len(polyhedra_data)} polyhedra.")

    # Iterate through the polyhedra in the master file and update their vertices
    for short_name, polyhedron_info in polyhedra_data.items():
        full_name = polyhedron_info.get("alternative_name")
        if not full_name:
            full_name = polyhedron_info.get("name")

        # full_name = polyhedron_info.get("name")
        source_file = polyhedron_info.get("source")

        if not full_name or not source_file:
            print(
                f"Warning: Skipping {short_name} due to missing field (name, source)"
                f"({full_name}, {source_file})",
            )
            continue
        if (
            "plat" in source_file
            or "arch" in source_file
            or "catalan" in source_file
            or "other" in source_file
        ):
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

        print(f"-> Updating vertices for '{full_name}' ({short_name})...")

        # Execute the command and capture the output
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            vertices_json_string = result.stdout.strip()

            # Parse the JSON string to get a Python list of vertices
            new_vertices = json.loads(vertices_json_string)

            # Perform assertions for volume and centroid
            np.testing.assert_allclose(ConvexPolyhedron(new_vertices).volume, 1)
            np.testing.assert_allclose(
                ConvexPolyhedron(new_vertices).centroid, 0, atol=1e-15
            )

            # Update the master file's data in memory
            polyhedra_data[short_name]["vertices"] = new_vertices
            print(f"   Successfully updated master file data for '{full_name}'.")

            # Load and update the source file data in memory, if not already loaded
            if source_file not in updated_source_data:
                try:
                    with open(source_file) as f:
                        updated_source_data[source_file] = json.load(f)
                except FileNotFoundError:
                    print("File not found: continuing")
                    updated_source_data[source_file] = ""

            # Update the source file's data in memory
            if full_name in updated_source_data[source_file]:
                updated_source_data[source_file][full_name]["vertices"] = new_vertices
                print(f"   Successfully updated source file data for '{full_name}'.")
            else:
                print(f"Warning: Could not find '{full_name}' in '{source_file}'")

        except subprocess.CalledProcessError as e:
            print(f"Error running wolframscript for '{full_name}': {e.stderr.strip()}")

    # Write the complete updated master data back to its file
    print(f"\n--- Writing to master file: '{input_path}' ---")
    if polyhedra_data:
        try:
            with open(input_path, "w") as f:
                json.dump(polyhedra_data, f, indent=4)
            print(f"Successfully updated '{input_path}'.")
        except OSError as e:
            print(f"Error: Could not write to file '{input_path}'. Reason: {e}")
    else:
        print(f"No data to write for '{input_path}'. The file was not modified.")

    # Write the complete updated source data back to each source file
    for source_file, data in updated_source_data.items():
        print(f"\n--- Writing to source file: '{source_file}' ---")
        if data:
            try:
                with open(source_file, "w") as f:
                    json.dump(data, f, indent=4)
                print(f"Successfully updated '{source_file}'.")
            except OSError as e:
                print(f"Error: Could not write to file '{source_file}'. Reason: {e}")
        else:
            print(f"No data to write for '{source_file}'. The file was not modified.")


if __name__ == "__main__":
    update_polyhedron_vertices_by_source(sys.argv[1])
