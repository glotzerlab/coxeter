---
title: 'coxeter: A Python package for generating shapes and performing robust and flexible geometric and physical calculations'
tags:
  - Python
  - geometry
  - physics
  - materials science
authors:
  - name: Vyas Ramasubramani
    orcid: 0000-0001-5181-9532
    affiliation: 1
  - name: Bradley Dice
    orcid: 0000-0002-9983-0770
    affiliation: 3
  - name: Tobias Dwyer
    orcid: 0000-0001-6443-7744
    affiliation: 1
  - name: Sharon C. Glotzer
    orcid: 0000-0002-7197-0085
    affiliation: "1, 2, 3, 4"
affiliations:
 - name: Department of Chemical Engineering, University of Michigan
   index: 1
 - name: Department of Materials Science and Engineering, University of Michigan
   index: 2
 - name: Department of Physics, University of Michigan
   index: 3
 - name: Biointerfaces Institute, University of Michigan
   index: 4
date: 8 Dec 2020
bibliography: paper.bib
---

# Summary

The coxeter Python package provides standardized representations of geometric objects.
The primary focus of this library is to work with polytopes in two and three dimensions, for which most calculations are not simple analytical formulas but instead require summations over triangulations or more complex integrals.
The shape classes in coxeter use Python properties to transparently expose many geometric attributes to the user.
These classes are designed to maximize flexibility, allowing various attributes to be set and recomputing the shape definition on the fly.
The package is targeted at applications in physics, so it emphasizes highly robust methods for calculating properties like moments of inertia as well as providing certain less common metrics like mean curvatures that find usage in certain physical calculations.

The library also serves as a repository for the generation of shapes.
While simple classes of shapes like spheres and ellipsoids can be described via a small fixed set of parameters, the definitions of polygons and polyhedra can be arbitrarily long depending on the number of vertices of these shapes.
The shape family API in coxeter provides a flexible way to define and work with collections of related shapes, ranging from enumerable sets like the Platonic solids to continuously defined sets of shapes [@Chen2014].
These different types of shape families are handled using identical APIs, so users can easily switch between shapes that have completely different mathematical definitions using a single line of code.
Additionally, since these shape families generate the coxeter's shape classes, calculating the various attributes of shapes is as simple as querying the properties of objects generated from families.


# Statement of Need

Considerations of shape are becoming increasingly important in materials science as improved synthetic capabilities have allowed the creation of a wide range of anisotropic particles [@Glotzer2007b].
Colloidal science in particular has seen immense growth in this area, and numerous studies have shown that particle shape is an important handle for controlling the self-assembly of colloidal crystals [@Damasceno2012d,@Glotzer2007b,@Chen2014].
Precise modeling of these systems requires robust, reproducible methods for generating shapes and calculating their properties [@Anderson2020,@Ramasubramani2020b,@Allen2006], a gap that coxeter aims to fill.
The shape families in coxeter address the first part, providing a standard method for generating shapes.
A number of such families are bundled into coxeter, but just as importantly, the framework allows users to work with arbitrary lists of shapes by providing an appropriately formatted JSON file, making it trivial to share representations of shapes.
One important feature of the library is a mapping from digital object identifiers (DOIs) to families, so that any user can contribute families associated with published research to make them immediately collectively accessible.
We anticipate that the set of shape families in coxeter will grow over time as users generate and contribute their shape families to coxeter, with the goal of providing a centralized repository for use in reproducing and extending prior research.

The shape classes in coxeter address the second stated requirement, the ability to perform robust calculation of various properties of shapes.
Physics-based models, for instance molecular dynamics simulations based on fundamental laws of motion, can be highly sensitive to the precise values of these quantities, necessitating their accurate evaluation.
Most computational geometry libraries typically focus on solving more complex problems like finding convex hulls, Voronoi tesselations, and Delaunay triangulations [@cgal].
The purpose of coxeter is to provide a standard implementation for simpler calculations such as volumes, bounding spheres, and inertia tensors, for which formulas are generally well-known but often require careful consideration to implement robust and efficient algorithms to compute.
Inertia tensors are a particularly notable example since they can be quite difficult to compute accurately for arbitrary polytopes and require specialized algorithms [@Kallay2006].
coxeter is readily available for installation via pip or using conda-forge.


# Acknowledgements

**Update acknowledgments**

We would like to acknowledge M. Eric Irrgang for prototype implementations of various parts of this code, as well as Bryan VanSaders and James Proctor for collecting the various early prototypes and relevant methods into a single code base.

# References
