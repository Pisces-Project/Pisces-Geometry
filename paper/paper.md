---
title: 'PyMetric: A Library for Geometric Computation'
tags:
  - Python
  - differential geometry
  - modeling
authors:
  - name: Eliza C. Diggins
    orcid: 0009-0005-9389-9098
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Daniel R. Wik
    orcid: 0000-0001-9110-2245
    affiliation: 1
affiliations:
 - name: University of Utah, USA
   index: 1
date: 11 June 2025
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
---

# Summary

PyMetric is a lightweight Python library designed to streamline differential geometry
and vector calculus operations in user-defined coordinate systems, with a focus on
applications in astrophysics and computational physics. In many physical modeling
tasks—especially in astrophysics—it is both natural and advantageous to work in
non-Cartesian coordinate systems that reflect the underlying symmetries of a system.
These coordinate systems may be highly nontrivial, such as ellipsoidal or spheroidal
coordinates, where the ability to express and manipulate tensorial quantities is essential
for deriving accurate and efficient models.

The core design of PyMetric emphasizes a seamless blend of symbolic and numerical computation.
Symbolic expressions are used to derive key geometric quantities such as metric tensors,
Christoffel symbols, and Jacobians, which are then converted into efficient numerical
functions suitable for use in array-based workflows. This enables users to write models
that respect the underlying geometry while retaining the performance and flexibility of
NumPy-style programming.

PyMetric further supports structured grid abstractions and offers extensibility
for multiple coordinate systems, buffer types (including HDF5 storage),
and geometric operations. It automates tasks such as computing gradients, divergences, and Laplacians—critical
for solving PDEs or modeling physical systems—while maintaining awareness of the coordinate system
and metric context. This makes it particularly useful for scientific domains that demand geometric
precision, such as general relativity, magnetohydrodynamics, and planetary dynamics.

By bringing geometric context to array mathematics, PyMetric provides a flexible
and modern foundation for coordinate-aware scientific computing.




# Statement of need

`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
