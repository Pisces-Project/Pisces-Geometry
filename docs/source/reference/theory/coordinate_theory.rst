.. _theory:
=========================
Geometric Theory Overview
=========================

In the context of the Pisces-Geometry package, a **coordinate system** refers to a coordinate system covering Euclidean
space. In such a coordinate system, each point :math:`p` in :math:`\mathbb{R}^N` is described uniquely by a set of **coordinate functions**
:math:`x^i: \mathbb{R}^N \to \mathbb{R}` such that the **coordinate map**

.. math::

    {\bf x} := (x^1(p),\ldots,x^N(p))

is a bijective map from :math:`\mathbb{R}^N \to \mathbb{R}^N`.

.. note::

    We intend to eventually expand the code base to be operable for general relativistic operations which would require
    loosening the restriction of Euclidean space.

Basis Vectors, Lame Coefficients, and the Metric
------------------------------------------------

Each point in the Euclidean space can be described abstractly by a vector :math:`{\bf r}` pointing to its position. In `orthogonal
coordinate systems <https://en.wikipedia.org/wiki/Orthogonal_coordinates>`_
(the class of coordinates treated in Pisces-Geometry), we define a **local-basis** at each point :math:`p \in \mathbb{R}^N` as

.. math::

    {\bf e}_i = \frac{\partial {\bf r}}{\partial x^i}.

It is a fact that these basis vectors span :math:`\mathbb{R}^N` as a vector space and form a valid, orthogonal. The `dual space <https://en.wikipedia.org/wiki/Dual_space>`_
to :math:`\mathbb{R}^N` is (as a special property of Euclidean space) also :math:`\mathbb{R}^N`. Likewise, the **local-basis**
induces a **local dual basis** (also called the contravariant basis) at each point such that

.. math::

    {\bf e}^i({\bf e}_j) = \delta_j^i.

Now, since :math:`\mathbb{R}^N` is an `inner product space <https://en.wikipedia.org/wiki/Inner_product_space>`_, the action of
a contravariant vector is well defined:

.. math::

    {\bf e}^i({\bf e}_j) = \left<{\bf e}^i,{\bf e}_j\right>.

Because the basis is orthogonal, this holds in almost all cases; however, the local basis is not necessarily a unit basis
and therefore

.. math::

    \left<{\bf e}^i,{\bf e}_i\right> = |{\bf e}^i||{\bf e}_i| = 1

We therefore define the **Lame-Coefficients** of the coordinate system as

.. math::

    h_i({\bf r}) = |{\bf e}_i|.

In which case, the contravariant basis vectors are

.. math::

    {\bf e}^i = \frac{1}{h_i^2} {\bf e}_i = \frac{1}{h_i} \hat{\bf e}_i.

We therefore have the foundations of the coordinate system in terms of the **local-basis** and the **covariant / contravariant**
vectors.

The Metric Tensor
+++++++++++++++++

In any coordinate system, the **metric tensor** describes the relationship between physical distance and changes in the
coordinate functions. Thus, for a coordinate system :math:`(x^1,x^2,\ldots,x^N)`, there is a metric tensor :math:`g_{\mu \nu}`
with the property that

.. math::

    ds^2 = g_{\mu \nu} dx^{\mu} dx^{\nu},

In a Euclidean space, the metric is defined as

.. math::

    g_{\mu \nu} = {\bf e}_\mu \cdot {\bf e}_{\nu},

which immediately furnishes the fact that :math:`g_{\mu \nu}` is symmetric and diagonal in orthogonal coordinates. Likewise,
we have previously established that

.. math::

    {\bf e}_{\mu} \cdot {\bf e}_{\mu} = g_{\mu\mu} = h_\mu^2.

Thus, the **differential behavior** of a coordinate system is uniquely dictated by the metric tensor.

Differential Geometry
---------------------

One of the critical aspects of doing physics in a generic coordinate system is resolving the meaning of differential operations
like the **gradient**, **divergence**, or **laplacian**. In Pisces-Geometry, knowledge of the coordinate system's metric is
used directly to perform these operations.

In any coordinate system, the gradient of a function :math:`f: \mathbb{R}^N \to \mathbb{R}` should be a vector with the property
that

.. math::

    \nabla f \cdot d{\bf r} = df = \partial_i f dx^i.

Now,

.. math::

    d{\bf r} = dx^i \partial_i {\bf r} = dx^i {\bf e}_i,

so

.. math::

    \nabla f \cdot d{\bf r} = (\nabla_j f)(dx^i) {\bf e^j}\cdot{\bf e_i} = \nabla_i f \; dx^i = \partial_i f dx^i,

thus,

.. math::

    \nabla_i f = \partial_i f \implies \nabla f = \boxed{{\bf e}^i \partial_i}.

Similarly,

.. math::

    \begin{aligned}
    \nabla \cdot {\bf F} &= \frac{1}{J} \partial_k \left[JF_k\right]\\
    \nabla \times {\bf F} &= \frac{{\bf e}_k}{J} \epsilon_{ijk} \partial_i F^j\\
    \nabla^2 \phi &=  \frac{1}{J} \partial_k \left[\frac{J}{h_k^2} \partial_k \phi\right].
    \end{aligned}
