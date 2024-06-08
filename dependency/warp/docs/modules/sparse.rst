warp.sparse
===============================

.. currentmodule:: warp.sparse

..
    .. toctree::
    :maxdepth: 2

Warp includes a sparse linear algebra module ``warp.sparse`` that implements some common sparse matrix operations that arise in simulation.

Sparse Matrices
-------------------------

Currently `warp.sparse` supports Block Sparse Row (BSR) matrices, the BSR format can also be used to represent Compressed Sparse Row (CSR) matrices as a special case with a 1x1 block size.

.. automodule:: warp.sparse
    :members:


Iterative Linear Solvers
------------------------

.. currentmodule:: warp.optim.linear

Warp provides a few common iterative linear solvers (:func:`cg`, :func:`bicgstab`, :func:`gmres`) with optional preconditioning.

.. note:: While primarily intended to work with sparse matrices, those solvers also accept dense linear operators provided as 2D Warp arrays.
    It is also possible to provide custom operators, see :class:`LinearOperator`.

.. automodule:: warp.optim.linear
    :members:
