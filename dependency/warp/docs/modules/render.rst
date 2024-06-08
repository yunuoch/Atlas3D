warp.render
===============================

.. currentmodule:: warp.render

The ``warp.render`` module provides a set of renderers that can be used for visualizing scenes involving shapes of various types.

Built on top of these stand-alone renderers, the ``warp.sim.render`` module provides renderers that can be used to visualize scenes directly from ``warp.sim.ModelBuilder`` objects and update them from ``warp.sim.State`` objects.

..
    .. toctree::
    :maxdepth: 2

Stand-alone renderers
---------------------

The ``OpenGLRenderer`` provides an interactive renderer to play back animations in real time, the ``UsdRenderer`` provides a renderer that exports the scene to a USD file that can be rendered in a renderer of your choice.

.. autoclass:: UsdRenderer
    :members:


.. autoclass:: OpenGLRenderer
    :members:

Simulation renderers
--------------------

Based on these renderers from ``warp.render``, the ``SimRendererUsd`` (which equals ``SimRenderer``) and ``SimRendererOpenGL`` classes from ``warp.sim.render`` are derived to populate the renderers directly from ``warp.sim.ModelBuilder`` scenes and update them from ``warp.sim.State`` objects.

.. currentmodule:: warp.sim.render

.. autoclass:: SimRendererUsd
    :members:

.. autoclass:: SimRendererOpenGL
    :members:

