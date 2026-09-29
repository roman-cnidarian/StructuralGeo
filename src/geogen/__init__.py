# -*- coding: utf-8 -*-

"""
GeoGen Geological Modeling Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GeoGen is a Python library for generating and manipulating synthetic geological models.
Basic usage:

   >>> import geogen
   >>> model = geogen.GeoModel(bounds=((0, 100), (0, 100), (0, 100)), resolution=(100, 100, 50))
   >>> word = geogen.word.SingleDikeWarped()
   >>> history = word.generate()
   >>> model.add_history(history)
   >>> model.compute_model(normalize=True)

You can visualize the model using PyVista or with some built-in visualization tools:

   >>> p = geogen.plot.volview(model) # Returns a PyVista plotter object
   >>> p.show()                       # Show the plotter window
   >>> geogen.plot.categorical_grid_view(model).show()  # Directly display the model

:copyright: (c) 2024 by Simon Ghyselincks.
:license: Apache 2.0, see LICENSE for more details.
"""

__title__ = "GeoGen"

import geogen.model as model
# TODO: Update these imports into a cohesive API
# Note, some very specific imports should be used to avoid importing the entire library
# from geogen.dataset import GeoData3DStreamingDataset as StreamingDataset

# This controls the import behaviour when using `from geogen import *`
__all__ = ["GeoData3DStreamingDataset", "model", "plot", "gen"]

# switched out old import with lazy loading to allow for torch-independent deployment
def __getattr__(name: str):
    if name == "GeoData3DStreamingDataset":
        from geogen.dataset import GeoData3DStreamingDataset

        return GeoData3DStreamingDataset

    raise AttributeError(
        f"Module {__name__!r} has no attribute {name!r}"
    )
