"""
PyTorch DataLoader for streaming GeoWord geological histories. 
"""

import os
from numbers import Integral

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, get_worker_info

# Two types of geological model generators provided
from geogen.generation.model_generators import (
    MarkovGeostoryGenerator,
    BlockedGeostoryGenerator,
)

_DEFAULT_GENERATOR_CLASS = MarkovGeostoryGenerator


class GeoData3DStreamingDataset(Dataset):
    """
    A PyTorch Dataset wrapper for streaming geological data from a GeostoryGenerator object.

    Parameters
    ----------
    model_bounds : tuple
        Bounds of the model as ((xmin, xmax), (ymin, ymax), (zmin, zmax)).
    model_resolution : tuple
        Resolution of the model as (x_res, y_res, z_res).
    generator_config : str
        A path to a configuration file for the particular generator class.
    dataset_size : int
        The total number of samples in one epoch.
    device : str
        Torch device where data is loaded.
    """

    def __init__(
        self,
        model_bounds=((-3840, 3840), (-3840, 3840), (-1920, 1920)),
        model_resolution=(256, 256, 128),
        generator_config=None,
        dataset_size=int(1e6),
        device="cpu",
        transform=None,  # Add the transform parameter
    ):
        self.model_generator = _DEFAULT_GENERATOR_CLASS(
            model_bounds=model_bounds,
            model_resolution=model_resolution,
            config=generator_config,
        )
        self.device = device
        self.size = dataset_size
        self.transform = transform  # Store the transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        model = self.model_generator.generate_model()
        model.fill_nans()
        data = model.get_data_grid()
        data_tensor = torch.from_numpy(data).float().unsqueeze(0).to(self.device)

        if self.transform:
            data_tensor = self.transform(data_tensor)  # Apply the transform

        return data_tensor


class OneHotTransform:
    def __init__(self, num_classes=15, min_val=-1):
        """
        Args:
            num_classes (int): Number of classes for one-hot encoding.
            min_val (int): Minimum value in the tensor.
        """
        self.num_classes = num_classes
        self.min_val = min_val

    def __call__(self, sample):
        """
        Args:
            sample (torch.Tensor): Input tensor of shape (C, X, Y, Z) with C=1.
        Returns:
            torch.Tensor: One-hot encoded tensor of shape (num_classes, X, Y, Z).
        """
        # Ensure the tensor has a single channel
        if sample.shape[0] != 1:
            raise ValueError(f"Expected channel dimension to be 1, but got {sample.shape[0]}")

        tensor = sample.squeeze(0)  # Shape: (X, Y, Z)
        tensor_shifted = tensor - self.min_val  # Shift to start from 0
        tensor_shifted = torch.clamp(tensor_shifted, 0, self.num_classes - 1)  # Clamp to range
        tensor_shifted = tensor_shifted.long()  # Convert to long for indexing

        # One-hot encode: (X, Y, Z) -> (X, Y, Z, num_classes)
        one_hot = F.one_hot(tensor_shifted, num_classes=self.num_classes)  # Shape: (X, Y, Z, num_classes)
        one_hot = one_hot.permute(3, 0, 1, 2).contiguous().float()

        return one_hot


# New class for taking in blockedgeostorygenerator inputs
class BlockedGeoData3DStreamingDataset(Dataset):
    """
    Stream geological histories generated as BlockedGenerator objects as CPU tensors of shape (1, X, Y, Z)

    Parameters
    ----------
    generator_config :
        is a parsed object with model, seed and blocks attributes from a configuration JSON
    catalog :
        geological component process registry required by BlockedGeostoryGenerator
    Bounds : tuple
        physical bounds of the model
    Resolution : tuple
        resolution default to the configuration
    dataset_size : int
        defines an artificial epoch length
    
    Notes
    ----------
    idx is ignored and every access generates a new model.
    Each worker gets a separate generator RNG. For repeatable streams, seed
    PyTorch (or the DataLoader generator) and the geological configuration.
    Exact restart reproducibility and components using other RNGs require
    additional handling outside this dataset.
    """

    def __init__(
        self,
        model_bounds=((-3840, 3840), (-3840, 3840), (-1920, 1920)),
        model_resolution=(256, 256, 128),
        generator_config=None,
        dataset_size=int(1e6),
        device="cpu",
        transform=None,
        *,
        catalog,
    ):
        if not all(hasattr(generator_config, name) for name in ("model", "seed", "blocks")):
            raise TypeError("generator_config must be a parsed blocked-generator configuration")
        if catalog is None or not callable(getattr(catalog, "get", None)):
            raise TypeError("catalog must provide get(component_id)")
        if isinstance(dataset_size, bool) or not isinstance(dataset_size, Integral) or dataset_size < 1:
            raise ValueError("dataset_size must be a positive integer")
        if torch.device(device).type != "cpu":
            raise ValueError("Generate on CPU; let the training loop transfer batches to the GPU")

        self.config = generator_config
        self.catalog = catalog
        self.model_bounds = (
            generator_config.model.bounds if generator_config.model.bounds is not None else model_bounds
        )
        self.model_resolution = tuple(
            generator_config.model.resolution if generator_config.model.resolution is not None else model_resolution
        )
        if len(self.model_resolution) != 3 or any(
            isinstance(n, bool) or not isinstance(n, Integral) or n < 1
            for n in self.model_resolution
        ):
            raise ValueError("model_resolution must contain three positive integers")
        if len(self.model_bounds) != 3 or any(
            len(axis) != 2 or not np.isfinite(axis).all() or axis[0] >= axis[1]
            for axis in self.model_bounds
        ):
            raise ValueError("model_bounds must contain three finite (minimum, maximum) pairs")
        self.size = int(dataset_size)
        self.transform = transform
        self.model_generator = None
        self._generator_context = None

    def _get_generator(self):
        worker = get_worker_info()
        worker_seed = worker.seed if worker is not None else torch.initial_seed()
        context = (os.getpid(), worker_seed)
        if self.model_generator is None or self._generator_context != context:
            generator = BlockedGeostoryGenerator(
                model_bounds=self.model_bounds,
                model_resolution=self.model_resolution,
                config=self.config,
                registry=self.catalog,
            )
            seeds = [worker_seed]
            if self.config.seed is not None:
                seeds.append(self.config.seed)
            generator.rng = np.random.default_rng(np.random.SeedSequence(seeds))
            self.model_generator = generator
            self._generator_context = context
        return self.model_generator

    def __getstate__(self):
        state = self.__dict__.copy()
        state["model_generator"] = None
        state["_generator_context"] = None
        return state

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        model = self._get_generator().generate_model()
        model.fill_nans()
        data = model.get_data_grid()
        if tuple(data.shape) != self.model_resolution:
            raise ValueError(
                f"Generated grid shape {data.shape} differs from requested {self.model_resolution}"
            )
        data_tensor = torch.from_numpy(data).float().unsqueeze(0)

        if self.transform:
            data_tensor = self.transform(data_tensor)  # Apply the transform

        return data_tensor
    
