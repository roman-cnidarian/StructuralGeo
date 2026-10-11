""" Categorical definitions for sampling of broad categories of GeoWords."""

__all__ = [
    "BaseStrata",
    "Sediment",
    "Erosion",
    "Dike",
    "Sills",
    "Pluton",
    "OreDeposit",
    "Fold",
    "Fault",
    "Slip",
    "Mountains",
    "End",
]

import warnings
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Callable, ClassVar

import numpy as np

from geogen.generation.geowords import *
from geogen.model import CompoundProcess, GeoProcess

@dataclass(frozen=True)
class EventCaseSpec:
    key: str
    label: str
    default_probability: float
    component_factories: tuple[Callable, ...]
    @property
    def component_names(self) -> list[str]:
        return [
            factory.__name__
            for factory in self.component_factories
        ]

class _EventTemplateClass(GeoWord):
    """
    A special case of GeoWord that selects from a set of cases with associated probabilities.
    This class is used to form more general categories of events that can be sampled from.

    The Events include a name, probability of selection, and a sequence of actions (GeoWords or GeoProcesses)
    that form the history of the event.

    Parameters
    ----------
    cases : List[Event]
        A list of Events to sample from. Each event should have a name, probability, and a sequence of GeoWords or GeoProcesses.
    rng : Optional[np.random.Generator]
        A random number generator for reproducibility.
    """

    Event = namedtuple("Case", ["name", "p", "processes"])

    CASE_SPECS: ClassVar[tuple[EventCaseSpec, ...]] = ()

    def __init__(self, probabilities: dict[str, float] | None = None, seed=None):
        super().__init__(seed)
        if not self.CASE_SPECS:
            raise ValueError(
                f"{self.__class__.__name__} defines no cases"
            )
        defaults = {
            case.key: case.default_probability
            for case in self.CASE_SPECS
        }
        selected_probabilities = (
            defaults
            if probabilities is None
            else probabilities
        )
        # check if the input keys agree with what the word expects
        expected = set(defaults)
        received = set(selected_probabilities)
        if received != expected:
            raise ValueError(
                f"Probability keys must be {sorted(expected)}; "
                f"received {sorted(received)}"
            )
        # isolate probability values
        values = {
            key: float(value)
            for key, value in selected_probabilities.items()
        }
        probability_array=np.asarray(
                [
                    values[case.key]
                    for case in self.CASE_SPECS
                ],
            dtype=float,
        )
        # run some checks on probability values
        if not np.all(np.isfinite(probability_array)):
            raise ValueError("Probabilities must be finite")
        if np.any(probability_array < 0):
            raise ValueError("Probabilities cannot be negative")
        total = float(probability_array.sum())
        if not np.isclose(total, 1.00):
            raise ValueError(
                f"Probabilities sum to {total: 6f}, not 1"
            )
        # convert to list of event class objects
        self.cases = [
            self.Event(
                name=case.label,
                p=values[case.key],
                processes=[
                    factory() for factory in case.component_factories
                ],
            )
            for case in self.CASE_SPECS
        ]
        self.selected_case = None
        self.probabilities = probability_array
        self._validate_cases()

    @classmethod
    def case_metadata(cls) -> list[dict]:
        return [
            {
                "key": case.key,
                "label": case.label,
                "default_probability": case.default_probability,
                "components": case.component_names,
            }
            for case in cls.CASE_SPECS
        ]

    def generate(self):
        """
        Generate the geological history by selecting a case and building the corresponding history.

        Returns
        -------
        geo.CompoundProcess
            A sampled geological history snippet with a CompoundProcess wrapper.
        """
        self.hist.clear()
        self.build_history()
        name = f"{self.__class__.__name__}: {self.selected_case.name}"
        geoprocess = CompoundProcess(self.hist.copy(), name=name)
        return geoprocess

    def build_history(self):
        """
        Randomly select a case based on probabilities and build the corresponding history.
        """
        assert self.probabilities is not None, "Probabilities are not defined."
        selected_index = self.rng.choice(len(self.cases), p=self.probabilities)
        self.selected_case = self.cases[selected_index]
        self.add_process(self.selected_case.processes)

    def _validate_cases(self):
        """
        Ensure that the case list is correctly defined with valid types.
        """
        if not self.cases:
            raise ValueError("Cases are not defined.")

        for case in self.cases:
            if not isinstance(case.name, str):
                raise TypeError(f"Case name must be a string, got {type(case.name).__name__}.")
            if not isinstance(case.p, float):
                raise TypeError(f"Case probability must be a float, got {type(case.p).__name__}.")
            if not isinstance(case.processes, list):
                raise TypeError(f"Case processes must be a list, got {type(case.processes).__name__}.")
            for process in case.processes:
                if not isinstance(process, (GeoProcess, GeoWord)):
                    raise TypeError(
                        f"Processes must be instances of GeoProcess or GeoWord, got {type(process).__name__}."
                    )

    # def _validate_probabilities(self):
    #     """
    #     Ensure that the probabilities sum to 1 and renormalize if necessary.
    #     """
    #     probabilities = np.array([case.p for case in self.cases])
    #     sum_prob = np.sum(probabilities)

    #     if not np.isclose(sum_prob, 1.0):
    #         warnings.warn(
    #             f"{self.__class__.__name__}: Probabilities sum to {sum_prob:.4f}, but should sum to 1.0. Renormalizing.",
    #             RuntimeWarning,
    #         )
    #         probabilities = np.array(probabilities) / sum_prob

    #     self.probabilities = probabilities

# class of categorical events used in the original implementation of StructuralGeo

class Sediment(_EventTemplateClass):
    """
    A sampling regime for sediment events.
    """

    CASE_SPECS = (
        EventCaseSpec(
            key="fine",
            label="Fine",
            default_probability=0.4,
            component_factories=(FineRepeatSediment,),
        ),
        EventCaseSpec(
            key="coarse",
            label="Coarse",
            default_probability=0.5,
            component_factories=(CoarseRepeatSediment,),
        ),
        EventCaseSpec(
            key="single",
            label="Single",
            default_probability=0.1,
            component_factories=(SingleRandSediment,),
        ),
    )

class BaseStrata(_EventTemplateClass):
    """
    A sampling regime for base strata.
    """
    CASE_SPECS = (
        EventCaseSpec(
            key="basement",
            label="Basement",
            default_probability=0.27,
            component_factories=(
                InfiniteBasement,
                Sediment,
            ),
        ),
        EventCaseSpec(
            key="sediment_markov",
            label="Sediment: Markov",
            default_probability=0.22,
            component_factories=(
                InfiniteSedimentMarkov,
            ),
        ),
        EventCaseSpec(
            key="sediment_uniform",
            label="Sediment: Uniform",
            default_probability=0.22,
            component_factories=(
                InfiniteSedimentUniform,
            ),
        ),
        EventCaseSpec(
            key="sediment_tilted",
            label="Sediment: Tilted Markov",
            default_probability=0.29,
            component_factories=(
                InfiniteSedimentTilted,
            ),
        ),
    )

class Erosion(_EventTemplateClass):
    """
    A sampling regime for erosion events.
    """

    CASE_SPECS = (
        EventCaseSpec(
            key="flat",
            label="Flat",
            default_probability=0.20,
            component_factories=(
                FlatUnconformity,
            ),
        ),
        EventCaseSpec(
            key="tilted",
            label="Tilted",
            default_probability=0.25,
            component_factories=(
                TiltedUnconformity,
            ),
        ),
        EventCaseSpec(
            key="tilt_cut_fill",
            label="Tilt cut-and-fill",
            default_probability=0.20,
            component_factories=(
                TiltCutFill,
            ),
        ),
        EventCaseSpec(
            key="wave",
            label="Wave",
            default_probability=0.35,
            component_factories=(
                WaveUnconformity,
            ),
        ),
    )


class Dike(_EventTemplateClass):
    """
    A sampling regime for intrusion events.
    """

    CASE_SPECS = (
        EventCaseSpec(
            key="dike",
            label="Dike",
            default_probability=0.4,
            component_factories=(
                DikePlaneWord,
            ),
        ),
        EventCaseSpec(
            key="warped_dike",
            label="Warped dike",
            default_probability=0.4,
            component_factories=(
                SingleDikeWarped,
            ),
        ),
        EventCaseSpec(
            key="dike_group",
            label="Dike group",
            default_probability=0.2,
            component_factories=(
                DikeGroup,
            ),
        ),
    )


class Sills(_EventTemplateClass):
    """
    A sampling regime for sill events.
    """

    CASE_SPECS = (
        EventCaseSpec(
            key="single_sill",
            label="Single sill",
            default_probability=0.2,
            component_factories=(
                SillWord,
            ),
        ),
        EventCaseSpec(
            key="sill_system",
            label="Sill system",
            default_probability=0.8,
            component_factories=(
                SillSystem,
            ),
        ),
    )


class Pluton(_EventTemplateClass):
    """
    A sampling regime for volcanic events.
    """

    CASE_SPECS = (
        EventCaseSpec(
            key="laccolith",
            label="Laccolith",
            default_probability=0.4,
            component_factories=(
                Laccolith,
            ),
        ),
        EventCaseSpec(
            key="lopolith",
            label="Lopolith",
            default_probability=0.6,
            component_factories=(
                Lopolith,
            ),
        ),
    )

class OreDeposit(_EventTemplateClass):
    """
    A sampling regime for ore deposit events.

    Can be expanded to include more types in the future
    """

    CASE_SPECS = (
        EventCaseSpec(
            key="blob_cluster",
            label="Blob cluster",
            default_probability=1.0,
            component_factories=(
                BlobCluster,
            ),
        ),
    )


class Fold(_EventTemplateClass):
    """
    A sampling regime for folding events.
    """

    CASE_SPECS = (
        EventCaseSpec(
            key="simple",
            label="Simple",
            default_probability=0.2,
            component_factories=(
                SimpleFold,
            ),
        ),
        EventCaseSpec(
            key="shaped",
            label="Shaped",
            default_probability=0.3,
            component_factories=(
                ShapedFold,
            ),
        ),
        EventCaseSpec(
            key="fourier",
            label="Fourier",
            default_probability=0.5,
            component_factories=(
                FourierFold,
            ),
        ),
    )


class Fault(_EventTemplateClass):
    """A sampling regime for fault events."""

    CASE_SPECS = (
        EventCaseSpec(
            key="normal",
            label="Normal",
            default_probability=0.10,
            component_factories=(
                FaultNormal,
            ),
        ),
        EventCaseSpec(
            key="reverse",
            label="Reverse",
            default_probability=0.10,
            component_factories=(
                FaultReverse,
            ),
        ),
        EventCaseSpec(
            key="strike_slip",
            label="Strike-slip",
            default_probability=0.35,
            component_factories=(
                FaultStrikeSlip,
            ),
        ),
        EventCaseSpec(
            key="horst_graben",
            label="Horst and graben",
            default_probability=0.10,
            component_factories=(
                FaultHorstGraben,
            ),
        ),
        EventCaseSpec(
            key="fully_random",
            label="Fully random",
            default_probability=0.20,
            component_factories=(
                FaultRandom,
            ),
        ),
        EventCaseSpec(
            key="sequence",
            label="Fault sequence",
            default_probability=0.15,
            component_factories=(
                FaultSequence,
            ),
        ),
    )
        
class Mountains(_EventTemplateClass):
    """A sampling regime for mountain events."""

    CASE_SPECS = (
        EventCaseSpec(
            key="tilted_mountains",
            label="Tilted mountains",
            default_probability=1.0,
            component_factories=(
                TiltedMountains,
            ),
        ),
    )


# TODO: Implement Slip events in GeoWords and add to the Slip class
class Slip(_EventTemplateClass):
    """A sampling regime for slip events."""

    CASE_SPECS = (
        EventCaseSpec(
            key="null",
            label="No slip",
            default_probability=1.0,
            component_factories=(
                NullWord,
            ),
        ),
    )

class End(_EventTemplateClass):
    """An ending flag for the geostory."""

    CASE_SPECS = (
        EventCaseSpec(
            key="termination",
            label="Termination of sequence",
            default_probability=1.0,
            component_factories=(
                NullWord,
            ),
        ),
    )


# Adapted events with exposed probabilities for greater customizability

# class BaseStrataApp(_EventTemplateClass):
#     """
#     A sampling regime for base strata with configurable selection probabilities. 
#     """

#     def __init__(self, seed=None):
#         cases = [
#             self.Event(name="Basement", p=0.27, processes=[InfiniteBasement(), Sediment()]),
#             self.Event(name="Sediment: Markov", p=0.22, processes=[InfiniteSedimentMarkov()]),
#             self.Event(name="Sediment: Uniform", p=0.22, processes=[InfiniteSedimentUniform()]),
#             self.Event(
#                 name="Sediment: Tilted Markov",
#                 p=0.29,
#                 processes=[InfiniteSedimentTilted()],
#             ),
#         ]
#         super().__init__(cases=cases, seed=seed)

