from __future__ import annotations

import math

import numpy as np
import pytest
from dtaidistance import dtw_ndim
from numpy.typing import NDArray

from river_matcher.costs import CostFactory, CostName, DynamicTimeWarpingDistance
from river_matcher.models import JunctionEdge, JunctionGraph

type FloatArray = NDArray[np.float64]
type CostRequest = tuple[int, int, int, int, int]


def make_edge(edge_id: int, u: int, v: int, points: list[tuple[float, float]]) -> JunctionEdge:
    return JunctionEdge(id=edge_id, u=u, v=v, polyline=np.asarray(points, dtype=np.float64))


def make_graphs() -> tuple[JunctionGraph, JunctionGraph]:
    source = JunctionGraph(name="source", coordinates={1: (0.0, 0.0), 2: (2.0, 0.0)},
                           edges=(make_edge(0, 1, 2, [(0.0, 0.0), (2.0, 0.0), ]), make_edge(1, 1, 2, [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), ]),))
    target = JunctionGraph(name="target", coordinates={10: (0.0, 0.0), 20: (1.0, 1.0), 30: (2.0, 0.0), 40: (0.0, 1.0), 50: (2.0, 1.0), 60: (10.0, 0.0), 70: (11.0, 0.0)},
                           edges=(make_edge(10, 10, 20, [(0.0, 0.0), (1.0, 1.0), ]), make_edge(11, 20, 30, [(1.0, 1.0), (2.0, 0.0), ]),
                                  make_edge(12, 40, 50, [(0.0, 1.0), (2.0, 1.0), ]), make_edge(13, 60, 70, [(10.0, 0.0), (11.0, 0.0), ]),))

    return source, target


def make_graphs() -> tuple[JunctionGraph, JunctionGraph]:
    source = JunctionGraph(name="source", coordinates={1: (0.0, 0.0), 2: (2.0, 0.0)},
                           edges=(make_edge(0, 1, 2, [(0.0, 0.0), (2.0, 0.0), ]), make_edge(1, 1, 2, [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), ]),))
    target = JunctionGraph(name="target", coordinates={10: (0.0, 0.0), 20: (1.0, 1.0), 30: (2.0, 0.0), 40: (0.0, 1.0), 50: (2.0, 1.0), 60: (10.0, 0.0), 70: (11.0, 0.0)},
                           edges=(make_edge(10, 10, 20, [(0.0, 0.0), (1.0, 1.0), ]), make_edge(11, 20, 30, [(1.0, 1.0), (2.0, 0.0), ]),
                                  make_edge(12, 40, 50, [(0.0, 1.0), (2.0, 1.0), ]), make_edge(13, 60, 70, [(10.0, 0.0), (11.0, 0.0), ]),))

    return source, target


def make_cost(*, rho: float = 2.0, edge_samples: int = 12, curve_samples: int = 33, warping_window: int | None = 8, normalize: bool = True, ) -> DynamicTimeWarpingDistance:
    source, target = make_graphs()
    cost = CostFactory(source, target).create(CostName.DYNAMIC_TIME_WARPING_DISTANCE, rho=rho, edge_samples=edge_samples, curve_samples=curve_samples,
        warping_window=warping_window, normalize=normalize, )
    assert isinstance(cost, DynamicTimeWarpingDistance)
    return cost

