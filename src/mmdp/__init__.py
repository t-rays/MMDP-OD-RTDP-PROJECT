"""MMDP planning with Baseline RTDP and Operator-Decomposition RTDP.

The package is organized in five layers:

- ``mmdp.domain``: the grid MMDP environment, heuristic, and map loading
- ``mmdp.planning``: RTDP/LRTDP engine, both planning domains, and their
  configs, components, and result dataclasses
- ``mmdp.evaluation`` / ``mmdp.resource_monitor``: fixed-policy evaluation
  and resource measurement
- ``mmdp.experiments``: experiment orchestration (schema, factory, runner)
- ``mmdp.analysis``: notebook visualizations
"""

from mmdp.planning.baseline_rtdp import BaselineDomain
from mmdp.planning.components import (
    DeterministicTieBreaker,
    DictValueStore,
    PlanningDomain,
    SetSolvedTracker,
    SolvedTracker,
    TieBreaker,
    ValueStore,
)
from mmdp.planning.config import DeadlineReached, MemoryLimitReached, RTDPConfig
from mmdp.planning.domain_base import RTDPDomainBase
from mmdp.evaluation import EvaluationConfig, MethodPolicyAdapter, evaluate_policy
from mmdp.domain.grid_mmdp import GridMMDP, MMDPConfig
from mmdp.domain.heuristic import ShortestPathHeuristic
from mmdp.domain.map_creator import MapInstance, create_map_instance
from mmdp.planning.od_rtdp import OperatorDecompositionDomain
from mmdp.planning.planner import RTDPPlanner
from mmdp.planning.results import (
    BasePlanningResult,
    ODRTDPPlanningResult,
    RTDPPlanningResult,
    TrialResult,
)

__all__ = [
    "BaselineDomain",
    "BasePlanningResult",
    "DeadlineReached",
    "DeterministicTieBreaker",
    "DictValueStore",
    "EvaluationConfig",
    "GridMMDP",
    "MapInstance",
    "MemoryLimitReached",
    "MethodPolicyAdapter",
    "MMDPConfig",
    "ODRTDPPlanningResult",
    "OperatorDecompositionDomain",
    "PlanningDomain",
    "RTDPConfig",
    "RTDPDomainBase",
    "RTDPPlanner",
    "RTDPPlanningResult",
    "SetSolvedTracker",
    "ShortestPathHeuristic",
    "SolvedTracker",
    "TieBreaker",
    "TrialResult",
    "ValueStore",
    "create_map_instance",
    "evaluate_policy",
]
