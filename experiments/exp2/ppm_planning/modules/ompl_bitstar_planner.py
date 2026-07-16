from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from modules.planner import PlanningResult
from modules.ppm_validity import PPMValidityChecker

if TYPE_CHECKING:
    from modules.graph_manager import GraphManager


_OMPL_IMPORT_ERROR = (
    "OMPL Python bindings are not installed. Install ompl in Linux/WSL with: "
    "python -m pip install ompl"
)


class OmplBitstarPPMPlanner:
    def __init__(
        self,
        graph_manager: GraphManager,
        validity_checker: PPMValidityChecker,
        time_limit_s: float,
        range_m: float | None = None,
        random_seed: int | None = None,
        simplify_solution: bool = True,
        state_validity_resolution: float | None = None,
    ) -> None:
        self.graph_manager = graph_manager
        self.validity_checker = validity_checker
        self.time_limit_s = float(time_limit_s)
        self.range_m = None if range_m is None else float(range_m)
        self.random_seed = None if random_seed is None else int(random_seed)
        self.simplify_solution = bool(simplify_solution)
        self.state_validity_resolution = (
            None if state_validity_resolution is None else float(state_validity_resolution)
        )
        self._seed_applied = False

    def plan(self, start: tuple[float, float], goal: tuple[float, float]) -> PlanningResult:
        started_at = time.perf_counter()
        try:
            ompl_modules = self._import_ompl()
            if ompl_modules is None:
                raise RuntimeError(_OMPL_IMPORT_ERROR)
            ob, og, ou = ompl_modules

            if self.random_seed is not None and not self._seed_applied:
                self._set_ompl_seed(ou, self.random_seed)
                self._seed_applied = True

            if not self.validity_checker.is_point_free(start):
                raise RuntimeError("Start point is outside PPM free space.")
            if not self.validity_checker.is_point_free(goal):
                raise RuntimeError("Goal point is outside PPM free space.")

            space = self._make_state_space(ob)
            setup = og.SimpleSetup(space)
            si = setup.getSpaceInformation()
            setup.setStateValidityChecker(
                lambda state: self.validity_checker.is_point_free(
                    (float(state[0]), float(state[1]))
                )
            )
            if self.state_validity_resolution is not None:
                si.setStateValidityCheckingResolution(self.state_validity_resolution)

            motion_validator = self._try_set_motion_validator(ob, si)

            start_state = space.allocState()
            start_state[0] = float(start[0])
            start_state[1] = float(start[1])
            goal_state = space.allocState()
            goal_state[0] = float(goal[0])
            goal_state[1] = float(goal[1])
            setup.setStartAndGoalStates(start_state, goal_state)

            planner = og.BITstar(si)
            self._try_set_range(planner)
            setup.setPlanner(planner)

            solved = setup.solve(self.time_limit_s)
            if not solved:
                raise RuntimeError("OMPL BITstar failed to find a solution.")

            raw_path = self._extract_path_xy(setup.getSolutionPath())
            chosen_path = self._choose_valid_solution_path(setup, raw_path)
            path_length = self._path_length(chosen_path)
            _ = motion_validator  # Keep the Python motion validator alive through solve/extraction.

            elapsed = time.perf_counter() - started_at
            return PlanningResult(
                success=1,
                time_ms=1000.0 * elapsed,
                path_length=path_length,
                path_xy=chosen_path,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started_at
            return PlanningResult(
                success=0,
                time_ms=1000.0 * elapsed,
                path_length=None,
                path_xy=None,
                error=str(exc),
            )

    def _import_ompl(self) -> tuple[Any, Any, Any | None] | None:
        try:
            from ompl import base as ob
            from ompl import geometric as og
        except ImportError:
            return None

        try:
            from ompl import util as ou
        except ImportError:
            ou = None
        return ob, og, ou

    def _set_ompl_seed(self, ou: Any | None, seed: int) -> None:
        if ou is None or not hasattr(ou, "RNG"):
            return
        rng_cls = ou.RNG
        if hasattr(rng_cls, "setSeed"):
            rng_cls.setSeed(seed)
        elif hasattr(rng_cls, "setLocalSeed"):
            rng_cls.setLocalSeed(seed)

    def _make_state_space(self, ob: Any) -> Any:
        ppm_bounds = self.validity_checker.bounds
        if ppm_bounds is None:
            raise RuntimeError("PPMValidityChecker.build() must be called before planning.")

        space = ob.RealVectorStateSpace(2)
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(0, float(ppm_bounds.min_x))
        bounds.setHigh(0, float(ppm_bounds.max_x))
        bounds.setLow(1, float(ppm_bounds.min_y))
        bounds.setHigh(1, float(ppm_bounds.max_y))
        space.setBounds(bounds)
        return space

    def _try_set_motion_validator(self, ob: Any, si: Any) -> Any | None:
        try:
            validity_checker = self.validity_checker

            class PPMMotionValidator(ob.MotionValidator):
                def __init__(self, space_information: Any) -> None:
                    super().__init__(space_information)

                def checkMotion(self, *args: Any) -> bool:
                    if len(args) < 2:
                        return False
                    state0 = args[0]
                    state1 = args[1]
                    return validity_checker.is_segment_free(
                        (float(state0[0]), float(state0[1])),
                        (float(state1[0]), float(state1[1])),
                    )

            motion_validator = PPMMotionValidator(si)
            si.setMotionValidator(motion_validator)
            return motion_validator
        except Exception:
            return None

    def _try_set_range(self, planner: Any) -> None:
        if self.range_m is None:
            return
        setter = getattr(planner, "setRange", None)
        if setter is not None:
            setter(self.range_m)

    def _extract_path_xy(self, path: Any) -> np.ndarray:
        states = path.getStates()
        path_xy = np.array([[float(state[0]), float(state[1])] for state in states], dtype=float)
        if path_xy.ndim != 2 or path_xy.shape[1] != 2 or len(path_xy) < 2:
            raise RuntimeError("OMPL BITstar produced an invalid path.")
        return path_xy

    def _choose_valid_solution_path(self, setup: Any, raw_path: np.ndarray) -> np.ndarray:
        if self.simplify_solution:
            setup.simplifySolution()
            simplified_path = self._extract_path_xy(setup.getSolutionPath())
            if self._validate_path_segments(simplified_path) is None:
                return simplified_path

        raw_error = self._validate_path_segments(raw_path)
        if raw_error is None:
            return raw_path
        raise RuntimeError(raw_error)

    def _validate_path_segments(self, path_xy: np.ndarray) -> str | None:
        for index in range(len(path_xy) - 1):
            p0 = (float(path_xy[index, 0]), float(path_xy[index, 1]))
            p1 = (float(path_xy[index + 1, 0]), float(path_xy[index + 1, 1]))
            if not self.validity_checker.is_segment_free(p0, p1):
                return f"OMPL BITstar path has an invalid PPM segment at index {index}."
        return None

    def _path_length(self, path_xy: np.ndarray) -> float:
        return float(
            sum(
                np.linalg.norm(path_xy[index + 1] - path_xy[index])
                for index in range(len(path_xy) - 1)
            )
        )

