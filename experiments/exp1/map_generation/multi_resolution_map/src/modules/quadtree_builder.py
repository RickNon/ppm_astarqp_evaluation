from dataclasses import dataclass
from typing import Callable

from .geometry2d import AARect
from .occupancy_classifier import OccupancyClassifier, CellState
from .quadtree_node import QuadtreeNode


@dataclass
class QuadtreeBuildConfig:
    min_resolution: float
    max_depth: int
    treat_mixed_as_occupied_at_min_resolution: bool = True


class QuadtreeBuilder:
    def __init__(
        self,
        classifier: OccupancyClassifier,
        config: QuadtreeBuildConfig,
    ) -> None:
        self._classifier = classifier
        self._config = config
        self._root: QuadtreeNode | None = None

    def build(
        self,
        root_rect: AARect,
        recorder=None,
        stop_condition: Callable[[], bool] | None = None,
    ) -> QuadtreeNode:
        self._root = QuadtreeNode(rect=root_rect, depth=0)
        self._classify_node(self._root)
        leaf_frontier: list[QuadtreeNode] = [self._root]

        if recorder is not None:
            # Record the root-only frontier before any depth expansion.
            recorder.record_from_leaves(list(leaf_frontier))
            if stop_condition is not None and stop_condition():
                return self._root

        current_level: list[QuadtreeNode] = [self._root]
        while current_level:
            # Expand all splittable cells at the same depth before moving deeper.
            splittable_nodes = [node for node in current_level if self._is_splittable(node)]
            if not splittable_nodes:
                break

            next_level: list[QuadtreeNode] = []
            for node in splittable_nodes:
                leaf_frontier.remove(node)
                node.split()
                for child in node.children:
                    self._classify_node(child)
                    next_level.append(child)
                    leaf_frontier.append(child)

            if recorder is not None and self._root is not None:
                # Record the updated leaf frontier after each depth-wide expansion.
                recorder.record_from_leaves(list(leaf_frontier))
                if stop_condition is not None and stop_condition():
                    break

            current_level = next_level

        return self._root

    def collect_leaves(self, node: QuadtreeNode) -> list[QuadtreeNode]:
        # Collect all final leaf cells in depth-first order.
        if node.is_leaf():
            return [node]

        leaves: list[QuadtreeNode] = []
        for child in node.children:
            leaves.extend(self.collect_leaves(child))
        return leaves

    def _classify_node(self, node: QuadtreeNode) -> None:
        state = self._classifier.classify(node.rect)
        node.state = state

        # Stop immediately for free or occupied cells.
        if state in (CellState.FREE, CellState.OCCUPIED):
            return

        # If the node's occupancy is mixed,
        # Check whether we have reached the minimum resolution or maximum depth limit.
        cell_size = node.size()
        reached_min_resolution = cell_size <= self._config.min_resolution
        reached_max_depth = node.depth >= self._config.max_depth

        if reached_min_resolution or reached_max_depth:
            # Force the final ambiguous cell to occupied for safety.
            if self._config.treat_mixed_as_occupied_at_min_resolution:
                node.state = CellState.OCCUPIED
            return

    def _is_splittable(self, node: QuadtreeNode) -> bool:
        return (
            node.state == CellState.MIXED
            and node.size() > self._config.min_resolution
            and node.depth < self._config.max_depth
        )
