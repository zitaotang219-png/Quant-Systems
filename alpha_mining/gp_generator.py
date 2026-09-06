from __future__ import annotations

import hashlib
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import GPConfig
from .dsl import (
    FactorNode,
    add,
    const,
    correlation,
    delay,
    delta,
    div,
    field,
    mul,
    pct_change,
    rank,
    rolling_mean,
    rolling_std,
    sub,
    ts_rank,
    zscore,
)
from .evaluation_types import VERY_BAD_FITNESS
from .hypothesis import HypothesisCandidate, HypothesisEvaluator


GPCandidate = HypothesisCandidate


class GPGenerator:
    def __init__(self, config: GPConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.generation_statistics: list[dict[str, Any]] = []
        self.candidate_audit: list[dict[str, Any]] = []
        self.stage_events: list[dict[str, Any]] = []
        self._archive: dict[str, GPCandidate] = {}

    def generate(
        self,
        panel: pd.DataFrame,
        evaluator: HypothesisEvaluator,
        *,
        deduplicate: bool = True,
    ) -> list[HypothesisCandidate]:
        """Generate unique hypotheses while keeping GP breeding internal."""
        self.evolve(panel, evaluator, deduplicate=deduplicate)
        return self.archive_candidates()

    def evolve(self, panel: pd.DataFrame, evaluator: HypothesisEvaluator, deduplicate: bool = True) -> list[GPCandidate]:
        self._archive = {}
        prepare_panel = getattr(evaluator, "prepare_panel", None)
        if callable(prepare_panel):
            panel = prepare_panel(panel)
        population = [self.random_tree(max_depth=self.config.init_max_depth) for _ in range(self.config.population_size)]
        ids = [self._individual_id(0, slot) for slot in range(len(population))]
        provenance = [{"operator": "initial", "parents": [], "parent_ids": []} for _ in population]
        candidates = self._evaluate_population(population, ids, panel, evaluator, deduplicate=deduplicate)
        self._record_generation(0, population, ids, candidates, provenance)
        thresholds = self._operator_thresholds()

        for generation in range(1, self.config.generations + 1):
            next_population = [candidate.node for candidate in candidates[: self.config.elitism]]
            next_ids = [self._individual_id(generation, slot) for slot in range(len(next_population))]
            provenance = [{"operator": "elite", "parents": [candidate.node.describe()], "parent_ids": [candidate.individual_id]} for candidate in candidates[: self.config.elitism]]
            operator_counts = {
                "elite_count": len(next_population),
                "crossover_attempted": 0, "crossover_effective": 0,
                "subtree_mutation_attempted": 0, "subtree_mutation_effective": 0,
                "point_mutation_attempted": 0, "point_mutation_effective": 0,
                "reproduction_count": 0,
                "offspring_same_as_parent_count": 0,
            }
            while len(next_population) < self.config.population_size:
                operator_choice = self.rng.random()
                if operator_choice < thresholds["crossover"]:
                    parent_left = self.tournament_selection(candidates)
                    parent_right = self.tournament_selection(candidates)
                    child = self.subtree_crossover(parent_left.node, parent_right.node)
                    operator, parents, parent_ids = "crossover", [parent_left.node.describe(), parent_right.node.describe()], [parent_left.individual_id, parent_right.individual_id]
                elif operator_choice < thresholds["subtree_mutation"]:
                    parent = self.tournament_selection(candidates)
                    child = self.subtree_mutation(parent.node)
                    operator, parents, parent_ids = "subtree_mutation", [parent.node.describe()], [parent.individual_id]
                elif operator_choice < thresholds["point_mutation"]:
                    parent = self.tournament_selection(candidates)
                    child = self.point_mutation(parent.node)
                    operator, parents, parent_ids = "point_mutation", [parent.node.describe()], [parent.individual_id]
                else:
                    parent = self.tournament_selection(candidates)
                    child = parent.node
                    operator, parents, parent_ids = "reproduction", [parent.node.describe()], [parent.individual_id]
                next_population.append(child)
                next_ids.append(self._individual_id(generation, len(next_ids)))
                provenance.append({"operator": operator, "parents": parents, "parent_ids": parent_ids})
                child_expression = child.describe()
                same_as_parent = child_expression == parents[0]
                if operator == "crossover":
                    operator_counts["crossover_attempted"] += 1
                    operator_counts["crossover_effective"] += int(child_expression not in parents)
                elif operator in {"subtree_mutation", "point_mutation"}:
                    operator_counts[f"{operator}_attempted"] += 1
                    operator_counts[f"{operator}_effective"] += int(not same_as_parent)
                else:
                    operator_counts["reproduction_count"] += 1
                operator_counts["offspring_same_as_parent_count"] += int(same_as_parent)
            population = next_population[: self.config.population_size]
            ids = next_ids[: self.config.population_size]
            candidates = self._evaluate_population(
                population, ids,
                panel,
                evaluator,
                deduplicate=deduplicate,
            )
            self._record_generation(generation, population, ids, candidates, provenance, operator_counts)
        return candidates

    def archive_candidates(self) -> list[GPCandidate]:
        """Unique first-seen hypotheses across generations, ranked by fast fitness."""
        return sorted(self._archive.values(), key=lambda candidate: candidate.evaluation.fitness, reverse=True)

    def telemetry_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return pd.DataFrame(self.generation_statistics), pd.DataFrame(self.candidate_audit)

    def stage_events_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.stage_events)

    def _individual_id(self, generation: int, slot: int) -> str:
        return f"generation-{generation}-slot-{slot}"

    def _record_generation(self, generation: int, population: list[FactorNode], ids: list[str], candidates: list[GPCandidate], provenance: list[dict[str, Any]], operator_counts: dict[str, int] | None = None) -> None:
        by_expression = {candidate.node.describe(): candidate for candidate in candidates}
        expressions = [node.describe() for node in population]
        fitness = [candidate.evaluation.fitness for candidate in candidates]
        complexity = [node.complexity() for node in population]
        rejected = [candidate for candidate in candidates if candidate.evaluation.fitness <= VERY_BAD_FITNESS]
        accepted = [candidate for candidate in candidates if candidate.evaluation.fitness > VERY_BAD_FITNESS]
        accepted_fitness = [candidate.evaluation.fitness for candidate in accepted]
        best = max(accepted, key=lambda candidate: candidate.evaluation.fitness) if accepted else None
        duplicate_slots = {
            slot for slot, expression in enumerate(expressions)
            if expression in expressions[:slot]
        }
        offspring_slots = [slot for slot, metadata in enumerate(provenance) if metadata["operator"] != "initial"]
        novel_offspring_count = sum(slot not in duplicate_slots for slot in offspring_slots)
        row = {"generation": generation, "population_size": len(population), "raw_individual_occurrences": len(population), "expression_evaluation_calls": len(population), "fast_evaluation_calls": len(population), "fast_unique_expression_survivors": len(candidates), "per_generation_unique_expressions": len(set(expressions)), "duplicate_count": len(population) - len(set(expressions)), "accepted_candidate_count": len(accepted), "best_fitness": best.evaluation.fitness if best else float("nan"), "accepted_mean_fitness": float(np.mean(accepted_fitness)) if accepted_fitness else float("nan"), "accepted_median_fitness": float(np.median(accepted_fitness)) if accepted_fitness else float("nan"), "accepted_fitness_std": float(np.std(accepted_fitness)) if accepted_fitness else float("nan"), "rejection_rate": float(len(rejected) / len(candidates)) if candidates else 0.0, "minimum_complexity": min(complexity) if complexity else 0, "median_complexity": float(np.median(complexity)) if complexity else 0.0, "best_fitness_candidate_complexity": best.node.complexity() if best else 0, "best_fitness_candidate_depth": best.node.depth() if best else 0, "rejected_candidate_count": len(rejected), "rejection_reasons": ";".join(sorted({str(c.evaluation.metrics.get("reject_reason", "unknown")) for c in rejected})), "offspring_duplicate_in_generation_count": sum(slot in duplicate_slots for slot in offspring_slots), "novel_offspring_count": novel_offspring_count, "novel_offspring_rate": float(novel_offspring_count / len(offspring_slots)) if offspring_slots else 0.0}
        row.update({key: int((operator_counts or {}).get(key, 0)) for key in ("elite_count", "crossover_attempted", "crossover_effective", "subtree_mutation_attempted", "subtree_mutation_effective", "point_mutation_attempted", "point_mutation_effective", "reproduction_count", "offspring_same_as_parent_count")})
        self.generation_statistics.append(row)
        first_seen: dict[str, int] = {}
        for slot, (node, individual_id, metadata) in enumerate(zip(population, ids, provenance)):
            expression = node.describe(); candidate = by_expression.get(expression)
            duplicate_of = first_seen.get(expression)
            if duplicate_of is None:
                first_seen[expression] = slot
            expression_hash = hashlib.sha256(expression.encode("utf-8")).hexdigest()
            reject_reason = candidate.evaluation.metrics.get("reject_reason", "") if candidate else "duplicate"
            status = "rejected" if candidate is not None and candidate.evaluation.fitness <= VERY_BAD_FITNESS else "passed"
            parent_ids = list(metadata["parent_ids"])
            self.candidate_audit.append({"individual_id": individual_id, "generation": generation, "slot": slot, "expression": expression, "expression_hash": expression_hash, "creation_operator": metadata["operator"], "parent_expressions": " | ".join(metadata["parents"]), "parent_individual_ids": " | ".join(parent_ids), "parent_1_id": parent_ids[0] if parent_ids else "", "parent_2_id": parent_ids[1] if len(parent_ids) > 1 else "", "complexity": node.complexity(), "depth": node.depth(), "fitness": candidate.evaluation.fitness if candidate else float("nan"), "reject_reason": reject_reason, "is_exact_duplicate": duplicate_of is not None, "duplicate_of_id": ids[duplicate_of] if duplicate_of is not None else ""})
            base = {"individual_id": individual_id, "expression": expression, "expression_hash": expression_hash, "generation": generation, "slot": slot, "reject_reason": reject_reason}
            self.stage_events.append({**base, "stage": "raw_gp_population", "status": "passed"})
            self.stage_events.append({**base, "stage": "fast_filter", "status": status})
            self.stage_events.append({**base, "stage": "exact_expression_dedup", "status": "rejected" if duplicate_of is not None else "passed", "reject_reason": "exact_duplicate" if duplicate_of is not None else ""})

    def record_stage_event(
        self,
        candidate: GPCandidate,
        *,
        stage: str,
        status: str,
        reject_reason: str = "",
    ) -> None:
        """Append an observational event without changing GP control flow."""
        expression = candidate.node.describe()
        self.stage_events.append(
            {
                "individual_id": candidate.individual_id,
                "expression": expression,
                "expression_hash": hashlib.sha256(expression.encode("utf-8")).hexdigest(),
                "generation": None,
                "slot": None,
                "stage": stage,
                "status": status,
                "reject_reason": reject_reason,
            }
        )

    def random_tree(self, max_depth: int | None = None) -> FactorNode:
        depth_limit = max_depth if max_depth is not None else self.config.max_depth
        for _ in range(100):
            node = self._random_tree(depth_limit)
            if node.depth() <= self.config.max_depth:
                return self._maybe_wrap_terminal(node)
        return self._maybe_wrap_terminal(field("close"))

    def tournament_selection(self, candidates: list[GPCandidate]) -> GPCandidate:
        tournament_size = min(self.config.tournament_size, len(candidates))
        picks = self.rng.choice(len(candidates), size=tournament_size, replace=False)
        pool = [candidates[int(index)] for index in picks]
        return max(pool, key=lambda candidate: candidate.evaluation.fitness)

    def subtree_crossover(self, left: FactorNode, right: FactorNode) -> FactorNode:
        left_paths = _node_paths(left)
        right_paths = _node_paths(right)
        for _ in range(50):
            left_path = left_paths[int(self.rng.integers(len(left_paths)))]
            right_path = right_paths[int(self.rng.integers(len(right_paths)))]
            donor = _get_subtree(right, right_path)
            child = _replace_subtree(left, left_path, donor)
            if child.depth() <= self.config.max_depth and _is_valid_tree(child):
                return child
        return left

    def subtree_mutation(self, node: FactorNode) -> FactorNode:
        paths = _node_paths(node)
        for _ in range(50):
            path = paths[int(self.rng.integers(len(paths)))]
            new_subtree = self.random_tree(max_depth=max(2, self.config.max_depth - len(path)))
            child = _replace_subtree(node, path, new_subtree)
            if child.depth() <= self.config.max_depth and _is_valid_tree(child):
                return child
        return node

    def point_mutation(self, node: FactorNode) -> FactorNode:
        paths = _node_paths(node)
        path = paths[int(self.rng.integers(len(paths)))]
        target = _get_subtree(node, path)
        mutated = self._mutate_node_pointwise(target)
        child = _replace_subtree(node, path, mutated)
        return child if child.depth() <= self.config.max_depth and _is_valid_tree(child) else node

    def _evaluate_population(
        self,
        population: list[FactorNode],
        individual_ids: list[str],
        panel: pd.DataFrame,
        evaluator: HypothesisEvaluator,
        deduplicate: bool = True,
    ) -> list[GPCandidate]:
        if not deduplicate:
            evaluated = [GPCandidate(node=node, evaluation=evaluator.fast_filter(node, panel), individual_id=individual_id) for node, individual_id in zip(population, individual_ids)]
            ordered = sorted(evaluated, key=lambda candidate: candidate.evaluation.fitness, reverse=True)
            for candidate in ordered:
                self._archive.setdefault(candidate.node.describe(), candidate)
            return ordered

        deduped: dict[str, GPCandidate] = {}
        for node, individual_id in zip(population, individual_ids):
            evaluation = evaluator.fast_filter(node, panel)
            expression = node.describe()
            previous = deduped.get(expression)
            candidate = GPCandidate(node=node, evaluation=evaluation, individual_id=individual_id)
            if previous is None or evaluation.fitness > previous.evaluation.fitness:
                deduped[expression] = candidate
        ordered = sorted(deduped.values(), key=lambda candidate: candidate.evaluation.fitness, reverse=True)
        for candidate in ordered:
            self._archive.setdefault(candidate.node.describe(), candidate)
        return ordered

    def _operator_thresholds(self) -> dict[str, float]:
        rates = {
            "crossover": max(0.0, float(self.config.crossover_rate)),
            "subtree_mutation": max(0.0, float(self.config.subtree_mutation_rate)),
            "point_mutation": max(0.0, float(self.config.point_mutation_rate)),
            "reproduction": max(0.0, float(self.config.reproduction_rate)),
        }
        total = sum(rates.values())
        if total <= 0.0:
            return {
                "crossover": 0.25,
                "subtree_mutation": 0.5,
                "point_mutation": 0.75,
                "reproduction": 1.0,
            }
        cumulative = 0.0
        thresholds: dict[str, float] = {}
        for key in ("crossover", "subtree_mutation", "point_mutation", "reproduction"):
            cumulative += rates[key] / total
            thresholds[key] = cumulative
        thresholds["reproduction"] = 1.0
        return thresholds

    def _random_tree(self, remaining_depth: int) -> FactorNode:
        if remaining_depth <= 1 or (remaining_depth > 1 and self.rng.random() < 0.3):
            return self._random_terminal()

        builders: list[Callable[[], FactorNode]] = [
            lambda: pct_change(self._random_tree(remaining_depth - 1), self._random_periods()),
            lambda: rolling_mean(self._random_tree(remaining_depth - 1), self._random_window()),
            lambda: rolling_std(self._random_tree(remaining_depth - 1), self._random_window()),
            lambda: delay(self._random_tree(remaining_depth - 1), self._random_periods()),
            lambda: delta(self._random_tree(remaining_depth - 1), self._random_periods()),
            lambda: ts_rank(self._random_tree(remaining_depth - 1), self._random_window()),
            lambda: zscore(self._random_tree(remaining_depth - 1)),
            lambda: rank(self._random_tree(remaining_depth - 1)),
            lambda: add(self._random_tree(remaining_depth - 1), self._random_tree(remaining_depth - 1)),
            lambda: sub(self._random_tree(remaining_depth - 1), self._random_tree(remaining_depth - 1)),
            lambda: mul(self._random_tree(remaining_depth - 1), self._random_tree(remaining_depth - 1)),
            lambda: div(self._random_tree(remaining_depth - 1), self._safe_denominator(remaining_depth - 1)),
            lambda: correlation(
                self._random_tree(remaining_depth - 1),
                self._random_tree(remaining_depth - 1),
                self._random_window(),
            ),
        ]
        for _ in range(100):
            builder = builders[int(self.rng.integers(len(builders)))]
            try:
                return builder()
            except ValueError:
                continue
        return self._random_terminal()

    def _random_terminal(self) -> FactorNode:
        if self.rng.random() < 0.75:
            blocked = set(str(name) for name in self.config.disallowed_raw_field_names)
            allowed_fields = [name for name in self.config.field_names if str(name) not in blocked]
            if allowed_fields:
                name = allowed_fields[int(self.rng.integers(len(allowed_fields)))]
                return field(name)
        low, high = self.config.constant_range
        return const(float(self.rng.uniform(low, high)))

    def _safe_denominator(self, remaining_depth: int) -> FactorNode:
        for _ in range(100):
            candidate = self._random_tree(max(1, remaining_depth))
            if not candidate.is_raw_field():
                return candidate
        return rolling_std(field("close"), 5)

    def _maybe_wrap_terminal(self, node: FactorNode) -> FactorNode:
        if not self.config.wrap_final_with_rank_or_zscore:
            return node
        if node.op in {"rank", "zscore"}:
            return node
        return rank(node) if self.rng.random() < 0.5 else zscore(node)

    def _mutate_node_pointwise(self, node: FactorNode) -> FactorNode:
        if node.op == "field":
            return self._random_terminal()
        if node.op == "const":
            low, high = self.config.constant_range
            return const(float(self.rng.uniform(low, high)))

        if len(node.children) == 1:
            unary_ops = [
                lambda child: pct_change(child, self._random_periods()),
                lambda child: rolling_mean(child, self._random_window()),
                lambda child: rolling_std(child, self._random_window()),
                lambda child: delay(child, self._random_periods()),
                lambda child: delta(child, self._random_periods()),
                lambda child: ts_rank(child, self._random_window()),
                lambda child: rank(child),
                lambda child: zscore(child),
            ]
            child = node.children[0]
            for _ in range(50):
                try:
                    return unary_ops[int(self.rng.integers(len(unary_ops)))](child)
                except ValueError:
                    continue
            return node

        if len(node.children) == 2:
            binary_ops = [
                lambda left, right: add(left, right),
                lambda left, right: sub(left, right),
                lambda left, right: mul(left, right),
                lambda left, right: div(left, right),
                lambda left, right: correlation(left, right, self._random_window()),
            ]
            left, right = node.children
            for _ in range(50):
                try:
                    return binary_ops[int(self.rng.integers(len(binary_ops)))](left, right)
                except ValueError:
                    continue
        return node

    def _random_periods(self) -> int:
        return int(self.config.periods_choices[int(self.rng.integers(len(self.config.periods_choices)))])

    def _random_window(self) -> int:
        return int(self.config.window_choices[int(self.rng.integers(len(self.config.window_choices)))])


def _node_paths(node: FactorNode, current_path: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    paths = [current_path]
    for index, child in enumerate(node.children):
        paths.extend(_node_paths(child, current_path + (index,)))
    return paths


def _get_subtree(node: FactorNode, path: tuple[int, ...]) -> FactorNode:
    current = node
    for step in path:
        current = current.children[step]
    return current


def _replace_subtree(node: FactorNode, path: tuple[int, ...], replacement: FactorNode) -> FactorNode:
    if not path:
        return replacement
    index = path[0]
    children = list(node.children)
    children[index] = _replace_subtree(children[index], path[1:], replacement)
    return FactorNode(op=node.op, children=tuple(children), value=node.value, params=node.params)


def _is_valid_tree(node: FactorNode) -> bool:
    if node.op == "field" and str(node.value) in {"open", "high", "low", "close"}:
        return False
    if node.op == "add" and len(node.children) == 2:
        if node.children[0].describe() == node.children[1].describe():
            return False
    if node.op == "div" and len(node.children) == 2:
        if node.children[1].is_raw_field():
            return False
    if node.div_count() > 2:
        return False
    return all(_is_valid_tree(child) for child in node.children)
