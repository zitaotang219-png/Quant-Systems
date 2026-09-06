from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
import re
from typing import Any, Callable

import numpy as np
import pandas as pd

PanelTransform = Callable[[pd.Series], pd.Series]

PANEL_REQUIRED_COLUMNS = ("date", "symbol", "open", "high", "low", "close", "volume")
_FIELD_OP = "field"
_CONST_OP = "const"
_UNARY_OPS = {"pct_change", "rolling_mean", "rolling_std", "zscore", "rank", "delay", "delta", "ts_rank"}
_BINARY_OPS = {"add", "sub", "mul", "div", "correlation"}
_NUMERIC_TOKEN_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")


@dataclass(frozen=True)
class FactorNode:
    op: str
    children: tuple["FactorNode", ...] = ()
    value: Any | None = None
    params: tuple[tuple[str, Any], ...] = dataclass_field(default_factory=tuple)

    def evaluate(self, panel: pd.DataFrame) -> pd.Series:
        _validate_panel(panel)

        if self.op == _FIELD_OP:
            column_name = str(self.value)
            if column_name not in panel.columns:
                raise KeyError(f"Panel is missing field column: {column_name}")
            return panel[column_name].astype(float)

        if self.op == _CONST_OP:
            return pd.Series(float(self.value), index=panel.index, dtype=float)

        if self.op == "pct_change":
            node = self.children[0].evaluate(panel)
            periods = int(self.param("periods", 1))
            return _by_symbol(panel, node, lambda s: s.pct_change(periods=periods, fill_method=None))

        if self.op == "rolling_mean":
            node = self.children[0].evaluate(panel)
            window = int(self.param("window", 5))
            return _by_symbol(panel, node, lambda s: s.rolling(window=window, min_periods=window).mean())

        if self.op == "rolling_std":
            node = self.children[0].evaluate(panel)
            window = int(self.param("window", 5))
            return _by_symbol(panel, node, lambda s: s.rolling(window=window, min_periods=window).std())

        if self.op == "delay":
            node = self.children[0].evaluate(panel)
            periods = int(self.param("periods", 1))
            return _by_symbol(panel, node, lambda s: s.shift(periods))

        if self.op == "delta":
            node = self.children[0].evaluate(panel)
            periods = int(self.param("periods", 1))
            delayed = _by_symbol(panel, node, lambda s: s.shift(periods))
            return node - delayed

        if self.op == "ts_rank":
            node = self.children[0].evaluate(panel)
            window = int(self.param("window", 5))
            return _by_symbol(panel, node, lambda s: _rolling_percentile_rank(s, window))

        if self.op == "zscore":
            node = self.children[0].evaluate(panel)
            return _by_date_zscore(panel, node)

        if self.op == "rank":
            node = self.children[0].evaluate(panel)
            return _by_date_rank(panel, node)

        if self.op == "add":
            left, right = self._evaluate_binary(panel)
            return left + right

        if self.op == "sub":
            left, right = self._evaluate_binary(panel)
            return left - right

        if self.op == "mul":
            left, right = self._evaluate_binary(panel)
            return left * right

        if self.op == "div":
            left, right = self._evaluate_binary(panel)
            safe_denominator = right.where(right.abs() > 1e-12, np.nan)
            return left / safe_denominator

        if self.op == "correlation":
            left, right = self._evaluate_binary(panel)
            window = int(self.param("window", 5))
            return _rolling_correlation(panel, left, right, window)

        raise ValueError(f"Unsupported operator: {self.op}")

    def describe(self) -> str:
        if self.op == _FIELD_OP:
            return str(self.value)
        if self.op == _CONST_OP:
            return repr(self.value)
        if self.op in {"pct_change", "rolling_mean", "rolling_std", "delay", "delta", "ts_rank"}:
            child = self.children[0].describe()
            key = "periods" if self.op in {"pct_change", "delay", "delta"} else "window"
            return f"{self.op}({child}, {self.param(key)})"
        if self.op in {"zscore", "rank"}:
            return f"{self.op}({self.children[0].describe()})"
        if self.op == "correlation":
            left = self.children[0].describe()
            right = self.children[1].describe()
            return f"correlation({left}, {right}, {self.param('window')})"
        if self.op in _BINARY_OPS:
            left = self.children[0].describe()
            right = self.children[1].describe()
            return f"{self.op}({left}, {right})"
        return self.op

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)

    def complexity(self) -> int:
        return 1 + sum(child.complexity() for child in self.children)

    def param(self, name: str, default: Any = None) -> Any:
        for key, value in self.params:
            if key == name:
                return value
        return default

    def _evaluate_binary(self, panel: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        if len(self.children) != 2:
            raise ValueError(f"Binary operator {self.op} requires two children.")
        return self.children[0].evaluate(panel), self.children[1].evaluate(panel)

    def div_count(self) -> int:
        current = 1 if self.op == "div" else 0
        return current + sum(child.div_count() for child in self.children)

    def is_raw_field(self) -> bool:
        return self.op == _FIELD_OP


def field(name: str) -> FactorNode:
    return FactorNode(op=_FIELD_OP, value=name)


def const(value: float) -> FactorNode:
    return FactorNode(op=_CONST_OP, value=float(value))


def pct_change(node: FactorNode, periods: int) -> FactorNode:
    return FactorNode(op="pct_change", children=(node,), params=(("periods", int(periods)),))


def rolling_mean(node: FactorNode, window: int) -> FactorNode:
    return FactorNode(op="rolling_mean", children=(node,), params=(("window", int(window)),))


def rolling_std(node: FactorNode, window: int) -> FactorNode:
    return FactorNode(op="rolling_std", children=(node,), params=(("window", int(window)),))


def delay(node: FactorNode, periods: int) -> FactorNode:
    return FactorNode(op="delay", children=(node,), params=(("periods", int(periods)),))


def delta(node: FactorNode, periods: int) -> FactorNode:
    return FactorNode(op="delta", children=(node,), params=(("periods", int(periods)),))


def ts_rank(node: FactorNode, window: int) -> FactorNode:
    return FactorNode(op="ts_rank", children=(node,), params=(("window", int(window)),))


def zscore(node: FactorNode) -> FactorNode:
    return FactorNode(op="zscore", children=(node,))


def rank(node: FactorNode) -> FactorNode:
    return FactorNode(op="rank", children=(node,))


def add(left: FactorNode, right: FactorNode) -> FactorNode:
    if left.describe() == right.describe():
        raise ValueError("add(x, x) is not allowed.")
    return FactorNode(op="add", children=(left, right))


def sub(left: FactorNode, right: FactorNode) -> FactorNode:
    return FactorNode(op="sub", children=(left, right))


def mul(left: FactorNode, right: FactorNode) -> FactorNode:
    return FactorNode(op="mul", children=(left, right))


def div(left: FactorNode, right: FactorNode) -> FactorNode:
    if right.is_raw_field():
        raise ValueError("div(x, raw field) is not allowed.")
    if 1 + left.div_count() + right.div_count() > 2:
        raise ValueError("At most 2 div operators are allowed in one expression.")
    return FactorNode(op="div", children=(left, right))


def correlation(left: FactorNode, right: FactorNode, window: int) -> FactorNode:
    return FactorNode(
        op="correlation",
        children=(left, right),
        params=(("window", int(window)),),
    )


def parse_expression(expression: str) -> FactorNode:
    parser = _ExpressionParser(expression)
    return parser.parse()


def _validate_panel(panel: pd.DataFrame) -> None:
    missing = [column for column in PANEL_REQUIRED_COLUMNS if column not in panel.columns]
    if missing:
        raise KeyError(f"Panel is missing required columns: {missing}")


def _by_symbol(panel: pd.DataFrame, values: pd.Series, transform: PanelTransform) -> pd.Series:
    aligned = values.astype(float).copy()
    ordered = (
        pd.DataFrame(
            {
                "date": pd.to_datetime(panel["date"]),
                "symbol": panel["symbol"],
                "_value": aligned,
            },
            index=panel.index,
        )
        .sort_values(["symbol", "date"], kind="mergesort")
    )
    grouped = ordered.groupby("symbol", sort=False)["_value"]
    transformed = grouped.transform(transform)
    return transformed.reindex(panel.index).astype(float)


def _by_date_rank(panel: pd.DataFrame, values: pd.Series) -> pd.Series:
    aligned = values.astype(float).copy()
    frame = pd.DataFrame({"date": pd.to_datetime(panel["date"]), "_value": aligned}, index=panel.index)
    ranked = frame.groupby("date", sort=False)["_value"].rank(method="average", pct=True)
    return ranked.astype(float)


def _by_date_zscore(panel: pd.DataFrame, values: pd.Series) -> pd.Series:
    aligned = values.astype(float).copy()
    frame = pd.DataFrame({"date": pd.to_datetime(panel["date"]), "_value": aligned}, index=panel.index)
    grouped = frame.groupby("date", sort=False)["_value"]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    return ((frame["_value"] - mean) / std).astype(float)


def _rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).rank(method="average", pct=True)


def _rolling_correlation(panel: pd.DataFrame, left: pd.Series, right: pd.Series, window: int) -> pd.Series:
    ordered = (
        pd.DataFrame(
            {
                "date": pd.to_datetime(panel["date"]),
                "symbol": panel["symbol"],
                "_left": left.astype(float),
                "_right": right.astype(float),
            },
            index=panel.index,
        )
        .sort_values(["symbol", "date"], kind="mergesort")
    )
    pairwise = (
        ordered.groupby("symbol", sort=False)[["_left", "_right"]]
        .rolling(window=window, min_periods=window)
        .corr()
    )
    result = pairwise.xs("_left", level=-1)["_right"].droplevel(0)
    result = result.reindex(panel.index).astype(float)
    result.name = None
    return result


class _ExpressionParser:
    def __init__(self, expression: str) -> None:
        self.tokens = self._tokenize(expression)
        self.position = 0

    def parse(self) -> FactorNode:
        node = self._parse_expr()
        if self.position != len(self.tokens):
            raise ValueError(f"Unexpected token sequence in expression: {self.tokens[self.position:]}")
        return node

    def _parse_expr(self) -> FactorNode:
        token = self._consume_identifier()
        if self._match("("):
            self._consume("(")
            if token in {"pct_change", "rolling_mean", "rolling_std", "delay", "delta", "ts_rank"}:
                child = self._parse_expr()
                self._consume(",")
                value = self._consume_number()
                self._consume(")")
                if token == "pct_change":
                    return pct_change(child, value)
                if token == "rolling_mean":
                    return rolling_mean(child, value)
                if token == "rolling_std":
                    return rolling_std(child, value)
                if token == "delay":
                    return delay(child, value)
                if token == "delta":
                    return delta(child, value)
                return ts_rank(child, value)
            if token in {"zscore", "rank"}:
                child = self._parse_expr()
                self._consume(")")
                return zscore(child) if token == "zscore" else rank(child)
            if token in {"add", "sub", "mul", "div"}:
                left = self._parse_expr()
                self._consume(",")
                right = self._parse_expr()
                self._consume(")")
                if token == "add":
                    return add(left, right)
                if token == "sub":
                    return sub(left, right)
                if token == "mul":
                    return mul(left, right)
                return div(left, right)
            if token == "correlation":
                left = self._parse_expr()
                self._consume(",")
                right = self._parse_expr()
                self._consume(",")
                window = self._consume_number()
                self._consume(")")
                return correlation(left, right, window)
            raise ValueError(f"Unsupported function in expression: {token}")
        if _NUMERIC_TOKEN_RE.match(token):
            return const(float(token))
        return field(token)

    def _tokenize(self, expression: str) -> list[str]:
        tokens: list[str] = []
        current: list[str] = []
        for char in expression.strip():
            if char in {"(", ")", ","}:
                if current:
                    tokens.append("".join(current).strip())
                    current = []
                tokens.append(char)
                continue
            current.append(char)
        if current:
            tokens.append("".join(current).strip())
        return [token for token in tokens if token]

    def _match(self, token: str) -> bool:
        return self.position < len(self.tokens) and self.tokens[self.position] == token

    def _consume(self, token: str) -> str:
        if not self._match(token):
            current = self.tokens[self.position] if self.position < len(self.tokens) else "<eof>"
            raise ValueError(f"Expected token {token!r}, got {current!r}")
        self.position += 1
        return token

    def _consume_identifier(self) -> str:
        if self.position >= len(self.tokens):
            raise ValueError("Unexpected end of expression.")
        token = self.tokens[self.position]
        self.position += 1
        return token

    def _consume_number(self) -> int:
        token = self._consume_identifier()
        if not _NUMERIC_TOKEN_RE.match(token):
            raise ValueError(f"Expected numeric token, got {token!r}")
        return int(float(token))
