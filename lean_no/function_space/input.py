from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from typing import Any

import torch

Number = int | float | str | Decimal | Fraction
TensorFn = Callable[[torch.Tensor], torch.Tensor]
Exprs = tuple["Expr", ...]

@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple[Expr, ...] = ()
    data: Fraction | None = None

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        if self.op == "const":
            return x.new_tensor(float(self.data))
        if self.op == "pi":
            return x.new_tensor(torch.pi)
        if self.op == "coord":
            return x
        if self.op == "add":
            return self.args[0].eval(x) + self.args[1].eval(x)
        if self.op == "mul":
            return self.args[0].eval(x) * self.args[1].eval(x)
        if self.op == "div":
            return self.args[0].eval(x) / self.args[1].eval(x)
        if self.op == "neg":
            return -self.args[0].eval(x)
        if self.op == "pow":
            return self.args[0].eval(x) ** int(self.data)
        if self.op == "sin_pi":
            return torch.sin(torch.pi * self.args[0].eval(x))
        if self.op == "cos_pi":
            return torch.cos(torch.pi * self.args[0].eval(x))
        if self.op in unary_ops:
            return getattr(torch, self.op)(self.args[0].eval(x))
        if self.op == "abs":
            return self.args[0].eval(x).abs()
        if self.op == "min":
            return torch.minimum(self.args[0].eval(x), self.args[1].eval(x))
        if self.op == "max":
            return torch.maximum(self.args[0].eval(x), self.args[1].eval(x))
        raise ValueError(f"unknown expression operation: {self.op}")

    def eval_arb(self, x: Any, analytic: bool = False) -> Any:
        return self.eval_memo(x, analytic, {})

    def eval_memo(
        self,
        x: Any,
        analytic: bool,
        memo: dict[int, Any],
    ) -> Any:
        key = id(self)
        if key in memo:
            return memo[key]
        if self.op == "const":
            data = f"{self.data.numerator}/{self.data.denominator}"
            value = type(x)(data)
        elif self.op == "pi":
            value = type(x).pi()
        elif self.op == "coord":
            value = x
        elif self.op in ("add", "mul", "div"):
            left = self.args[0].eval_memo(x, analytic, memo)
            right = self.args[1].eval_memo(x, analytic, memo)
            if self.op == "add":
                value = left + right
            elif self.op == "mul":
                value = left * right
            else:
                value = left / right
        elif self.op == "neg":
            value = -self.args[0].eval_memo(x, analytic, memo)
        elif self.op == "pow":
            value = self.args[0].eval_memo(x, analytic, memo) ** int(self.data)
        elif self.op == "sin_pi":
            value = self.args[0].eval_memo(x, analytic, memo).sin_pi()
        elif self.op == "cos_pi":
            value = self.args[0].eval_memo(x, analytic, memo).cos_pi()
        elif self.op in unary_ops:
            arg = self.args[0].eval_memo(x, analytic, memo)
            value = getattr(arg, self.op)()
        elif self.op == "abs":
            value = (
                self.args[0].eval_memo(x, analytic, memo).real_abs(analytic=analytic)
            )
        elif self.op in ("min", "max"):
            left = self.args[0].eval_memo(x, analytic, memo)
            right = self.args[1].eval_memo(x, analytic, memo)
            if self.op == "min":
                value = left.real_min(right, analytic=analytic)
            else:
                value = left.real_max(right, analytic=analytic)
        else:
            raise ValueError(f"unknown expression operation: {self.op}")
        memo[key] = value
        return value

    def diff(self, order: int = 1) -> Expr:
        if type(order) is not int or order < 0:
            raise ValueError("derivative order must be a nonnegative integer")
        result = self
        for _ in range(order):
            result = result.diff_once().simplify()
        return result

    def diff_once(self) -> Expr:
        if self.op in ("const", "pi"):
            return to_expr(0)
        if self.op == "coord":
            return to_expr(1)
        if self.op == "add":
            return self.args[0].diff_once() + self.args[1].diff_once()
        if self.op == "mul":
            left, right = self.args
            return left.diff_once() * right + left * right.diff_once()
        if self.op == "div":
            left, right = self.args
            top = left.diff_once() * right - left * right.diff_once()
            return top / right**2
        if self.op == "neg":
            return -self.args[0].diff_once()
        if self.op == "pow":
            power = int(self.data)
            if power == 0:
                return to_expr(0)
            base = self.args[0]
            return power * base ** (power - 1) * base.diff_once()

        arg = self.args[0]
        part = arg.diff_once()
        if self.op == "sin_pi":
            return pi() * cos_pi(arg) * part
        if self.op == "cos_pi":
            return -pi() * sin_pi(arg) * part
        if self.op == "sin":
            return cos(arg) * part
        if self.op == "cos":
            return -sin(arg) * part
        if self.op == "tan":
            return (1 + tan(arg) ** 2) * part
        if self.op == "sinh":
            return cosh(arg) * part
        if self.op == "cosh":
            return sinh(arg) * part
        if self.op == "tanh":
            return (1 - tanh(arg) ** 2) * part
        if self.op == "exp":
            return exp(arg) * part
        if self.op == "log":
            return part / arg
        if self.op == "sqrt":
            return part / (2 * sqrt(arg))
        if self.op == "asin":
            return part / sqrt(1 - arg**2)
        if self.op == "acos":
            return -part / sqrt(1 - arg**2)
        if self.op == "atan":
            return part / (1 + arg**2)
        if self.op == "asinh":
            return part / sqrt(1 + arg**2)
        if self.op == "acosh":
            return part / (sqrt(arg - 1) * sqrt(arg + 1))
        if self.op == "atanh":
            return part / (1 - arg**2)
        raise ValueError(f"expression operation is not differentiable: {self.op}")

    def simplify(self) -> Expr:
        if not self.args:
            return self
        args = tuple(arg.simplify() for arg in self.args)
        if self.op == "neg":
            arg = args[0]
            if is_const(arg, 0):
                return arg
            if arg.op == "const":
                return Expr("const", data=-arg.data)
            if arg.op == "neg":
                return arg.args[0]
            return Expr("neg", args)
        if self.op in ("add", "mul", "div"):
            left, right = args
            if self.op == "add":
                if is_const(left, 0):
                    return right
                if is_const(right, 0):
                    return left
                if left.op == right.op == "const":
                    return Expr("const", data=left.data + right.data)
            elif self.op == "mul":
                if is_const(left, 0) or is_const(right, 0):
                    return to_expr(0)
                if is_const(left, 1):
                    return right
                if is_const(right, 1):
                    return left
                if left.op == right.op == "const":
                    return Expr("const", data=left.data * right.data)
            else:
                if is_const(left, 0):
                    return left
                if is_const(right, 1):
                    return left
                if left.op == right.op == "const":
                    return Expr("const", data=left.data / right.data)
            return Expr(self.op, args)
        if self.op == "pow":
            if self.data == 0:
                return to_expr(1)
            if self.data == 1:
                return args[0]
            return Expr("pow", args, self.data)
        return Expr(self.op, args, self.data)

    def __add__(self, other: Expr | Number) -> Expr:
        return Expr("add", (self, to_expr(other)))

    def __radd__(self, other: Expr | Number) -> Expr:
        return to_expr(other) + self

    def __sub__(self, other: Expr | Number) -> Expr:
        return self + -to_expr(other)

    def __rsub__(self, other: Expr | Number) -> Expr:
        return to_expr(other) - self

    def __mul__(self, other: Expr | Number) -> Expr:
        return Expr("mul", (self, to_expr(other)))

    def __rmul__(self, other: Expr | Number) -> Expr:
        return to_expr(other) * self

    def __truediv__(self, other: Expr | Number) -> Expr:
        return Expr("div", (self, to_expr(other)))

    def __rtruediv__(self, other: Expr | Number) -> Expr:
        return to_expr(other) / self

    def __neg__(self) -> Expr:
        return Expr("neg", (self,))

    def __pow__(self, power: int) -> Expr:
        if power < 0:
            raise ValueError("expression powers must be nonnegative")
        return Expr("pow", (self,), Fraction(power))

    def __abs__(self) -> Expr:
        return Expr("abs", (self,))

def to_expr(value: Expr | Number) -> Expr:
    if isinstance(value, Expr):
        return value
    data = Fraction.from_float(value) if isinstance(value, float) else Fraction(value)
    return Expr("const", data=data)

def is_const(expr: Expr, value: int) -> bool:
    return expr.op == "const" and expr.data == value

def coord() -> Expr:
    return Expr("coord")

def pi() -> Expr:
    return Expr("pi")

def sin_pi(value: Expr | Number) -> Expr:
    return Expr("sin_pi", (to_expr(value),))

def cos_pi(value: Expr | Number) -> Expr:
    return Expr("cos_pi", (to_expr(value),))

unary_ops = (
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "exp",
    "log",
    "sqrt",
    "asin",
    "acos",
    "atan",
    "asinh",
    "acosh",
    "atanh",
)

def unary_expr(op: str, value: Expr | Number) -> Expr:
    return Expr(op, (to_expr(value),))

def sin(value: Expr | Number) -> Expr:
    return unary_expr("sin", value)

def cos(value: Expr | Number) -> Expr:
    return unary_expr("cos", value)

def tan(value: Expr | Number) -> Expr:
    return unary_expr("tan", value)

def sinh(value: Expr | Number) -> Expr:
    return unary_expr("sinh", value)

def cosh(value: Expr | Number) -> Expr:
    return unary_expr("cosh", value)

def tanh(value: Expr | Number) -> Expr:
    return unary_expr("tanh", value)

def exp(value: Expr | Number) -> Expr:
    return unary_expr("exp", value)

def log(value: Expr | Number) -> Expr:
    return unary_expr("log", value)

def sqrt(value: Expr | Number) -> Expr:
    return unary_expr("sqrt", value)

def asin(value: Expr | Number) -> Expr:
    return unary_expr("asin", value)

def acos(value: Expr | Number) -> Expr:
    return unary_expr("acos", value)

def atan(value: Expr | Number) -> Expr:
    return unary_expr("atan", value)

def asinh(value: Expr | Number) -> Expr:
    return unary_expr("asinh", value)

def acosh(value: Expr | Number) -> Expr:
    return unary_expr("acosh", value)

def atanh(value: Expr | Number) -> Expr:
    return unary_expr("atanh", value)

def min_expr(left: Expr | Number, right: Expr | Number) -> Expr:
    return Expr("min", (to_expr(left), to_expr(right)))

def max_expr(left: Expr | Number, right: Expr | Number) -> Expr:
    return Expr("max", (to_expr(left), to_expr(right)))

def expr_fn(items: Exprs) -> TensorFn:
    def evaluate(query: torch.Tensor) -> torch.Tensor:
        return torch.stack(tuple(item.eval(query) for item in items))

    return evaluate

def meta_tensor(values: torch.Tensor | Sequence[float]) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values
    return torch.tensor(tuple(values), dtype=torch.float32)

@dataclass(frozen=True, init=False)
class InputBounds:

    lower: TensorFn
    upper: TensorFn
    supremum: torch.Tensor
    lipschitz: torch.Tensor
    curvature: torch.Tensor | None
    lower_expr: Exprs | None = field(default=None)
    upper_expr: Exprs | None = field(default=None)
    derivative_enclosure: Callable | None = field(default=None)

    def __init__(
        self,
        lower: TensorFn | Exprs,
        upper: TensorFn | Exprs,
        supremum: torch.Tensor | Sequence[float],
        lipschitz: torch.Tensor | Sequence[float],
        curvature: torch.Tensor | Sequence[float] | None = None,
        derivative_enclosure: Callable | None = None,
    ) -> None:
        lower_expr = lower if isinstance(lower, tuple) else None
        upper_expr = upper if isinstance(upper, tuple) else None
        object.__setattr__(self, "lower_expr", lower_expr)
        object.__setattr__(self, "upper_expr", upper_expr)
        object.__setattr__(self, "lower", expr_fn(lower) if lower_expr else lower)
        object.__setattr__(self, "upper", expr_fn(upper) if upper_expr else upper)
        object.__setattr__(self, "supremum", meta_tensor(supremum))
        object.__setattr__(self, "lipschitz", meta_tensor(lipschitz))
        object.__setattr__(
            self,
            "curvature",
            None if curvature is None else meta_tensor(curvature),
        )
        object.__setattr__(self, "derivative_enclosure", derivative_enclosure)
        self.validate()

    @property
    def channels(self) -> int:
        return self.supremum.numel()

    @property
    def symbolic(self) -> bool:
        return self.lower_expr is not None and self.upper_expr is not None

    @property
    def is_exact(self) -> bool:
        if self.symbolic:
            return self.lower_expr == self.upper_expr
        return self.lower is self.upper

    def derivative_bounds(
        self, points: torch.Tensor, length: float, order: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if self.derivative_enclosure is not None:
            return self.derivative_enclosure(points, length, order)
        if isinstance(order, tuple) or not self.is_exact or not self.symbolic:
            return None
        try:
            expressions = tuple(expr.diff(order) for expr in self.lower_expr)
        except ValueError:
            return None
        from flint import arb, ctx

        coordinates = points.detach().cpu().tolist()
        ends = coordinates[1:] + [length]
        rows = [[], [], [], []]
        with ctx.workprec(80):
            for expr in expressions:
                values = [[], [], [], []]
                for left, right in zip(coordinates, ends):
                    node = arb(left)
                    for offset, domain in ((0, node), (2, node.union(arb(right)))):
                        result = expr.eval_arb(domain)
                        if not result.is_finite():
                            raise ValueError(
                                "input derivative is not finite on the domain"
                            )
                        values[offset].append(float(result.lower()))
                        values[offset + 1].append(float(result.upper()))
                for row, value in zip(rows, values):
                    row.append(value)
        return tuple(points.new_tensor(row) for row in rows)

    @classmethod
    def exact(
        cls,
        value: TensorFn | Exprs,
        supremum: torch.Tensor | Sequence[float],
        lipschitz: torch.Tensor | Sequence[float],
        curvature: torch.Tensor | Sequence[float] | None = None,
        derivative_enclosure: Callable | None = None,
    ) -> InputBounds:
        return cls(
            value,
            value,
            supremum,
            lipschitz,
            curvature,
            derivative_enclosure=derivative_enclosure,
        )

    def validate(self) -> None:
        valid_lipschitz = self.lipschitz.shape == self.supremum.shape or (
            self.lipschitz.ndim == 2
            and self.lipschitz.shape[0] == self.supremum.numel()
            and self.lipschitz.shape[1] > 0
        )
        if self.supremum.ndim != 1 or not valid_lipschitz:
            raise ValueError("input channel metadata has the wrong shape")
        if (
            not torch.isfinite(self.supremum).all()
            or not torch.isfinite(self.lipschitz).all()
        ):
            raise ValueError("input metadata must be finite")
        if torch.any(self.supremum < 0) or torch.any(self.lipschitz < 0):
            raise ValueError("input metadata must be nonnegative")
        if self.curvature is not None:
            hessian = self.lipschitz.shape == (
                self.channels,
                2,
            ) and self.curvature.shape == (self.channels, 2, 2)
            if self.curvature.shape != self.supremum.shape and not hessian:
                raise ValueError(
                    "input curvature bounds must have shape [channels] or [channels, 2, 2]"
                )
            if not torch.isfinite(self.curvature).all():
                raise ValueError("input curvature bounds must be finite")
            if torch.any(self.curvature < 0):
                raise ValueError("input curvature bounds must be nonnegative")
        for items in (self.lower_expr, self.upper_expr):
            if items is not None and len(items) != self.channels:
                raise ValueError("input bounds have the wrong shape")

@dataclass(frozen=True)
class FourierInput:

    nominal: InputBounds
    epsilon: torch.Tensor | float
    lipschitz: torch.Tensor | Sequence[float]

    def __post_init__(self):
        if not isinstance(self.nominal, InputBounds) or not self.nominal.is_exact:
            raise ValueError("FourierInput requires an exact nominal InputBounds")
        self.nominal.validate()
        if torch.as_tensor(self.epsilon).is_complex():
            raise ValueError("epsilon must be real")
        epsilon = torch.as_tensor(self.epsilon, dtype=torch.float64).detach().cpu()
        epsilon = torch.broadcast_to(epsilon, (self.nominal.channels,)).clone()
        if not torch.isfinite(epsilon).all() or (epsilon < 0).any():
            raise ValueError("epsilon must be finite and nonnegative")
        lipschitz = meta_tensor(self.lipschitz).detach().clone()
        if lipschitz.shape != self.nominal.lipschitz.shape:
            raise ValueError("full-input Lipschitz shape must match nominal metadata")
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "lipschitz", lipschitz)
        if (
            lipschitz.is_complex()
            or not torch.isfinite(lipschitz).all()
            or (lipschitz < 0).any()
        ):
            raise ValueError(
                "full-input Lipschitz bounds must be finite, real and nonnegative"
            )

    def envelope(self, points):
        values = self.nominal.lower(points)
        spread = self.epsilon.to(values).reshape(
            (self.nominal.channels,) + (1,) * (values.ndim - 1)
        )
        return values - spread, values + spread

    def bounds(self):
        nominal = self.nominal
        supremum = nominal.supremum + self.epsilon.to(device=nominal.supremum.device)
        if not self.epsilon.any():
            return InputBounds.exact(
                nominal.lower_expr or nominal.lower, supremum, self.lipschitz
            )
        if nominal.symbolic:
            values = self.epsilon.tolist()
            lower = tuple(
                expr - value for expr, value in zip(nominal.lower_expr, values)
            )
            upper = tuple(
                expr + value for expr, value in zip(nominal.upper_expr, values)
            )
        else:
            lower = lambda x: self.envelope(x)[0]
            upper = lambda x: self.envelope(x)[1]
        return InputBounds(lower, upper, supremum, self.lipschitz)
