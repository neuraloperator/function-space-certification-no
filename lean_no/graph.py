from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import Tensor

from lean_no.bounds.activations import gelu_first

def activation_value(name, value):
    if name == "identity":
        return value
    return {"relu": torch.relu, "tanh": torch.tanh, "gelu": torch.nn.functional.gelu}[
        name
    ](value)

operator_kinds = frozenset(
    (
        "affine",
        "activation",
        "normalize",
        "spectral",
        "add",
        "concat",
        "pad",
        "initial_blend",
        "input",
        "profile",
    )
)

@dataclass(frozen=True)
class Operator:

    name: str
    kind: str
    parents: tuple[int, ...]
    data: dict = field(default_factory=dict)

    @property
    def weight(self):
        return self.data["weight"]

    @property
    def bias(self):
        return self.data.get("bias")

    @property
    def n_modes(self):
        return self.weight.shape[2:]

    @property
    def in_channels(self):
        return self.weight.shape[1]

    @property
    def out_channels(self):
        return self.weight.shape[0]

@dataclass(frozen=True)
class Graph:
    ops: tuple[Operator, ...]
    metadata: dict = field(default_factory=dict)

    def validate(self):
        names = set()
        for index, op in enumerate(self.ops, 1):
            if op.kind not in operator_kinds:
                raise ValueError(f"unsupported operator: {op.kind}")
            if op.name in names:
                raise ValueError(f"duplicate operator name: {op.name}")
            names.add(op.name)
            if (not op.parents and op.kind not in ("spectral", "input")) or any(
                p < 0 or p >= index for p in op.parents
            ):
                raise ValueError(f"invalid parents for operator {op.name}")
            if any(isinstance(value, torch.nn.Module) for value in op.data.values()):
                raise TypeError("operator payloads must not contain model modules")
            if op.kind == "affine" and op.weight.ndim != 2:
                raise ValueError("affine weights must have shape [output,input]")
            if op.kind == "activation" and op.data["activation"] not in (
                "identity",
                "relu",
                "tanh",
                "gelu",
            ):
                raise ValueError("unsupported activation")

    @property
    def dimension(self):
        return self.metadata["dimension"]

    @property
    def in_channels(self):
        return self.metadata["input_channels"]

    @property
    def out_channels(self):
        return self.metadata["output_channels"]

    @property
    def reference(self):
        return self.ops[0].weight.real

    def append_coordinates(self, values):
        if not self.grid_embedding:
            return values
        axes = [torch.linspace(0, 1, n + 1)[:-1].to(values) for n in values.shape[2:]]
        coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij"))
        return torch.cat(
            (values, coordinates[None].expand(len(values), -1, *values.shape[2:])), 1
        )

    @property
    def device(self):
        return self.reference.device

    @property
    def dtype(self):
        return self.reference.dtype

    @property
    def grid_embedding(self):
        return self.metadata.get("coordinate_channels", False)

    @property
    def lifting(self):
        return LayerGroup(self, self.metadata["lifting"])

    @property
    def projection(self):
        return LayerGroup(self, self.metadata["projection"])

    @property
    def blocks(self):
        return tuple(BlockView(self, ids) for ids in self.metadata["blocks"])

    @property
    def activations(self):
        return tuple(
            op.data["activation"] for op in self.ops if op.kind == "activation"
        )

@dataclass(frozen=True)
class LayerGroup:

    graph: Graph
    nodes: tuple[int, ...]

    @property
    def layers(self):
        return tuple(
            self.graph.ops[i - 1]
            for i in self.nodes
            if self.graph.ops[i - 1].kind == "affine"
        )

    @property
    def activation(self):
        return next(
            (
                self.graph.ops[i - 1].data["activation"]
                for i in self.nodes
                if self.graph.ops[i - 1].kind == "activation"
            ),
            "identity",
        )

@dataclass(frozen=True)
class BlockView:
    graph: Graph
    nodes: dict

    def operation(self, name):
        index = self.nodes.get(name)
        return None if index is None else self.graph.ops[index - 1]

    @property
    def conv(self):
        return self.operation("spectral")

    @property
    def skip(self):
        return self.operation("skip")

    @property
    def norm0(self):
        return self.operation("norm0")

    @property
    def norm1(self):
        return self.operation("norm1")

    @property
    def channel_mlp(self):
        return self.nodes.get("channel_mlp")

    @property
    def activation(self):
        ids = self.nodes["activations"]
        return self.graph.ops[ids[0] - 1].data["activation"] if ids else None

    @property
    def activation_repetitions(self):
        return len(self.nodes["activations"])

def execute(graph: Graph, inputs: Any, apply: Callable, progress=None):
    graph.validate()
    cache = {0: inputs}
    uses = Counter(p for op in graph.ops for p in set(op.parents))
    for index, op in enumerate(graph.ops, 1):
        result = apply(op, tuple(cache[p] for p in op.parents))
        cache[index] = result
        if progress is not None:
            progress(op, result)
        for parent in set(op.parents):
            uses[parent] -= 1
            if not uses[parent]:
                del cache[parent]
    return cache[len(graph.ops)]

@dataclass
class GraphWalker:
    rule: Any
    cache: dict
    key: Callable | None

    def __call__(self, node, *context):
        token = self.key(node, *context) if self.key else node
        if token not in self.cache:
            operation = (
                self.rule[node.kind] if isinstance(self.rule, dict) else self.rule
            )
            self.cache[token] = operation(node, self, *context)
        return self.cache[token]

def walk(rule, cache=None, key=None):
    return GraphWalker(rule, {} if cache is None else cache, key)

@dataclass
class NodeState:
    error: Tensor
    sizes: Tensor
    data: dict = field(default_factory=dict)

@dataclass
class FieldRun:

    metadata: dict
    token: object = field(default_factory=object)
    ops: list = field(default_factory=list)
    states: list = field(default_factory=lambda: [None])
    imports: dict = field(default_factory=dict)

    def import_field(self, source):
        if source.run is self:
            return source.node
        key = (source.run.token, source.node)
        if key not in self.imports:
            op = source.operation
            parents = tuple(self.import_field(Field(source.run, i)) for i in op.parents)
            self.ops.append(
                Operator(f"field.{len(self.ops) + 1}", op.kind, parents, op.data)
            )
            self.states.append(source.state)
            result = Field(self, len(self.ops))
            self.imports[key] = result.node
        return self.imports[key]

class FieldData:

    def __init__(self, source):
        self.source = source

    def __getitem__(self, key):
        source = self.source
        links = source.operation.data.get("links", {})
        if key in links:
            return Field(source.run, source.operation.parents[links[key]])
        if key in source.state.data:
            return source.state.data[key]
        return source.operation.data[key]

    def __setitem__(self, key, value):
        if isinstance(value, Field):
            raise TypeError(
                "field dependencies must be supplied when constructing the node"
            )
        self.source.state.data[key] = value

    def __iter__(self):
        return iter(
            dict.fromkeys(
                (
                    *self.source.operation.data,
                    *self.source.operation.data.get("links", {}),
                    *self.source.state.data,
                )
            )
        )

    def __len__(self):
        return sum(1 for _ in self)

    def __contains__(self, key):
        return (
            key in self.source.operation.data
            or key in self.source.state.data
            or key in self.source.operation.data.get("links", {})
        )

    def get(self, key, default=None):
        return self[key] if key in self else default

    def keys(self):
        return tuple(key for key in self if key != "links")

    def update(self, values=None, **kwargs):
        for key, value in {**(values or {}), **kwargs}.items():
            self[key] = value

@dataclass(eq=False, frozen=True)
class Field:

    run: FieldRun
    node: int

    def __hash__(self):
        return hash((id(self.run), self.node))

    def __eq__(self, other):
        return (
            isinstance(other, Field)
            and self.run is other.run
            and self.node == other.node
        )

    @property
    def operation(self):
        return self.run.ops[self.node - 1]

    @property
    def graph(self):
        return Graph(tuple(self.run.ops[: self.node]), self.metadata)

    @property
    def state(self):
        return self.run.states[self.node]

    @property
    def kind(self):
        return self.operation.kind

    @property
    def data(self):
        return FieldData(self)

    @property
    def parents(self):
        count = len(self.operation.parents) - len(self.operation.data.get("links", {}))
        return tuple(Field(self.run, i) for i in self.operation.parents[:count])

    @property
    def error(self):
        return self.state.error

    @property
    def sizes(self):
        return self.state.sizes

    @property
    def size(self):
        return self.sizes[0]

    @property
    def metadata(self):
        return self.run.metadata

    @property
    def chunk_size(self):
        return self.metadata.get("chunk_size", 16)

    @property
    def is_exact(self):
        return not bool((self.error != 0).any())

    @property
    def channel_width(self):
        return 2 * self.error

    @property
    def width(self):
        return self.channel_width.amax()

    @property
    def domain(self):
        from lean_no.domain import PeriodicBox

        active = (self.periods[0] * self.metadata.get("split", 1.0), *self.periods[1:])
        return PeriodicBox(
            self.periods,
            self.metadata.get("axes", ()),
            self.metadata.get("periodic", ()),
            active,
        )

    @property
    def periods(self):
        return self.metadata.get("lengths", (1.0, 1.0))

    @property
    def orders(self):
        return self.metadata.get("orders", ((0,) * len(self.periods),))

    @property
    def arithmetic(self):
        return "affine"

    def enclose(
        self, centers, halfwidth, orders=None, values_only=False, profile_cache=None
    ):
        from lean_no.propagation.derivative_cells import field_jets

        orders = (
            tuple(orders)
            if orders is not None
            else self.orders[:1]
            if values_only
            else self.orders
        )
        return field_jets(
            self,
            centers,
            halfwidth,
            orders,
            self.periods,
            profile_cache=profile_cache,
            split=self.metadata.get("split", 1.0),
        )

    def with_error(self, error):
        return make_field(
            self.kind, dict(self.data), self.parents, error, self.sizes, self.metadata
        )

    def with_parents(self, parents):
        return make_field(
            self.kind, dict(self.data), parents, self.error, self.sizes, self.metadata
        )

    @torch.no_grad()
    def query(self, points):
        if self.metadata.get("basis") != "spherical":
            from lean_no.propagation.rollout import output_bounds

            return output_bounds(self, points)
        from lean_no.propagation.spherical import evaluate

        if ((points[:, 0] < 0) | (points[:, 0] > torch.pi)).any():
            raise ValueError("colatitude must lie in [0, pi]")
        values = [
            evaluate(self, part[:, 0], part[:, 1], False, {})
            for part in points.split(self.chunk_size)
        ]
        center = (
            torch.cat(values) if values else self.error.new_empty((0, len(self.error)))
        )
        from lean_no.certificate import QueryBounds

        return QueryBounds(center, self.error.expand_as(center))

def make_field(kind, data, parents, error, sizes, metadata=None):
    data = dict(data)
    data.pop("links", None)
    named = {key: value for key, value in data.items() if isinstance(value, Field)}
    dependencies = (*parents, *named.values())
    run = (
        dependencies[0].run
        if dependencies
        else FieldRun({} if metadata is None else dict(metadata))
    )
    if metadata is not None and run.metadata != metadata:
        run = FieldRun(dict(metadata))
    indices = tuple(run.import_field(source) for source in dependencies)
    for key in named:
        del data[key]
    if named:
        data["links"] = {key: len(parents) + i for i, key in enumerate(named)}
    run.ops.append(Operator(f"field.{len(run.ops) + 1}", kind, indices, data))
    run.states.append(NodeState(error, sizes[None] if sizes.ndim == 1 else sizes))
    result = Field(run, len(run.ops))
    return result

def affine(source, weight, bias):
    return make_field(
        "affine",
        dict(weight=weight, bias=bias),
        (source,),
        weight.abs() @ source.error,
        weight.abs() @ source.size + bias.abs(),
    )

def add(left, right):
    return make_field(
        "add", {}, (left, right), left.error + right.error, left.size + right.size
    )

def gelu(source):
    return make_field(
        "activation",
        {"activation": "gelu"},
        (source,),
        gelu_first * source.error,
        source.size,
    )

def fourier(coefficients, modes, error, metadata=None):
    modes = tuple(modes)
    if coefficients.shape[1:] != tuple(len(axis) for axis in modes):
        raise ValueError("coefficient shape must match the mode axes")
    symmetry = torch.where(modes[-1] == 0, 1.0, 2.0)
    return make_field(
        "spectral",
        dict(coefficients=coefficients, modes=modes),
        (),
        error,
        (coefficients.abs() * symmetry).flatten(1).sum(-1),
        metadata,
    )
