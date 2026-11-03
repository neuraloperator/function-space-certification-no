import heapq
import time

import torch

def refine(bound_cells, leaves, witness, seconds, batch=16384):
    started = time.perf_counter()
    parents_per_batch = batch // 4
    if parents_per_batch < 1 or seconds < 0:
        raise ValueError("require batch >= 4 and a nonnegative compute budget")
    queue, accepted = [], []
    serial = 0

    def retain(address, upper):
        for level in address[:, 0].unique().tolist():
            mask = address[:, 0] == level
            accepted.append(
                dict(
                    level=level,
                    indices=address[mask, 1:].clone(),
                    upper=upper[mask].clone(),
                )
            )

    def push(address, upper):
        nonlocal serial
        closed = upper <= witness["absolute_lower"]
        if bool(closed.any()):
            retain(address[closed], upper[closed])
        address, upper = address[~closed], upper[~closed]
        order = upper.argsort(descending=True)
        for begin in range(0, len(order), parents_per_batch):
            ids = order[begin : begin + parents_per_batch]
            serial += 1
            heapq.heappush(
                queue,
                (
                    -float(upper[ids[0]]),
                    serial,
                    address[ids].clone(),
                    upper[ids].clone(),
                ),
            )

    active_addresses, active_upper = [], []
    for leaf in leaves:
        mask = leaf["upper"] > witness["absolute_lower"]
        if bool((~mask).any()):
            accepted.append(
                dict(
                    level=leaf["level"],
                    indices=leaf["indices"][~mask].clone(),
                    upper=leaf["upper"][~mask].clone(),
                )
            )
        if bool(mask.any()):
            ids = leaf["indices"][mask]
            active_addresses.append(
                torch.cat(
                    (torch.full((len(ids), 1), leaf["level"], dtype=torch.int64), ids),
                    1,
                )
            )
            active_upper.append(leaf["upper"][mask])
    if active_addresses:
        push(torch.cat(active_addresses), torch.cat(active_upper))
    del active_addresses, active_upper

    offsets = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    reason = "compute budget reached"
    while queue and time.perf_counter() - started < seconds:
        _, _, address, upper = heapq.heappop(queue)
        closed = upper <= witness["absolute_lower"]
        if bool(closed.any()):
            retain(address[closed], upper[closed])
        address, upper = address[~closed], upper[~closed]
        if not len(address):
            continue
        if int(address[:, 0].max()) >= 20:
            push(address, upper)
            reason = "integer coverage depth limit reached"
            break
        level = address[:, 0].repeat_interleave(4) + 1
        indices = (2 * address[:, None, 1:] + offsets).reshape(-1, 2)
        parent_upper = upper.repeat_interleave(4)
        raw_upper, candidate = bound_cells(level, indices, parent_upper)
        child_upper = torch.minimum(raw_upper.reshape(-1).cpu(), parent_upper)
        if candidate["absolute_lower"] > witness["absolute_lower"]:
            witness = candidate
        push(torch.cat((level[:, None], indices), 1), child_upper)
    for _, _, address, upper in queue:
        retain(address, upper)
    if not queue:
        reason = "all cells below the certified witness"
    return accepted, witness, reason

def audit_cover(leaves, nt, nx):
    depth = max(leaf["level"] for leaf in leaves)
    total = nt * nx * 4**depth
    if total >= 2**63:
        raise ArithmeticError("coverage audit exceeds its int64 address range")
    starts, ends = [], []
    for leaf in leaves:
        level, indices = leaf["level"], leaf["indices"]
        it, ix = indices.T
        if bool(
            (it < 0).any()
            | (ix < 0).any()
            | (it >= nt * 2**level).any()
            | (ix >= nx * 2**level).any()
        ):
            raise ArithmeticError("leaf index outside the domain")
        address = (it >> level) * nx + (ix >> level)
        path = torch.zeros_like(address)
        for bit in range(level - 1, -1, -1):
            path = 4 * path + 2 * ((it >> bit) & 1) + ((ix >> bit) & 1)
        size = 4 ** (depth - level)
        start = address * 4**depth + path * size
        starts.append(start)
        ends.append(start + size)
    starts, ends = torch.cat(starts), torch.cat(ends)
    order = starts.argsort()
    starts, ends = starts[order], ends[order]
    if (
        int(starts[0]) != 0
        or int(ends[-1]) != total
        or not torch.equal(starts[1:], ends[:-1])
    ):
        raise ArithmeticError("leaf partition has an overlap or gap")
    return {
        "complete": True,
        "disjoint": True,
        "leaf_count": len(starts),
        "maximum_depth": depth,
        "integer_domain_measure": total,
        "covered_integer_measure": int((ends - starts).sum()),
    }
