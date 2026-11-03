from functools import cache

@cache
def chain_terms(order):
    if not any(order):
        return {(0, ()): 1}
    axis = next(i for i, n in enumerate(order) if n)
    previous = tuple(n - (i == axis) for i, n in enumerate(order))
    unit = tuple(int(i == axis) for i in range(len(order)))
    terms = {}
    for (power, factors), count in chain_terms(previous).items():
        key = power + 1, tuple(sorted((*factors, unit)))
        terms[key] = terms.get(key, 0) + count
        for j, factor in enumerate(factors):
            raised = tuple(n + (i == axis) for i, n in enumerate(factor))
            key = power, tuple(sorted((*factors[:j], *factors[j + 1 :], raised)))
            terms[key] = terms.get(key, 0) + count
    return terms

def compose_jet(jet, orders, compose):
    if len(jet) != len(orders) or not orders or any(orders[0]):
        raise ValueError("matching jet orders starting with the value required")
    parts = dict(zip(orders, jet))
    outer = [compose(jet[0], j) for j in range(max(map(sum, orders)) + 1)]
    result = []
    for order in orders:
        grouped = {}
        for (power, factors), count in chain_terms(tuple(order)).items():
            product = 1
            for factor in sorted(set(factors)):
                value = parts[factor]
                repeats = factors.count(factor)
                while repeats:
                    if repeats % 2:
                        product = product * value
                    repeats //= 2
                    if repeats:
                        value = value.square()
            product = count * product
            grouped[power] = grouped[power] + product if power in grouped else product
        result.append(sum(outer[power] * grouped[power] for power in sorted(grouped)))
    return result
