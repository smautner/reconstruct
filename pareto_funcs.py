#!/usr/bin/env python
"""Provides Pareto optimization functions."""

import numpy as np





def _remove_duplicates(costs, items):
    dedup_costs = []
    dedup_items = []
    costs = [tuple(c) for c in costs]
    prev_c = None
    for c, g in sorted(zip(costs, items),key=lambda x:x[0]):
        if prev_c != c:
            dedup_costs.append(c)
            dedup_items.append(g)
            prev_c = c
    return np.array(dedup_costs), dedup_items


def _is_pareto_efficient(costs):
    is_eff = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_eff[i]:
            is_eff[i] = False
            # Remove dominated points
            is_eff[is_eff] = np.any(costs[is_eff] < c, axis=1)
            is_eff[i] = True
    return is_eff


def _pareto_front(costs):
    return [i for i, p in enumerate(_is_pareto_efficient(costs)) if p]


def _pareto_set(items, costs, return_costs=False):
    ids = _pareto_front(costs)
    select_items = [items[i] for i in ids]
    if return_costs:
        select_costs = np.array([costs[i] for i in ids])
        return select_items, select_costs
    else:
        return select_items


def get_pareto_set(items, costs, return_costs=False):
    """get_pareto_set."""
    costs, items = _remove_duplicates(costs, items)
    return _pareto_set(items, costs, return_costs)


def _manage_int_or_float(input_val, ref_val):
    assert (ref_val > 0), 'Error: ref val not >0'
    out_val = None
    if isinstance(input_val, int):
        out_val = min(input_val, ref_val)
    elif isinstance(input_val, float):
        msg = 'val=%.3f should be >0 and <=1'
        assert(0 < input_val <= 1), msg
        out_val = int(input_val * float(ref_val))
    else:
        raise Exception('Error on val type')
    out_val = max(out_val, 2)
    return out_val