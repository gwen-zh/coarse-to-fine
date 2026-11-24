"""Nested 32-target budgets, preserving eight-target per-case noise/LR control."""
from few_target_flow import learning_rate as eight_rate


def updates(arm):
    if arm=="fixed_total":return 2048
    if arm=="matched_presentations":return 8192
    raise ValueError("Unknown 32-target budget")


def event(step):
    if step<0:raise ValueError("Nonnegative step required")
    return step%32,[4*(step//32)+slot for slot in range(4)]


def learning_rate(arm,step):
    if step<0 or step>=updates(arm):raise ValueError("Invalid update")
    virtual=step if arm=="fixed_total" else (step//32)*8+step%8
    return eight_rate("fixed_total",virtual)
