"""Separate total-update and per-target-presentation budgets."""
import math


def updates(arm):
    if arm=="fixed_total":return 2048
    if arm=="matched_presentations":return 16384
    raise ValueError("Unknown few-target budget")


def event(step,case_count=8,batch_size=4):
    if step<0 or case_count!=8 or batch_size!=4:raise ValueError("Fixed eight-target design required")
    case=step%case_count;exposure=step//case_count
    return case,[exposure*batch_size+slot for slot in range(batch_size)]


def learning_rate(arm,step,high=3e-4,low=3e-5):
    if step<0 or step>=updates(arm):raise ValueError("Invalid update")
    exposure=step if arm=="fixed_total" else step//8
    return low+.5*(high-low)*(1+math.cos(math.pi*exposure/2047))
