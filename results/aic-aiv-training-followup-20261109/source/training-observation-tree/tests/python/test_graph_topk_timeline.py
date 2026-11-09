import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('graph_topk_analyse', Path(__file__).parents[2] / 'tools/graph_topk_analyse.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_overlapping_tasks_are_not_double_counted():
    assert module.duration([(0, 5), (3, 8), (8, 10)]) == 10
    assert module.intersection([(0, 5), (3, 8)], [(4, 6), (7, 10)]) == 3


def test_disjoint_tasks_and_empty_side():
    assert module.intersection([(0, 2)], [(2, 5)]) == 0
    assert module.intersection([], [(2, 5)]) == 0
