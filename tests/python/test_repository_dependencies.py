"""Resolve active repository entrypoints; historical evidence is not executable."""
from tools.check_repository import check_repository


def test_active_dependency_graph_resolves():
    assert check_repository() == []
