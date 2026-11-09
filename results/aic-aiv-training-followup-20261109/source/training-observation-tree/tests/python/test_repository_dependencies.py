"""Catch dangling local imports and CMake source references after curation."""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_explicit_local_imports_resolve():
    missing = []
    for folder in ('python', 'experiments', 'tests'):
        for source in (ROOT / folder).rglob('*.py'):
            tree = ast.parse(source.read_text(encoding='utf-8-sig'))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [item.name for item in node.names]
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    names = [node.module or '']
                else:
                    continue
                for name in names:
                    if not name.startswith(('python.', 'experiments.')):
                        continue
                    target = ROOT.joinpath(*name.split('.'))
                    if not target.is_dir() and not target.with_suffix('.py').is_file():
                        missing.append((str(source.relative_to(ROOT)), name))
    assert not missing, missing


def test_cmake_sources_exist():
    text = (ROOT / 'CMakeLists.txt').read_text(encoding='utf-8')
    paths = re.findall(r'(?m)^\s*((?:src|tests)/[^\s)]+\.c)\s*$', text)
    assert paths
    assert all((ROOT / path).is_file() for path in paths)
