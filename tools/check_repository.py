#!/usr/bin/env python3
"""Check local imports, executable targets and build/profile source references."""
import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ('python', 'experiments', 'tests', 'tools', 'scripts')
RETIRED = {'direct_checkpoint','c_bindings','chunk_helpers','disk_layout','full_checkpoint_protocol','training_cell','delta_protocol','delta_cell','s2_delta','r0_pipeline','multirank_protocol'}


def exists(module):
    parts = module.split('.')
    for base in (ROOT, ROOT / 'python'):
        path = base.joinpath(*parts)
        if path.with_suffix('.py').is_file() or path.is_dir():
            return True
    return False


def inspect_source(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    errors = []
    def check(module, line):
        if module.split('.')[0] in RETIRED or (module.startswith(('python.', 'npu_nvme', 'experiments.', 'tools.')) and not exists(module)):
            errors.append(f'{path.relative_to(ROOT)}:{line}: missing local module {module}')
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names: check(item.name, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = path.parent
                for _ in range(node.level - 1): base = base.parent
                target = base.joinpath(*(node.module or '').split('.'))
                if not target.is_dir() and not target.with_suffix('.py').is_file():
                    errors.append(f'{path.relative_to(ROOT)}:{node.lineno}: unresolved relative import')
            else:
                check(node.module or '', node.lineno)
        elif isinstance(node, ast.Call) and node.args:
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ''
            if name in ('__import__','import_module') and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                check(node.args[0].value, node.lineno)
    # Resolve static source paths used in command lists. Output/log literals
    # and historical explanatory strings are deliberately not dependencies.
    constants = {'ROOT':ROOT, 'REPO_ROOT':ROOT}
    def resolve(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str): return node.value
        if isinstance(node, ast.Name): return constants.get(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            a,b = resolve(node.left),resolve(node.right)
            if a is not None and b is not None: return Path(a)/b
        if isinstance(node, ast.Call) and node.args:
            name = node.func.id if isinstance(node.func, ast.Name) else ''
            if name in ('str','Path'): return resolve(node.args[0])
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = resolve(node.value)
            if value is not None:
                for target in node.targets:
                    if isinstance(target, ast.Name): constants[target.id] = value
    for node in ast.walk(tree):
        if isinstance(node, (ast.List,ast.Tuple)):
            for item in node.elts:
                value = resolve(item)
                if isinstance(value, Path) and value.suffix == '.py' and ROOT in value.parents and not value.is_file():
                    errors.append(f'{path.relative_to(ROOT)}:{item.lineno}: missing command target {value.relative_to(ROOT)}')
    return errors


def check_repository():
    errors = []
    sources = [p for folder in FOLDERS for p in (ROOT/folder).rglob('*.py')] + [ROOT/'train.py']
    for path in sources:
        errors.extend(inspect_source(path))
    for path in (ROOT/'config/gates').glob('*.json'):
        for case in json.loads(path.read_text())['cases']:
            target = ROOT / case['nodeid'].split('::')[0]
            if not target.exists(): errors.append(f'{path.relative_to(ROOT)}: missing {target}')
    cmake = (ROOT/'CMakeLists.txt').read_text()
    for name in re.findall(r'(?m)^\s*((?:src|tests)/[^\s)]+\.c)\s*$', cmake):
        if not (ROOT/name).is_file(): errors.append(f'CMake: missing {name}')
    return errors


if __name__ == '__main__':
    errors = check_repository()
    print(json.dumps({'status':'fail' if errors else 'pass','errors':errors}, indent=2))
    raise SystemExit(bool(errors))
