"""Validate measured gate evidence; hashes detect damage, not hostile writers."""
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

EXECUTION = {'planned', 'completed', 'blocked', 'aborted'}
VALIDATION = {'pass', 'fail', 'not_applicable', 'invalid'}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def artifact(root, name):
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError('artifact path must be relative')
    path = (root / name).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise ValueError(f'missing or escaping artifact: {name}')
    return path


def junit_counts(path):
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as error:
        raise ValueError('invalid JUnit XML') from error
    cases = list(root.iter('testcase'))
    return {'tests': len(cases),
            'skipped': sum(x.find('skipped') is not None for x in cases),
            'failures': sum(x.find('failure') is not None for x in cases),
            'errors': sum(x.find('error') is not None for x in cases)}


def validate_profile(profile):
    if not isinstance(profile, dict) or type(profile.get('schema_version')) is not int or profile.get('schema_version') != 1 or not isinstance(profile.get('profile_id'), str) or not profile['profile_id']:
        raise ValueError('invalid profile identity/version')
    if not isinstance(profile.get('cases'), list) or not profile['cases']:
        raise ValueError('profile must declare cases')
    ids = set()
    for case in profile['cases']:
        if not isinstance(case, dict):
            raise ValueError('case must be an object')
        if not isinstance(case.get('id'), str) or not case['id'] or case['id'] in ids:
            raise ValueError('duplicate or empty case identity')
        ids.add(case['id'])
        if type(case.get('required')) is not bool:
            raise ValueError('required must be boolean')
        if case.get('tier') not in ('CPU', 'FAKE', 'C_IMPL', 'HW1', 'HW4', 'HW-LONG'):
            raise ValueError('unknown execution tier')
        if not isinstance(case.get('nodeid'), str) or not case['nodeid']:
            raise ValueError('missing pytest nodeid')
        value = case.get('timeout_seconds')
        if type(value) not in (int, float) or not 0 < value <= 86400:
            raise ValueError('case timeout must be finite and bounded')


def aggregate(cases):
    required = [c for c in cases if c['required']]
    if not required or any(c['execution_status'] == 'blocked' for c in required):
        return 'blocked', 'invalid'
    if any(c['execution_status'] == 'aborted' for c in required):
        return 'aborted', 'invalid'
    if any(c['validation_status'] != 'pass' for c in required):
        return 'completed', 'fail'
    return 'completed', 'pass'


def validate_manifest(path):
    path = Path(path)
    root = path.parent.resolve()
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict) or type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1:
        raise ValueError('invalid manifest version')
    names = set()
    for item in manifest['artifacts']:
        name = item['path']
        if name in names:
            raise ValueError('duplicate artifact')
        names.add(name)
        current = artifact(root, name)
        if type(item['bytes']) is not int or current.stat().st_size != item['bytes']:
            raise ValueError(f'artifact size changed: {name}')
        if sha256_file(current) != item['sha256']:
            raise ValueError(f'artifact hash changed: {name}')
    if not {'result.json', 'profile.json', 'source_manifest.json'} <= names:
        raise ValueError('missing required artifacts')
    result = json.loads(artifact(root, 'result.json').read_text())
    profile = json.loads(artifact(root, 'profile.json').read_text())
    validate_profile(profile)
    if not isinstance(result, dict) or type(result.get('schema_version')) is not int or result['schema_version'] != 1 or result.get('profile_id') != profile['profile_id']:
        raise ValueError('result identity/version differs from profile')
    if result.get('profile_sha256') != sha256_file(root / 'profile.json'):
        raise ValueError('profile digest mismatch')
    if result.get('execution_status') not in EXECUTION or result.get('validation_status') not in VALIDATION:
        raise ValueError('unknown result status')
    declared = {c['id']: c for c in profile['cases']}
    actual = result.get('cases', [])
    if len(actual) != len(declared) or {c['id'] for c in actual} != set(declared):
        raise ValueError('case coverage differs from profile')
    count = 0
    for case in actual:
        spec = declared[case['id']]
        if case.get('tier') != spec['tier']:
            raise ValueError('execution tier differs from profile')
        if type(case.get('required')) is not bool or case['required'] != spec['required']:
            raise ValueError('required case changed')
        if case['execution_status'] not in EXECUTION or case['validation_status'] not in VALIDATION:
            raise ValueError('unknown case status')
        if case.get('junit'):
            if case['junit'] not in names:
                raise ValueError('unhashed junit report')
            measured = junit_counts(artifact(root, case['junit']))
            if measured != case['counts']:
                raise ValueError('junit counts disagree with result')
        else:
            measured = dict(tests=0, skipped=0, failures=0, errors=0)
        count += measured['tests']
        if case['validation_status'] == 'pass':
            if (case['execution_status'] != 'completed' or case['returncode'] != 0 or
                    not measured['tests'] or any(measured[k] for k in ('skipped','failures','errors'))):
                raise ValueError('pass without executed passing cases')
        for binary in case.get('binary_artifacts', []):
            if binary not in names:
                raise ValueError('unhashed test binary')
        if case['tier'] == 'C_IMPL' and case['validation_status'] == 'pass' and not case.get('binary_artifacts'):
            raise ValueError('C_IMPL pass requires actual hashed test binary')
        for key in ('stdout', 'stderr'):
            if case.get(key) not in names:
                raise ValueError('missing hashed raw output')
    if type(result.get('case_count')) is not int or result['case_count'] != count:
        raise ValueError('case_count mismatch')
    if (result['execution_status'], result['validation_status']) != aggregate(actual):
        raise ValueError('aggregate status disagrees with required cases')
    for metric in result.get('metrics', {}).values():
        if not isinstance(metric, dict) or not isinstance(metric.get('unit'), str):
            raise ValueError('metric requires a unit')
        n, warmup = metric.get('sample_count'), metric.get('warmup_count')
        if type(n) is not int or n <= 0 or type(warmup) is not int or warmup < 0:
            raise ValueError('metric requires positive measured samples')
        if not metric.get('statistics', {}).get('method') or metric.get('raw_values') not in names:
            raise ValueError('metric requires method and hashed samples')
        values = json.loads(artifact(root, metric['raw_values']).read_text())
        if not isinstance(values, list) or len(values) != n + warmup or any(type(v) not in (int,float) or not math.isfinite(v) for v in values):
            raise ValueError('metric samples disagree with declaration')
    return result
