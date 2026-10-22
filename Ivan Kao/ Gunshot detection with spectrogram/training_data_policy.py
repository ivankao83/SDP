"""Keep reserved evaluation audio out of optimization and model selection."""
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent


def assert_development_allowed(rows):
    registry_path = PROJECT / 'dataset_roles.json'
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    reserved_root = (PROJECT / registry.get('root', 'dataset/test_only_reserve')).resolve()
    names = {r['name'] for r in registry.get('datasets', [])}
    for row in rows:
        path = Path(row['path'])
        path = (path if path.is_absolute() else PROJECT / path).resolve()
        blocked = (row.get('role') == 'test_only' or reserved_root == path
                   or reserved_root in path.parents
                   or row.get('dataset', row.get('domain')) in names
                   or 'shotspotter' in str(path).lower())
        if blocked:
            raise ValueError(f'Test-only data cannot train, validate, or select thresholds: {path}')
