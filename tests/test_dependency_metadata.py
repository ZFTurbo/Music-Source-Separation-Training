"""Keep installation paths aligned on runtime dependencies."""
from pathlib import Path

from packaging.requirements import Requirement


def test_runtime_dependency_requirements_are_consistent():
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    root = Path(__file__).resolve().parents[1]
    with (root / 'pyproject.toml').open('rb') as file:
        metadata = tomllib.load(file)
    project = {req.name.lower(): str(req.specifier)
               for req in map(Requirement, metadata['project']['dependencies'])}
    requirements = {req.name.lower(): str(req.specifier)
                    for req in map(Requirement, (root / 'requirements.txt').read_text().splitlines())}
    assert project['torch'] == requirements['torch'] == '>=2.6.0'
    assert project['filelock'] == requirements['filelock'] == '>=3.12'
