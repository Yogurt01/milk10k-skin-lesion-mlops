"""Every third-party module imported under serving/src is declared in serving/pyproject.toml (core or an extra)."""

import ast
import re
import sys
import tomllib
from importlib.metadata import packages_distributions

import pytest

from .conftest import ROOT

pytestmark = pytest.mark.unit
SRC = ROOT / "serving" / "src"


def _declared() -> set[str]:
    project = tomllib.loads((ROOT / "serving" / "pyproject.toml").read_text())["project"]
    names = list(project["dependencies"]) + [
        d for extra in project.get("optional-dependencies", {}).values() for d in extra
    ]
    return {re.split(r"[\s<>=!~;\[]", n, maxsplit=1)[0].lower().replace("_", "-") for n in names}


def _imported_top_level() -> set[str]:
    modules = set()
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    return {m for m in modules if m not in sys.stdlib_module_names and m != "milk10k_serving"}


def test_every_third_party_import_is_declared():
    dists = packages_distributions()
    undeclared = {}
    for module in sorted(_imported_top_level()):
        names = {d.lower().replace("_", "-") for d in dists.get(module, [module])}
        if not names & _declared():
            undeclared[module] = sorted(names)
    assert not undeclared, f"imported under serving/src but not declared in serving/pyproject.toml: {undeclared}"
