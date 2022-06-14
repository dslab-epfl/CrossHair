import re

from crosshair.fnutil import FunctionInfo
from crosshair.options import DEFAULT_OPTIONS
from crosshair.path_cover import CoverageType, path_cover


def _foo(x: int) -> int:
    if x > 100:
        return 100
    return x


def _regex(x: str) -> bool:
    compiled = re.compile("f(o)+")
    return bool(compiled.fullmatch(x))


OPTS = DEFAULT_OPTIONS.overlay(max_iterations=10, per_condition_timeout=10.0)
foo = FunctionInfo.from_fn(_foo)
regex = FunctionInfo.from_fn(_regex)


def test_path_cover() -> None:
    paths = []
    path_list, exausted = path_cover(foo, OPTS, CoverageType.OPCODE)
    assert exausted
    for p in path_list:
        paths.append(p.result[1]["ret"])
    print(paths)
    assert len(paths) == 2
    path_str = list(map(lambda p: str(p), paths))
    assert "100" in path_str and "x_2" in path_str


def test_path_cover_regex() -> None:
    paths, exausted = path_cover(regex, OPTS, CoverageType.OPCODE)
    input_output = set((p.args.arguments["x"], p.result[1]["ret"]) for p in paths)
    assert ("foo", True) in input_output
