import copy
import dataclasses
import enum
import time
from dataclasses import dataclass
from inspect import BoundArguments, Signature
from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple, Type

from crosshair.condition_parser import condition_parser
from crosshair.core import ExceptionFilter, Patched, deep_realize, gen_args
from crosshair.fnutil import FunctionInfo
from crosshair.libimpl.builtinslib import SymbolicNumberAble
from crosshair.options import AnalysisOptions
from crosshair.statespace import (
    CallAnalysis,
    RootNode,
    StateSpace,
    StateSpaceContext,
    VerificationStatus,
)
from crosshair.tracers import COMPOSITE_TRACER, NoTracing
from crosshair.util import (
    CoverageResult,
    UnexploredPath,
    debug,
    measure_fn_coverage,
    name_of_type,
)


class CoverageType(enum.Enum):
    OPCODE = "OPCODE"
    PATH = "PATH"


@dataclass
class PathSummary:
    args: BoundArguments
    result: Tuple[List[Any], Dict[str, object]]
    exc: Optional[Type[BaseException]]
    post_args: BoundArguments
    coverage: CoverageResult


def dataclass_ret_to_dict(ret) -> dict[str, Any]:
    # convert a returned dataclass to a dict of [variable_name,z3_expr or value]
    if not dataclasses.is_dataclass(ret):
        debug("unprocessed ret")
        return ret
    tmp = {}
    for f in dataclasses.fields(ret):
        dataclass_val = getattr(ret, f.name)
        z3var = extract_value_of_return_var(dataclass_val)
        tmp[f.name] = z3var
    return tmp


def run_iteration(
    fn: Callable, sig: Signature, space: StateSpace
) -> Optional[PathSummary]:
    with NoTracing():
        args = gen_args(sig)
    pre_args = copy.deepcopy(args)
    ret = None
    return_dict: Dict[str, object] = {}
    with measure_fn_coverage(fn) as coverage, ExceptionFilter() as efilter:
        # coverage = lambda _: CoverageResult(set(), set(), 1.0)
        # with ExceptionFilter() as efilter:
        ret = fn(*args.args, **args.kwargs)
        with NoTracing():
            # SVSHI variables are returned either as a dictionary of dataclasses or a bool
            # The goal is to change the return of realized variables (that have a value)
            # to symbolic variables (SymbolicNumberAble) that can be used for z3 expressions
            if dataclasses.is_dataclass(ret):
                # return is only a dataclass
                dataclass_class_name = type(ret).__name__
                return_dict[dataclass_class_name] = dataclass_ret_to_dict(ret)
            elif isinstance(ret, dict):
                # return is a dict (like in SVSHI iteration)
                for k, v in ret.items():
                    return_dict[k] = dataclass_ret_to_dict(v)
            elif isinstance(ret, List):
                raise NotImplementedError("List aren't implemented yet", type(ret))
            else:
                z3var = extract_value_of_return_var(ret)
                return_dict["ret"] = z3var
    space.detach_path()
    c = copy.deepcopy(space.solver.assertions())
    if efilter.user_exc is not None:
        exc = efilter.user_exc[0]
        print("user-level exception found", repr(exc), *efilter.user_exc[1])
        return PathSummary(pre_args, (c, return_dict), type(exc), args, coverage(fn))
    elif efilter.ignore:
        return None
    else:
        return PathSummary(
            deep_realize(pre_args),
            (c, return_dict),
            None,
            deep_realize(args),
            coverage(fn),
        )


def extract_value_of_return_var(ret):
    # extracts if possible the symbolic value of the variable ret
    if isinstance(ret, SymbolicNumberAble):
        z3var = getattr(ret, "var")
    else:
        z3var = ret
    return z3var


def path_cover(
    ctxfn: FunctionInfo, options: AnalysisOptions, coverage_type: CoverageType
) -> Tuple[List[PathSummary], bool]:
    fn, sig = ctxfn.callable()
    search_root = RootNode()
    condition_start = time.monotonic()
    paths: List[PathSummary] = []
    for i in range(1, options.max_iterations):
        debug("Iteration ", i)
        itr_start = time.monotonic()
        if itr_start > condition_start + options.per_condition_timeout:
            debug(
                "Stopping due to --per_condition_timeout=",
                options.per_condition_timeout,
            )
            break
        space = StateSpace(
            execution_deadline=itr_start + options.per_path_timeout,
            model_check_timeout=options.per_path_timeout / 2,
            search_root=search_root,
        )
        with condition_parser(
            options.analysis_kind
        ), Patched(), COMPOSITE_TRACER, StateSpaceContext(space):
            summary = None
            try:
                summary = run_iteration(fn, sig, space)
                verification_status = VerificationStatus.CONFIRMED
            except UnexploredPath:
                verification_status = VerificationStatus.UNKNOWN
            debug("Verification status:", verification_status)
            top_analysis, exhausted = space.bubble_status(
                CallAnalysis(verification_status)
            )
            debug("Path tree stats", search_root.stats())
            if summary:
                paths.append(summary)
            if exhausted:
                debug("Stopping due to code path exhaustion. (yay!)")
                break
    return paths, exhausted


def repr_boundargs(boundargs: BoundArguments) -> str:
    pieces = list(map(repr, boundargs.args))
    pieces.extend(f"{k}={repr(v)}" for k, v in boundargs.kwargs.items())
    return ", ".join(pieces)


def output_argument_dictionary_paths(
    fn: Callable, paths: List[PathSummary], stdout: TextIO, stderr: TextIO
) -> int:
    for path in paths:
        stdout.write("(" + repr_boundargs(path.args) + ")\n")
    stdout.flush()
    return 0


def output_eval_exression_paths(
    fn: Callable, paths: List[PathSummary], stdout: TextIO, stderr: TextIO
) -> int:
    for path in paths:
        stdout.write(fn.__name__ + "(" + repr_boundargs(path.args) + ")\n")
    stdout.flush()
    return 0


def output_pytest_paths(
    fn: Callable, paths: List[PathSummary], stdout: TextIO, stderr: TextIO
) -> int:
    fn_name = fn.__name__
    lines: List[str] = []
    lines.append(f"from {fn.__module__} import {fn_name}")
    lines.append("")
    import_pytest = False
    for idx, path in enumerate(paths):
        test_name_suffix = "" if idx == 0 else "_" + str(idx + 1)
        exec_fn = f"{fn_name}({repr_boundargs(path.args)})"
        lines.append(f"def test_{fn_name}{test_name_suffix}():")
        if path.exc is None:
            lines.append(f"    assert {exec_fn} == {repr(path.result)}")
        else:
            import_pytest = True
            lines.append(f"    with pytest.raises({name_of_type(path.exc)}):")
            lines.append(f"        {exec_fn}")
        lines.append("")
    if import_pytest:
        lines.insert(0, "import pytest")
    stdout.write("\n".join(lines) + "\n")
    stdout.flush()
    return 0
