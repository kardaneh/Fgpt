import ast
from dataclasses import dataclass, field
from typing import Dict,List,Optional,Callable,Any

@dataclass
class IntrinsicSignature:
    name: str
    args: List[str]                     # canonical order
    optional: set[str]                  # optional args
    defaults: Dict[str, Any]            # default values if omitted
    varargs: bool = False               # e.g. MAX, MIN
    arg_map: Dict[str, str] = None      # Fortran -> Python names
    transform: Optional[Callable] = None

def normalize_intrinsic_call(signature: IntrinsicSignature,
                            positional_args: list,
                            keyword_args: dict) -> dict:

    # Step 1: initialize with defaults
    normalized = dict(signature.defaults)

    # Step 2: positional arguments
    if signature.varargs:
        normalized[signature.args[0]] = positional_args
    else:
        for arg_name, value in zip(signature.args, positional_args):
            normalized[arg_name] = value

    # Step 3: keyword arguments override
    for key, value in keyword_args.items():
        normalized[key.lower()] = value

    # Step 4: required args check
    for arg in signature.args:
        if arg not in signature.optional and normalized.get(arg) is None:
            raise ValueError(f"Missing required argument: {arg}")

    # Step 5: transform
    if signature.transform:
        normalized = signature.transform(normalized)

    return normalized

def dim_to_axis(dim_ast: ast.AST):
    if dim_ast is None:
        return None

    if isinstance(dim_ast, ast.Constant) and isinstance(dim_ast.value, int):
        return ast.Constant(value=dim_ast.value - 1)

    # fallback: symbolic expression
    return ast.BinOp(
        left=dim_ast,
        op=ast.Sub(),
        right=ast.Constant(value=1)
    )

SUM = IntrinsicSignature(
    name="SUM",
    args=["array", "dim", "mask"],
    optional={"dim", "mask"},
    defaults={"dim": None, "mask": None},
    arg_map={
        "array": "a",
        "axis": "axis",
        "mask": "where"
    },
    transform=lambda args: {
        "array": args["array"],
        "axis": dim_to_axis(args["dim"]),
        "where": args["mask"]
    }
)

MAXLOC = IntrinsicSignature(
    name="MAXLOC",
    args=["array", "dim", "mask"],
    optional={"dim", "mask"},
    defaults={"dim": None, "mask": None},
    arg_map={"array": "a", "axis": "axis"},
    transform=lambda args: {
        "array": args["array"],
        "axis": dim_to_axis(args["dim"])
    }
)

MINLOC = IntrinsicSignature(
    name="MINLOC",
    args=["array", "dim", "mask"],
    optional={"dim", "mask"},
    defaults={"dim": None, "mask": None},
    arg_map={"array": "a", "axis": "axis"},
    transform=lambda args: {
        "array": args["array"],
        "axis": dim_to_axis(args["dim"])
    }
)

SQRT = IntrinsicSignature(
    name="SQRT",
    args=["array"],
    optional=set(),
    defaults={},
    arg_map={"array": "a"}
)

MAXVAL = IntrinsicSignature(
    name="MAXVAL",
    args=["array", "dim", "mask"],
    optional={"dim", "mask"},
    defaults={"dim": None, "mask": None},
    arg_map={"array": "a", "axis": "axis"},
    transform=lambda args: {
        "array": args["array"],
        "axis": dim_to_axis(args["dim"])
    }
)

MINVAL = IntrinsicSignature(
    name="MINVAL",
    args=["array", "dim", "mask"],
    optional={"dim", "mask"},
    defaults={"dim": None, "mask": None},
    arg_map={"array": "a", "axis": "axis"},
    transform=lambda args: {
        "array": args["array"],
        "axis": dim_to_axis(args["dim"])
    }
)

RESHAPE = IntrinsicSignature(
    name="RESHAPE",
    args=["source", "shape", "pad", "order"],
    optional={"pad", "order"},
    defaults={
        "pad": None,
        "order": ast.Constant(value="F")  
    },
    arg_map={
        "source": "a",
        "shape": "newshape",
        "order": "order"
    },
    transform=lambda args: {
        "source": args["source"],
        "newshape": args["shape"],
        "order": args["order"]
    }
)

PRODUCT = IntrinsicSignature(
    name="PRODUCT",
    args=["array", "dim", "mask"],
    optional={"dim", "mask"},
    defaults={"dim": None, "mask": None},
    arg_map={
        "array": "a",
        "axis": "axis",
        "mask": "where"
    },
    transform=lambda args: {
        "array": args["array"],
        "axis": dim_to_axis(args["dim"]),
        "where": args["mask"]
    }
)

MIN = IntrinsicSignature(
    name="MIN",
    args=["values"],
    optional=set(),
    defaults={},
    varargs=True,
    arg_map={"values": "values"}
)

MAX = IntrinsicSignature(
    name="MAX",
    args=["values"],
    optional=set(),
    defaults={},
    varargs=True,
    arg_map={"values": "values"}
)

DOT_PRODUCT = IntrinsicSignature(
    name="DOT_PRODUCT",
    args=["a", "b"],
    optional=set(),
    defaults={},
    arg_map={"a": "a", "b": "b"}
)

MATMUL = IntrinsicSignature(
    name="MATMUL",
    args=["a", "b"],
    optional=set(),
    defaults={},
    arg_map={"a": "a", "b": "b"}
)

# Defines the ensemble of intrinsic signature which requires some modificaiton 
# due to the fact that they don't represent mirror like transformation
intrinsic_signatures = {
    "SUM": SUM,
    "PRODUCT": PRODUCT,
    "MAXVAL": MAXVAL,
    "MINVAL": MINVAL,
    "MAXLOC": MAXLOC,
    "MINLOC": MINLOC,
    "MATMUL": MATMUL,
    "DOT_PRODUCT": DOT_PRODUCT,
    "RESHAPE": RESHAPE,
    "MIN": MIN,
    "MAX": MAX,
    "SQRT": SQRT,
}