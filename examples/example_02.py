import sys

from fgpt import Isolator
from fgpt.core.common import Logger

# Fgpt Imports
from fgpt.core.frontend import Processor
from fgpt.isolator import parse_args

# Initialize Fgpt
logger = Logger()
processor = Processor(logger=logger)

# Define the options
sys.argv = [
    "my_program",
    "--rest_of_path",
    "Fgpt/examples",
    "--target_module",
    "example_01",
    "--work",
    "/home/kardaneh",
    "--target_model",
    "general",
    "--parent_subroutine",
    "test_procedures",
    "--target_subroutines",
    "test_procedures",
    "--openacc",
    "False",
    "--f2py",
    "False",
    "--py2jx",
    "False",
    "--tapenade",
    "False",
    "--mode",
    "jax",
    "--benchmark_dir",
    "/tmp/benchmark",
    "--config_path",
    "/home/kardaneh/Fgpt/template.yaml",
    "--vectorize",
    "kjpindex",
    "nvm",
    "npts",
]

args = parse_args()

isolator = Isolator(
    rest_of_path=args.rest_of_path,
    target_model=args.target_model,
    target_module=args.target_module,
    work=args.work,
    config_path=args.config_path,
    openacc=args.openacc,
    tapenade=args.tapenade,
    f2py=args.f2py,
    py2jx=args.py2jx,
)

isolator.run(
    benchmark_dir=args.benchmark_dir,
    vectorize=args.vectorize,
    mode=args.mode,
    parent_subroutine=args.parent_subroutine,
    target_subroutines=args.target_subroutines,
)
