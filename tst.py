import os
from processor import Processor
target_module = "hydrol.f90"
target_module_dir = os.path.join(os.getcwd(), target_module.split('.')[0])
Processor().compile_and_run(os.getcwd(), target_module_dir)

