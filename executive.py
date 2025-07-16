import os
import argparse
from processor import Processor

class Executive:
    """
    A class dedicated to executing the specified Fortran module
    using the Processor class.
    
    Methods:
        execute(target_module): Compiles and runs the specified target module.
    """
    
    @staticmethod
    def execute(target_module, mode="CPU"):
        """
        Executes the specified Fortran module.
        
        Parameters:
            target_module (str): The Fortran file name to compile and run.
        """
        target_module_dir = os.path.join(os.getcwd(), target_module.split('.')[0])
        
        print(f"Executing compilation for {target_module} in directory {target_module_dir}")
        Processor().compile_and_run(os.getcwd(), target_module_dir, mode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute the specified Fortran module.")
    parser.add_argument("--target_module", type=str, help="The Fortran file name to compile and run.")
    parser.add_argument("--mode", type=str, default="CPU", choices=["CPU", "GPU"], 
                        help="The compilation mode: 'CPU' or 'GPU' (default: 'CPU').")
    
    args = parser.parse_args()
    Executive.execute(args.target_module, mode=args.mode)

