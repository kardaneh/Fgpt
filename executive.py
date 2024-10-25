import os
from processor import Processor

class Executive:
    """
    A class dedicated to executing the specified Fortran module
    using the Processor class.
    
    Methods:
        execute(target_module): Compiles and runs the specified target module.
    """
    
    @staticmethod
    def execute(target_module):
        """
        Executes the specified Fortran module.
        
        Parameters:
            target_module (str): The Fortran file name to compile and run.
        """
        target_module_dir = os.path.join(os.getcwd(), target_module.split('.')[0])
        
        print(f"Executing compilation for {target_module} in directory {target_module_dir}")
        Processor().compile_and_run(os.getcwd(), target_module_dir)

if __name__ == "__main__":
    Executive.execute("hydrol.f90")

