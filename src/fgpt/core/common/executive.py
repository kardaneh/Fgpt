# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Shivamshan Sivanesan, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import argparse
import os

from fgpt.core.common.logger import Logger
from fgpt.core.frontend.processor import Processor


class Executive:
    """
    A class dedicated to executing the specified Fortran module
    using the Processor class.
    """

    @staticmethod
    def run(target_module, mode="CPU"):
        """
        Executes the specified Fortran module.

        Parameters:
            target_module (str): The Fortran file name to compile and run.
        """
        logger = Logger(console_output=True, file_output=True, record=True)
        logger.show_header("executive")
        processor = Processor(logger=logger)
        target_module_dir = os.path.join(os.getcwd(), target_module.split(".")[0])

        print(
            f"Executing compilation for {target_module} in directory {target_module_dir}"
        )
        for subdir in os.listdir(target_module_dir):
            subdir_path = os.path.join(target_module_dir, subdir)
            processor.compile_and_run(os.getcwd(), subdir_path, mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute the specified Fortran module."
    )
    parser.add_argument(
        "--target_module", type=str, help="The Fortran file name to compile and run."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="CPU",
        choices=["CPU", "GPU"],
        help="The compilation mode: 'CPU' or 'GPU' (default: 'CPU').",
    )

    args = parser.parse_args()
    Executive.run(args.target_module, mode=args.mode)
