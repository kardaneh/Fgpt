# Fgpt

## Table of Contents

1. [Introdction](#introduction)
2. [Project Structure](#project-structure)
3. [Code Transformation Core](#code-transformation-core)  
    A. [Processor](#processor)  
    B. [Extractor](#extractor)  
    C. [Navigator](#navigator)  
    D. [Isolator](#isolator)  
    E. [Modifier](#modifier)  
    F. [Exectuive](#executive)  
    G. [F2NP](#f2np)  
    H. [Shaper](#shaper)
4. [Installation](#installation)
5. [Configuration](#configuration)

## Introduction

The purpose of this project is to transform a FORTRAN-based codebase into a more Pythonic and NumPy-oriented architecture. The initial phase of this transformation followed the F2NP approach, which involved converting **FORTRAN code** into plain Python. However, this method presented several issues, particularly related to **inconsistent code structure and indentation errors**. Since FORTRAN syntax differs significantly from Python’s, especially in terms of block scoping and indentation-based logic, the resulting plain Python code was often difficult to read, maintain, or execute correctly.

To address these challenges, the project shifted from a plain code transformation to an **AST (Abstract Syntax Tree)–based approach.** This method enables a more structured and semantic translation by parsing the FORTRAN code into an intermediate representation and then programmatically constructing valid, well-structured Python code. The AST approach ensures better syntactic correctness, enables deeper analysis and transformation of the code logic, and provides a scalable path to integrate NumPy optimizations and Pythonic design patterns.

## Project Structure
```
.
├── arch-nvhpc_HAL.env            # Environment file: modules to load for HAL system
├── arch-nvhpc_LEONARDO.env       # Environment file: modules to load for LEONARDO system
├── arch-nvhpc_spirit.env         # Environment file: modules to load for SPIRIT system
│
├── batch.sh                      # Batch script for running Python scripts on SCRUM
├── LICENSE                       
├── README.md                     
├── Makefile                      # Build and automation rules
|
├── org.f90                       # Original Fortran source file
├── mod.f90                       # Modified/generated Fortran module
│
├── executive.py                  # Executes and tests the generated .f90 files
├── extractor.py                  # Extracts subroutine/function information from Fortran
├── f2np.py                       # Converts Fortran code to plain Python
├── isolator.py                   # Isolates parent/child Fortran code blocks
├── line_length.py                
├── modifier.py                   # Adds CUDA-related code to .f90 files
├── navigator.py                  # Navigates directory tree to locate subroutines/variables
├── processor.py                  # Parses Fortran files using `fparser`, builds ASTs, and generates code
├── shaper.py                     # Determines size and intent of variables in Fortran modules
├── transformer.py                # Transforms Fortran code to Python via AST transformations
│
└── tst_f2np.py                  # Example of transformation done using F2NP approach on an isolated subroutine. 
```

## Code Transformation Core

The Code Transformation Core is the central engine of the project, responsible for parsing, analyzing, and transforming legacy Fortran code into Python or enhanced Fortran variants. It provides a modular pipeline that isolates subroutines, extracts metadata, builds abstract syntax trees (ASTs), and applies transformations. These components enable flexible and automated modernization of scientific codebases while preserving original logic and structure. 

### Processor 

The Processor class serves as the backbone of the Code Transformation Core, responsible for parsing Fortran code and constructing abstract syntax trees (ASTs) using the fparser library. It enables both high-level analysis and precise manipulation of code structures for transformation purposes. Parsing can be performed from a file, a raw string, or individual Fortran statements, making it highly flexible.

Core Responsibilities:

- **AST Generation**: Converts Fortran code into structured ASTs from files, strings, or statements.

- **Separation & Duplication**:
    - `separate_entity_declaration` and `separate_entity_allocation`: Split out individual declarations and allocations into unique statements for finer control.
    - `add_entity_declaration` and `add_entity_allocation`: Duplicate and modify the original entities to produce two versions: CPU and original for later comparison between the accelerated GPU version through binary precision control.

- **Declaration & Allocation Handling**:

    - `combine_allocate_declaration`: Merges allocatable declarations and associated allocate calls into a single, coherent statement.
    - `map_declaration`: Ensures variable dimensions are consistent across implicit and explicit declarations, supporting dynamic and implicit allocation patterns.

- **Validation & Checks**:

    - check_point: Verifies dimensional consistency and logical conditions (e.g., binary operations) between versions(CPU,GPU)

- **Output Generation**:

    - `out_module_fortran`: Generates a plain module_global.F90 file which will be progressively filled to contain shared declarations and utility routines.
    - `out_main_fortran`: Generates a plain main.F90 file,which will be filled with CALL and other specification elements and subroutine definitions. 

- **Transformation Support**:

    - `create_call_stmt`: Generates call statements for subroutine invocations, particularly useful with dummy variables.
    - `initialization_statement`: Automates initialization of variable sets and includes write statements for I/O if needed.
    - `process_queue`: Collects and organizes variable declarations (e.g., parameters and allocatables) for consistent handling by placing the scalars first, then arrays and finally other parameters. 

- **Update Mechanisms**:

    - `update_main` and `update_global_module`: Insert transformed or original code into the main and global files respectively, also handling instrumentation such as timing.
    
- **Support Methods**:

    - `compile_and_run`: Compiles and executes Fortran files.
    - `process_assign`: Handles assignment logic during analysis or transformation.

### Extractor 

The Extractor class is responsible for identifying, cleaning, and analyzing subroutines and variables within Fortran code. It operates directly on the parsed AST (generated by Processor) and focuses on extracting only the relevant external subroutines and their variable metadata, excluding unnecessary or internal elements. This class forms a critical part of understanding the structure and data flow in the Fortran codebase.

Core Responsibilities:
- **Subroutine Extraction & Filtering**

    - `find_subroutines`: Identifies all subroutines and their argument list, Distinguishes between internal and external subroutines, Filters out subroutines that don’t have corresponding call statements or are not listed as externally allowed, Maps dummy arguments to subroutines using key–value pairs, Allows external interface subroutines to be identified via the dummy arg list.
    - `clean_subroutine`: Uses `traverse_subroutine` to ensure extracted subroutines structurally match the AST version, Verifies declarations and call structure integrity, Cleans redundant or inconsistent declarations, Handles specific cases like `entity_declaration` cleanup.

- **Variable Analysis**

    - `find_variables`: Analyzes each subroutine to identify: Local variables (declared and used within the subroutine), Global variables (used in subroutine but declared externally), Dummy arguments with and without intent (IN, OUT, INOUT), Uses Fparser constructs like Intent_Attr_Spec to classify variables, Identifies modified variables to separate outputs from inputs.

    - `find_global_variables`: Traverses the broader codebase (via the Navigator class) to find and collect global variables referenced within a subroutine.
    - `extract_array_info`: Collects dimensional and shape information for arrays declared or used in subroutines, Checks global variables to see if they are modified (for output classification), Determines if they require dynamic allocation, Leverages Processor.combine_allocate_declaration() to unify allocation and declaration where needed, Handles both explicit shape specs and dynamically sized arrays.
    - `process_declaration_variables`: Final step in preparing variable metadata, resolving shapes, dimensions, and allocation needs across both local and dummy variables.

- **Intent & Declaration Utilities**
    - `add_intent`: Recursively traverses the subroutine block (traverse_block) to analyze how variables are used and sets their intent accordingly.

    - `remove_intent_save`: Cleans up redundant or unnecessary attributes like intent, save, or public.

- **Other methods**:
    - `extract_loop_vect` and `extract_indices`: both methods have there own usage, the first one is usef to retrieve the variables name that can be used outside of the loop as global for loop aspect and the latter can be used to extract indices present in the DO conditions as well as their upper and lower bounds values or variables. 

### Navigator

The Navigator class is responsible for traversing Fortran code modules using a breadth-first (row-by-row) search strategy to identify specific variables or external subroutines. It leverages a queue-based system to explore child modules, track visited modules, and dynamically build relationships between code elements. This class plays a key role in understanding module-level dependencies and the hierarchical structure of variable and subroutine usage across the codebase.

Core Responsibilities:

**Breadth-First Traversal of Fortran Modules**:  
- Utilizes BFS to explore module hierarchies efficiently.
- Maintains state through:  
    - `return_key_sc` (bool): Indicates if the target variable or subroutine was found.  
    - `visited_modules_sc` (set): Prevents redundant traversal of already visited modules.  
    - `child_modules_sc` (set): Tracks direct child modules encountered during search.  
    - `module_set_sc` (set): Records all modules traversed during the search.  

**Variable Resolution and Analysis**:  

- `find_variable_in_module`:
    Locates a specific variable within the current module, checks its allocation status, and stores relevant metadata for further analysis.

- `variable_finder`:
    Initiates a variable search across modules. Adds the variable and its containing module to the visited list and queue if found within a child module.

- `find_var_in_child_modules`:
    Looks for the variable in child modules, updates the module tree structure, and expands the search queue accordingly.

**External Subroutine Discovery**:

- `find_external_subroutines_in_module`:
    Identifies external subroutines within the current module. Searches for both Interface_stmt and Subroutine_Subprogram declarations, and injects appropriate use statements where needed.

- `external_subroutine_finder`:
    Starts the search for an external subroutine by adding it to the queue and marking it as visited. Checks for its existence in child modules and expands the search accordingly.

**Queue & Traversal Utilities**:

- `add_modules_to_queue`:
    Helper function to enqueue target entities (variables or subroutines) for continued breadth-first traversal.

### Isolator

The Isolator class is responsible for isolating subroutines and functions within Fortran code, using both the Extractor, Processor and Modifier classes to extract, clean, and modify the necessary elements. It ensures that each subroutine or function is treated independently, creating isolated versions of the code that can be further analyzed or optimized. The class handles both parent and child subroutines/functions, ensuring that interdependencies are managed and that the final code structure is clean and modular.  This class works in a hierarchical manner, first isolating child subroutines/functions before addressing parent subroutines.

Core Responsibilities:

- `create_target_directory`:
    Creates separate directories for each module to store independent outputs of functions and subroutines, keeping them isolated and organized for further processing.  

- `run`:
    Coordinates the overall process by calling create_target_directory and invoking process_subroutines.
    Uses the Extractor class to identify subroutines and extract loop indices, isolating both parent and child subroutines and functions for further analysis or optimization. The process follows a children → parent isolation flow, ensuring that each element is processed in its proper hierarchical order.

**Subroutine and Function Differentiation**:

- **Subroutines**: Can return multiple variables and use the CALL statement for invocation.
- **Functions**: Can only return a single variable and are invoked directly by their name.
    These differences are essential for handling their isolation and optimization correctly.


**Process of isolation for the child subroutine**:

The isolation of child subroutines begins by extracting the subroutine tree using the cls.subroutine[subroutine_key]. The following steps are performed:

- Extraction:
    - extract_intent()
    - clean_subroutine()
    - find_variables()
    - extract_names()
    - Processor().parse_fortran_string() (for working tree)

- Variable Handling:
    - Global and Local Variables are processed using find_global_variables() and process_declaration_variables() to retrieve their types and dimensions.

    - Shape Information: Variables are categorized into shapes_variables, scalar_variables, and var_global. The subtraction operation (cls.shapes_variables[subroutine_key] - cls.scalar_variables[subroutine_key] - cls.var_global[subroutine_key]) helps identify variables with unknown shapes, updating global variables as needed.

    - Array Information: Extracts array data from both global and dummy variables.

**Isolation Flow**:

- `isolate_parent_subroutine` and `isolate_child_subroutine`:
    The isolation process begins with children and proceeds to the parent subroutine. This ensures that child subroutines and functions are isolated before their parent subroutine is processed.

**Modifier Class Integration**:

The Modifier class is used extensively throughout the isolation process to modify and optimize the code for GPU support.

**Final Processing**:

- **Global and Main Program Updates**:
    After isolating child subroutines, the Processor class is used to update the global module and main program files. These files include the necessary global declarations and any modifications resulting from the isolation process. The update_global_module and update_main_program methods ensure that the isolated code is correctly generated and stored.

- **Parent Subroutine**:
    After isolating the child subroutines and functions, the parent subroutine is then isolated and processed in the same way, ensuring consistency in the modularization process.

### Modifier

The Modifier class is primarily responsible for transforming Fortran code, applying computational, array-related, and hardware-specific optimizations, particularly with a focus on GPU support. It works closely with the Processor class to modify Fortran statements, handle unsupported intrinsic functions, and optimize loops and array accesses for more efficient computation. The transformations include handling array operations, reducing array dimensions, and replacing unsupported functions with manual loops.

Core Responsibilities:

**Code Transformation & Optimization**:

- **Array-Related Transformations**:
    Modifies the handling of arrays and loops, particularly for optimization on GPU architectures. Includes adding DO loops for array assignments and ensuring proper handling of array dimensions.

- **Handling Unsupported Functions**:
    Replaces unsupported intrinsic functions (e.g., MAXLOC, MINLOC) with manual loops or alternative methods that are compatible with the current Fortran version or GPU architecture.

- **Intrinsic Function Support**:
    Processes intrinsic functions and handles vector reductions, modifications, and assignments efficiently.


**Array & Loop Transformations**:

- `traverse_expression`:
    Recursively processes arrays and their names. If arrays have child elements (e.g., subarrays), the method continues to traverse them, ensuring that all relevant subarrays or array expressions are correctly handled.

- `add_dos`:
    Adds DO statements for array assignments, automatically handling the creation of loops for multi-dimensional arrays. This method creates loops to iterate over array elements based on their dimensions. 

- `extract_intrinsic_names`:
    Retrieves intrinsic methods (e.g., MAX, MIN, etc.) from the Fortran code and processes their arguments.

    - If the intrinsic function is part of the mathematical functions attribute, it returns the parent node, which consists of two child nodes: the function itself (left) and the function’s arguments (right).

    - It identifies reduction operations (e.g., vector reductions) and classifies the function’s arguments by their dimensions (e.g., dim_key for the argument name and dim_value for the argument's value).

    - The output is a dictionary mapping intrinsic function names to their argument and dimension specifications, providing a structured representation of the intrinsic functions in use.

- `process_section_subscript_list`:
    Processes the section subscript list to identify dimensions for reduction operations. This is particularly useful for recognizing operations that apply reductions across all dimensions (e.g., summing all elements in a matrix).

**Array Modification Methods**:

- `modify_colon_array`:
    Modifies an array’s section subscript list based on whether it involves a reduction operation (e.g., ALL) or just a dimension modification. It adjusts the array’s bounds and modifies the section subscripts accordingly, ensuring that any vectorized operation is appropriately handled.

- `modify_colon_array_vec`:
    Similar to modify_colon_array, but specifically handles cases where the upper bound is dynamically determined (e.g., kjpindex). This method ensures that the array bounds are properly adjusted for dynamic indices.

- `remove_vec_for_locals_in_assigns`:
    Removes vector dimensions for local arrays in assignments, leaving global arrays intact. 

**Declaration and Specification Modifications**:

- `traverse_declaration_statement`:
    Traverses a declaration statement in the Fortran code, constructing the type declaration statement for variables. This method uses an inner function `inner_traverse` to recursively go through the children first, then backtrack to handle the grandparent nodes.

- `modify_specification_part`:
    Modifies the entity declaration in the Fortran code. This includes adding assumed shape specifications and enabling GPU support. The method adjusts the specification to ensure compatibility with GPU architecture by adding attributes or directives for hardware-specific operations.

- `remove_vec_for_locals_in_specification`:
    Similar to remove_vec_for_locals_in_assigns, but this method focuses on local array declarations. It removes unnecessary vector dimensions for local arrays in the declaration statements. 

**Unsupported Function Handling**:

- `replace_unsupported_function_with_manual_loop`:
    Replaces unsupported functions (e.g., MINLOC, MAXLOC) in Fortran assignments with manual loops. Since these functions may not be supported in certain versions or GPU environments, they are manually rewritten as loops to achieve the same functionality.

- `replace_vec_colon_with_index`:
    Converts array slicing using the colon operator (:) into indexed loops. 

**Loop Optimization**:

- `merge_vector_loop`:
    Merges vector loops by modifying child loops to match vectorized operations. The method loops through all conditional statements, recursively adding them to the block. It also considers DO and IF statements, and removes irrelevant statements like WRITE, OPEN, and CLOSE that may interfere with vectorized operations.

- `replace_gpu_unsupported`:
    Modifies WRITE conditions and other I/O operations to be compatible with GPU-based systems. Since GPUs do not support traditional I/O operations like WRITE or CLOSE, this method replaces these operations with flags or alternative mechanisms that are GPU-friendly.

### Executive 

The Executive class is responsible for testing the isolated subroutines after they have been compiled and generated the binary values files. Once the isolated subroutines are available/isolated within the target folder, the Executive class iterates through them and tests them in a given mode (CPU or GPU) to ensure they function correctly and efficiently.

Core Responsibilities:

- `execute`:
    The core method of the Executive class. It is responsible for running the isolated subroutines in the chosen mode and iterating over them to perform the testing.

    - **Arguments**:

        - target_module: Specifies the target module (subroutine/module) to be tested.

        - mode {CPU, GPU}: Specifies the mode for testing—either CPU or GPU. The subroutines are tested in the selected mode to ensure correctness and performance.

    - **Process**:
        The method iterates over all the isolated subroutines and runs them in the specified mode. It performs the testing and gathers the results, ensuring that the isolated subroutines perform correctly under both CPU and GPU environments.

### F2NP

The F2NP class is designed to convert Fortran code into Python code using NumPy, translating each Fortran construct line by line. This includes converting variable declarations, loops, conditions, intrinsic functions, array indexing, and more. The result is a Pythonic representation of the original Fortran logic, enabling easier analysis, prototyping, or adaptation in modern Python-based workflows.

Core Responsibilities:

Convert Fortran source code to NumPy-compatible Python code, translating:

- Subroutines to functions
- Calls to Python function calls
- Type declarations to np.array initializations
- DO loops to Python for loops
- PRINT statements to print()
- Array slicing and assignments with adjusted indexing

**Key Methods**:

- `recursive`:
    Core recursive traversal function that walks through the entire AST. It dispatches nodes to appropriate handler functions based on their type, maintaining and adjusting indentation levels for nested constructs (e.g., loops, conditions).

**Statement Handlers**:

- `handle_end_stmt`:
    Converts Fortran END statements (e.g., END DO, END SUBROUTINE) into plain Python comments or closing lines (like # end do), marking the logical end of a block without actual structural necessity in Python.

- `handle_specification`:
    Converts Fortran variable declaration lines (that don't include type/shape) into Python comments for traceability.

- `handle_type_declaration_stmt`:
    Translates Fortran type declarations (e.g., REAL, DIMENSION(...) :: A) into NumPy array initializations. Determines the data type (REAL, INTEGER, etc.) and the shape of arrays using bounds parsed from the declaration.

- `handle_do_stmt`:
    Converts Fortran DO loops into Python for loops. Adjusts loop bounds using range() and handles the difference between Fortran's 1-based and Python's 0-based indexing.

- `simplify_limits`:
    Simplifies expressions for loop bounds or array dimensions, collapsing constant additions or subtractions and cleaning up arithmetic for clearer, more Pythonic loops.

- `handle_where`:
    Converts Fortran WHERE statements into Python np.where() expressions. Extracts the conditional logic from the left side of the statement to use in NumPy-style broadcasting.

- `handle_part_ref`:
    Extracts array slicing information from Fortran statements (via Part_Ref nodes), including upper and lower bounds. These are translated into Python array slice expressions, properly adjusted to Python’s indexing.

- `handle_intrinsic_function_reference`:
    Replaces Fortran intrinsic functions with their Python equivalents (e.g., MAX → np.maximum). Uses a predefined intrinsic_replacement map to translate known functions.

- `handle_print_stmt`:
    Translates Fortran PRINT statements into Python print() calls. Ensures format compatibility and basic data presentation during output.

### Shaper

The Shaper class is responsible for preparing and reshaping subroutines and functions by navigating through the codebase and identifying their arguments, dimensions, and proper usage context. It works in conjunction with the Processor and Navigator classes to analyze where and how subroutines and functions are called, and generates appropriate Fortran declarations for their arguments.

Core Responsibilities:

- `shaper_subroutine`:
    Identifies subroutines that are called from outside their current module and reconstructs their argument list with accurate declarations and dimensional information.

    - Uses the Navigator class to locate the source of the subroutine, retrieves its arguments, analyzes each argument’s usage and shape, and produces the corresponding Fortran declaration statements.

- `shaper_function`:
    Applies the same logic as shaper_subroutine, but for functions instead of subroutines.

- `shaper_intrinsic_size`:
    Determines the intrinsic size (i.e., dimensional characteristics) of variables used in subroutine or function arguments.
    - Analyzes the AST or context of a variable to infer its array dimensions and size based on its intrinsic properties or usage pattern.

- `find_enclosing_subroutine`:
    Finds the parent or enclosing subroutine of a given AST node.


- `find_fortran_files_subroutines`:
    Searches across files to locate where a given subroutine is defined or referenced.

    - Scans a given Fortran file and potentially other related files to identify subroutine declarations or calls, aiding in cross-module analysis.

## Installation

## Configuration