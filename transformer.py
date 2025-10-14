from typing import Dict, List,Literal,Optional,Tuple,Union
from fparser.two import Fortran2003 as F23
from collections import defaultdict
from fparser.two.utils import walk
from string import Template
import subprocess
import logging
import yaml
import stat
import ast
import os

from isolator import Isolator
from processor import Processor
from extractor import Extractor
from f2np import F2NP
from utils import * 

class Transformer:
    """
    Transformer class responsible for converting FORTRAN code into Python code using Abstract Syntax Trees (AST) as the intermediate representation.
    This class parses FORTRAN source code, constructs its corresponding AST, and then transforms that AST into a Python-compatible AST structure, which can be further compiled or interpreted in Python.
    """

    def __init__(self, benchmark_dir:str,
                 isolator:Isolator,
                 extractor:Extractor,
                 ignore_case:List[str],
                 config_path:str
                ):
        if benchmark_dir is None: # THe benchmark directory
            current_dir = os.getcwd()
            self.benchmark_dir = os.path.join(current_dir, 'benchmark')
            os.makedirs(self.benchmark_dir, exist_ok=True)
        else:
            self.benchmark_dir = benchmark_dir      
        self.subroutine_name = None             # Subroutine that will be isolated 
        self.ignore_case = ignore_case          # List of string of variables or functions names that are to be ignored
        self.isolator = isolator                # An instance of isolator class just used in the method retrieve_variable to retrieve variable order 
        self.extractor = extractor              # An instance of extractor class 
        self.cls_mode = False                   # Defines if we should create a class global module or not
        self.config_path = config_path          # Path to the template.yaml file 
        self.for_loop = False                   # If we want to create with either using using a for loop for the reading binary files 
        self.global_state = False               # Allows to define if the given code template is for global or not
        self.f2np = F2NP(extractor)             # Class in charge of transforming a subroutine from fortran to python

    ################################################################################# Helper functions #################################################################################
    @staticmethod
    def load_code_templates(config_path:str) -> Dict | None:
        """
        Load code templates from a YAML configuration file.

        This method reads a YAML file containing code templates and returns
        them as a dictionary. It handles file not found, YAML parsing errors,
        and any unexpected exceptions.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file containing the templates.

        Returns
        -------
        templates : dict or None
            Dictionary of code templates if the file is successfully loaded,
            otherwise `None` if an error occurs during file access or parsing.
        """
        try:
            with open(config_path, 'r') as file:
                templates = yaml.safe_load(file)
        except FileNotFoundError:
            logging.error(f"Error: The file '{config_path}' was not found.")
            templates = None
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML file: {e}")
            templates = None
        except Exception:
            logging.exception(f"An unexpected error occurred in load_code_templates")
            templates = None

        return templates
    
    def get_imports_from_specs(self,import_specs: List[Tuple[str, List[str]]]) -> List[ast.ImportFrom] | None:
        """
        Generate AST `ImportFrom` nodes for the given import specifications.

        If a class is being imported, it is assumed that the class name matches the
        module file name.

        Parameters
        ----------
        import_specs : list of tuple
            A list of tuples, where each tuple is of the form:
            (module_name: str, names_to_import: list of str)

        Returns
        -------
        import_nodes : list of ast.ImportFrom or None
            A list of `ast.ImportFrom` nodes if successful, otherwise `None`.
        """

        if not import_specs:
            return None

        import_nodes = []

        try:
            for module_name, names in import_specs:
                if not names:
                    continue

                aliases = [ast.alias(name=name) for name in names]
                import_node = ast.ImportFrom(
                    module=module_name,
                    names=aliases,
                    level=0
                )
                import_nodes.append(import_node)

            return import_nodes

        except Exception:
            # Turns out https://stackoverflow.com/questions/5191830/how-do-i-log-a-python-error-with-debug-information using logging.exception is much better than just
            # to use logging.error since it gives out the stack trace along side the error message
            logging.exception(f"Exception in get_imports_from_specs")
            return None 
    
    def create_instances(self, nodes: List, self_mode: bool = False) -> List[ast.Assign]:
        """
        Create instance assignment nodes for given class definitions.

        Depending on `self_mode`, it generates either:
        - `variable = Class()` or
        - `self.variable = Class()`

        Parameters
        ----------
        nodes : list
            List of AST nodes representing class definitions.
        self_mode : bool
            If True, generates `self.variable = Class()`.
            If False, generates `variable = Class()`.

        Returns
        -------
        instance_nodes: list
            List of `ast.Assign` nodes representing the created class instances.
        """
        instance_nodes = []
        try:
            for node in nodes:
                if not isinstance(node, ast.ClassDef):
                    raise ValueError(f"Node is not an ast.ClassDef: {node}")

                instance_name = get_instance_name(node.name)

                # Create Class() constructor call
                constructor_call = ast.Call(
                    func=ast.Name(id=node.name, ctx=ast.Load()),
                    args=[],
                    keywords=[]
                )

                # Choose target based on self_mode
                if self_mode:
                    target = ast.Attribute(
                        value=ast.Name(id='self', ctx=ast.Load()),
                        attr=instance_name,
                        ctx=ast.Store()
                    )
                else:
                    target = ast.Name(id=instance_name, ctx=ast.Store())

                assign_node = ast.Assign(
                    targets=[target],
                    value=constructor_call
                )

                instance_nodes.append(assign_node)

            return instance_nodes

        except Exception:
            logging.exception("Failed to create instance nodes in create_instances.")
            return None

    
    def create_cls_info(self, out_module: ast.Module,instance_node:List = None, self_mode:bool = False) -> Dict:
        """
        Retrieve class information including attributes and methods from an AST module.

        Parses the provided `ast.Module` and extracts structured information about each class defined within it. The result is a nested dictionary where each key is a
        class name (or instance identifier), and the value is another dictionary with two keys: `attributes` and `methods`. This also takes into account the other class
        instances intialized inside the class itself. 

        Parameters
        ----------
        out_module : ast.Module
            The abstract syntax tree (AST) of the Python module to analyze.
        instance_node : List
            Instances nodes of classes intialized inside the current class. 

        Returns
        -------
        cls_info : dict of dict
            A dictionary where each key represents a class instance, and the value is a dictionary with:
            - 'attributes': list of attribute names
            - 'methods': list of method names

        Notes
        -----
        This function assumes standard class structure and may not detect dynamically defined
        attributes or methods.
        """
        cls_info = {}
        import_nodes,instance_nodes = None,None
        try:
            class_defs = list(ast_walk(out_module, ast.ClassDef))

            if not class_defs:
                raise ValueError("There are no class definition are in this module")
            # Collect attributes and methods separately of a class 
            class_members = {}
            for class_def in class_defs:
                class_name = class_def.name
                attributes = {}
                methods = {}
                instances = {}
        
                # Collect instance attributes (self.x)
                for assign in ast_walk(class_def, ast.Assign):
                    for target in assign.targets:
                        if (
                            isinstance(target, ast.Attribute) and
                            isinstance(target.value, ast.Name) and
                            target.value.id == 'self'
                        ):
                            # attributes.add(target.attr)
                            if isinstance(assign.value, ast.Call) and isinstance(assign.value.func, ast.Name) and instance_node:
                                object_class_name = assign.value.func.id   # "Global_module_hydrol_vegupd"
                                for instance in instance_node:
                                    if instance.value.func.id == object_class_name:
                                        instances[target.attr] = {
                                            "class_name": instance.targets[0] ,
                                            "attributes": {},  # will fill later
                                            "methods": {} 
                                        }
                            # case: self.x = np.int32(val) or np.float64(val)
                            if isinstance(assign.value, ast.Call) and isinstance(assign.value.func, ast.Attribute):
                                if assign.value.func.attr in ['int32', 'float64']:
                                    if isinstance(assign.value.args[0], (ast.BinOp,ast.Constant)):
                                        
                                        evaluated_value = safe_eval_expr(assign.value.args[0])
                                        if evaluated_value is None:
                                            raise ValueError(f'evaluated_value is None')
                                        
                                        attributes[target.attr] = [
                                            evaluated_value,
                                            assign.value.func.attr       # dtype
                                        ]
                                    else:
                                        attributes[target.attr] = [
                                            assign.value.args[0],
                                            assign.value.func.attr       # dtype
                                        ]
                                        
                                elif assign.value.func.attr in ['bool'] and isinstance(assign.value.args[0], ast.Constant):
                                    attributes[target.attr] = [
                                        assign.value.args[0].value,  # constant value
                                        assign.value.func.attr       # dtype
                                    ]
                                # case: self.x = np.zeros([...], dtype=np.float64)
                                elif assign.value.func.attr in ['zeros','array'] :
                                    # using cls.all_array_info to retrieve all the array info for the global elements
                                    array_info = self.extractor.all_array_info[self.subroutine_name].get(target.attr)
                                    if array_info is None:
                                        raise ValueError(f'Information about the array is not present inside the cls.all_array_info for the array \
                                                        array :{target.attr}')
                                    # extract dtype from keywords
                                    dtype = None
                                    for kw in assign.value.keywords:
                                        if kw.arg == "dtype":
                                            if isinstance(kw.value, ast.Attribute):
                                                dtype = kw.value.attr
                                            elif isinstance(kw.value, ast.Name):
                                                dtype = kw.value.id

                                    attributes[target.attr] = [array_info,dtype]
                            # Case of self.x = self.y 
                            if isinstance(assign.value,(ast.Attribute)):
                                # First we verify that the value is that it's an attribute and not an array being affected
                                for child in ast.walk(assign.value):
                                    if isinstance(child,ast.Name):
                                        dtype = attributes.get(assign.value.attr)
                                        if dtype:
                                            attributes[target.attr] = [
                                                    assign.value.attr,  # constant value
                                                    dtype[1] # THis is because we the self.y is usually present before the self.x thus the type of self.x is that of self.y
                                                ]
                # Collect methods (def method(self): ...)
                for node in class_def.body:
                    if isinstance(node, ast.FunctionDef):
                        methods[node.name] = node

                if not instance_node:
                    class_members[class_name] = {
                        'attributes': attributes,
                        'methods': methods,
                    }
                else:
                    class_members[class_name] = {
                        'attributes': attributes,
                        'methods': methods,
                        'instances': instances
                    }

        
            # Create imports
            specs = [('module_global', [class_name]) for class_name in class_members]
            # print(ast.unparse(ast.fix_missing_locations(import_nodes[0])))
            import_nodes = self.get_imports_from_specs(specs)
            if import_nodes is None:
                raise ValueError("Import nodes are None")

            # We could have also used the class definition present inside the out_module to create the instances
            instance_nodes = self.create_instances(class_defs)
            if instance_nodes is None:
                raise ValueError("Instance nodes are None")
        
            assert len(instance_nodes) == len(import_nodes), "Import nodes and instance nodes should match"
        
            # Match instance nodes to class names
            for class_def, inst_node in zip(class_defs, instance_nodes):
                class_name = class_def.name
               
                if isinstance(inst_node, ast.Assign):
                    target = inst_node.targets[0]
                    if isinstance(target, ast.Name):
                        instance_name = target.id if not self_mode else 'self'
                        if instance_node:
                            cls_info[class_name] = {
                                instance_name: {
                                    'attributes': class_members[class_name]['attributes'],
                                    'methods': class_members[class_name]['methods'],
                                    'instances': class_members[class_name]['instances']
                                }
                            }
                        else:
                            cls_info[class_name] = {
                                instance_name: {
                                    'attributes': class_members[class_name]['attributes'],
                                    'methods': class_members[class_name]['methods'],
                                }
                            }
                    else:
                        raise AttributeError(f"Unexpected target type in assignment: {ast.dump(target)}")
                else:
                    raise AttributeError(f"Unexpected instance node type: {type(inst_node)}")
                
            return cls_info, import_nodes, instance_nodes
        
        except Exception:
            logging.exception(f"An exception occurred in create_cls_info")
            return None,None,None

    def add_instance(self,idx:int, instance_node:ast.Assign, cls_info:Dict,functions_def:ast.FunctionDef,method_name:List[str]) -> None:
        """
        Insert an instance and optional method calls into a given function definition.

        This method adds an instance (as an `ast.Assign` node) at a specified index inside a
        function or method body. It also allows appending method calls to the instance after
        initialization, based on the provided class information.

        Parameters
        ----------
        idx : int
            The index at which the instance node should be inserted in the function body.

        instance_nodes : ast.Assign
            The AST assignment node representing the instance to be inserted.

        cls_info : dict
            Dictionary containing class-related metadata, used to determine if methods should
            be called after initialization.

        functions_def : ast.FunctionDef
            The function or method definition (`ast.FunctionDef`) where the instance is to be added.

        method_name : list of str
            List of method names that should be called on the instance after initialization.

        Returns
        -------
        None
            This method modifies the `functions_def` node in place.
        """
        try:
            if functions_def:
                # Determine the correct insert position
                insert_idx = idx
            
                if insert_idx is None:
                    # Look for the last assignment in the function body
                    last_assign_idx = -1
                    for i, stmt in enumerate(functions_def.body):
                        if isinstance(stmt, ast.Assign):
                                last_assign_idx = i
                    if last_assign_idx != -1:
                        insert_idx = last_assign_idx + 1
                    else:
                        insert_idx = len(functions_def.body)
                else:
                    # Need to test the case where the idx might be out of range or if the variables that might be dependant on this instances are 
                    # declared beforehand, to do so, we retrieve the first variables which is dependant on the instance itself, most of the time,
                    # these aren't scalar or logical variables but arrays espcially their shape
                    var_pos = [i for i, stmt in enumerate(functions_def.body) if isinstance(stmt,ast.Assign) and isinstance(stmt.value, ast.Call)]
                    var_pos = var_pos[0] if var_pos else None
                    if var_pos and insert_idx < var_pos:
                        logging.info(f"Given idx:{insert_idx} is smaller that the variables that depends on this, placing the instance before it")
                        insert_idx = var_pos
                    
                functions_def.body.insert(insert_idx, instance_node)
                insert_idx += 1  # We add the +1 for the method call such as declaration_initialization to be append just after
            
                # Handle optional method call like :instance.declaration_initialization() or any other methods and also need to check the case whether these require values or such
                if method_name:
                    if isinstance(instance_node.value, ast.Call) and isinstance(instance_node.value.func, ast.Name):
                        class_name = instance_node.value.func.id
                        instance_method = None
                        instance_name = instance_node.targets[0].id if isinstance(instance_node.targets[0], ast.Name) else instance_node.targets[0].attr
                        if isinstance(instance_node.targets[0], ast.Name):
                            instance_method = instance_name
                        elif isinstance(instance_node.targets[0], ast.Attribute):
                            instance_method = instance_node.targets[0] # this is for the cases of self.gm.method_name() or self.method_name()

                        methods = cls_info.get(class_name, {}).get(instance_name, {}).get("methods", None)
                        
                        if methods:
                            for method in method_name:
                                method_ast = methods.get(method) # THis is to check if the method to be added is present inside the class methods, thus avoiding to call ghost methods
                                if method_ast:
                                    expression = self.create_call_statements(method_ast,instance_method)
                                    functions_def.body.insert(insert_idx, expression)
                                    insert_idx += 1
                                else:
                                    raise ValueError(f'Given method name:{method} is not present among the methods of this class:{class_name}')
                        elif method_name and not methods:
                            logging.info(f"This class:{class_name} doesn't have any methods so no possibility of adding given method name")
            else:
                raise NotImplementedError(f'add_instance is not implemented for adding instances outside of functions')
        except Exception:
            logging.exception(f'Exception Error in add_instance')
            raise
        
    def create_call_statements(self, function_ast: ast.FunctionDef, instance:Optional[Union[str,ast.AST]]=None) -> ast.Expr | ast.Assign:
        """
        Generate a function or method call AST node with appropriate arguments and return handling.

        Creates an `ast.Expr` or `ast.Assign` node representing a call to the given function, based on its arguments and return values. If the function has return values, the result
        will be wrapped in an `ast.Assign` node; otherwise, it returns an `ast.Expr` node.

        Parameters
        ----------
        function_ast : ast.FunctionDef
            The AST node representing the function for which the call statement should be generated.

        instance_name : str, optional
            The name of the instance from which the method is called, if applicable
            (i.e., `instance_name.method_name()`). If `None`, assumes a standalone function call.

        Returns
        -------
        ast.Expr or ast.Assign
            An `ast.Expr` node if the function returns nothing, or an `ast.Assign` node if it has one
            or more return values.
        """
        try:
            # Get the function definition
            function_def = next(iter(ast_walk(function_ast, ast.FunctionDef)),None)

            if function_def is None:
                raise ValueError(f'Value error: function_def is None')
            
            function_name = function_def.name
            class_method = any(arg for arg in function_def.args.args if arg.arg == 'self') # This is case we need to handle cases like
            # self.method_name() or gm.method_name instances

            args = [ast.Name(id=arg.arg, ctx=ast.Load()) for arg in function_def.args.args if arg.arg != 'self']

            # Look for a return statement
            return_stmt = next(ast_walk(function_def, ast.Return), None)

            # Create a function call expression(ex: read_dummy())
            if not class_method:
                call_expr = ast.Call(
                    func=ast.Name(id=function_name, ctx=ast.Load()),
                    args=args,
                    keywords=[]
                )
            else:
                if instance is None:
                    value = ast.Name(id="self", ctx=ast.Load())
                elif isinstance(instance, str):
                    value = ast.Name(id=instance, ctx=ast.Load())
                elif isinstance(instance, ast.AST):
                    value = instance
                else:
                    raise TypeError(f"Unexpected type for instance: {type(instance)}")
                
                call_expr = ast.Call(
                    func = ast.Attribute(
                        value = value,
                        attr = function_name,
                        ctx = ast.Load()
                    ),
                    args = args,
                    keywords = []
                )

            # If function returns something(ex: ins = read_dummy(), ins,mcr = read_dummy)
            if return_stmt and return_stmt.value: 
                return_val = return_stmt.value

                if isinstance(return_val, ast.Name): # This means we return only one element
                    target = ast.Name(id=return_val.id, ctx=ast.Store())
                    call_stmt = ast.Assign(
                        targets=[target],
                        value=call_expr
                    )

                elif isinstance(return_val, ast.Tuple): # This means we return multiple elements
                    targets = [
                        ast.Name(id=elt.id, ctx=ast.Store())
                        for elt in return_val.elts if isinstance(elt, ast.Name)
                    ]
                    call_stmt = ast.Assign(
                        targets=[ast.Tuple(elts=targets, ctx=ast.Store())],
                        value=call_expr
                    )
                else:
                    raise AttributeError(f'Unexcpected return_val type:{type(return_val)}')
            else:
                call_stmt = ast.Expr(value=call_expr)

            return call_stmt
        
        except Exception:
            logging.exception(f'Exception in create_call_statements')
            return None  
    
    def get_timer(self) -> ast.FunctionDef|None:
        """
        Retrieve the Python `@timer` decorator function definition.

        Returns the `ast.FunctionDef` node representing the `timer` decorator function, which can be used to measure the execution time of a function. If the decorator cannot be generated
        or an error occurs, returns `None`.

        Returns
        -------
        parsed_ast : ast.FunctionDef or None
            The AST node representing the `@timer` decorator function, or `None` if retrieval fails.

        """
        try:
            templates = self.load_code_templates(self.config_path)
            if templates is None:
                raise ValueError("Templates could not be loaded due to a prior error.")
            
            code_template = templates["Python_templates"]["Python_timer_template"]["template"]
            path = f'{self.benchmark_dir}/{self.subroutine_name}/time.txt'
            code_template = Template(code_template).substitute(
                path= f'"{path}"'
            )
            try:
                parsed_ast =  ast.parse(code_template).body[0]
                return parsed_ast 
            except SyntaxError as e:
                logging.error(f"Syntax error while parsing the timer template: {e}")
                raise
        except Exception:
            logging.exception(f"Exception occurred in get_timer")
            return None
    
    
    def correct_function(self, function_def:ast.FunctionDef,cls_info:Dict,timer_tree:ast.AST=None,subroutine_key:str=None,parent_mode:bool=False) -> None:
        """
        Update a translated Python function (from Fortran) by applying argument, decorator, and return modifications.

        This method performs the following operations on a given `ast.FunctionDef` node:

        1. **Argument Replacement for Global Instances**  
        Replaces any global variable names (from a global module) found in the function's arguments
        with an instance of the class that holds those attributes.

        2. **Timer Decorator Insertion**  
        If a timer decorator (`timer_tree`) is provided, it is added to the function's decorators
        to enable execution time measurement.

        3. **Return Statement Insertion for Modified Output Variables**  
        - Inspects the dummy variables from the original Fortran subroutine.
        - Identifies variables marked as `OUT` or `INOUT`.
        - Filters those that are **scalars**, **modified**, and **not part of a global class instance**.
        - Appends a `return` statement to return these variables (as a tuple or single return).

        Parameters
        ----------
        function_def : ast.FunctionDef
            The function definition in Python AST format, translated from Fortran.

        cls_info : dict
            Information about the class whose attributes might be referenced within the function.

        timer_tree : ast.AST
            The AST subtree representing the `@timer` decorator to be optionally inserted.

        parent_mode : bool
            Parent mode allows us to ensure that the arguments(scalars) are not modified to ensure that 
        
        Returns
        -------
        None
            The input `function_def` is modified in place.

        """
        return_list = []
        variables_output = []
        
        module_names = list(cls_info.keys())
        try:
            for module_name in module_names:
                filtered_args = []
                common_args = set()
                instance_name = list(cls_info[module_name].keys())[0]
                global_attr = cls_info[module_name][instance_name]["attributes"]

                # THE Dummy arg list contains the supposed args to the function itself
                # Step 1: Get dummy args and actual argument positions
                dummy_arg_list = self.extractor.dummy_arg_list[subroutine_key]
                arg_poses = []
                for args_list in self.extractor.actual_arg_spec_list[subroutine_key]:
                    for i, arg in enumerate(args_list):
                        if arg not in dummy_arg_list:
                            arg_poses.append(i)
                # Step 2: Prepare for filtering arguments from function_def
                args_list = [arg.arg for arg in function_def.args.args]
                common_global_args = set(args_list) & set(global_attr) # we first retrieve the common global args ffrom the class attributes itself
                # Then check for the atttributes coming form the object classes 
                common_other_object_args = set()
                other_object_instances = cls_info[module_name][instance_name].get('instances', {}) # We check if the the actual class has any other attributes
                if other_object_instances:
                    for _, instance in other_object_instances.items():
                        other_attrs = set(instance.get('attributes', []))
                        matching_args = set(args_list) & other_attrs
                        if matching_args:
                            common_other_object_args |= matching_args
                common_args = common_global_args | common_other_object_args
                
                if not arg_poses and common_args:
                    function_def.args.args = [arg for arg in function_def.args.args if arg.arg not in common_args and arg.arg != 'self']
                    
                elif arg_poses and common_args:
                    for i, arg in enumerate(function_def.args.args):
                        arg_name = arg.arg
                        if i in arg_poses:
                            # Argument is used at this position
                            filtered_args.append(arg)
                        elif arg_name not in common_args and arg_name != 'self':
                            filtered_args.append(arg)
                    function_def.args.args = filtered_args
                if instance_name == 'self': # Self should always be placed first 
                    function_def.args.args.insert(0,ast.arg(arg=instance_name))
                else:
                    function_def.args.args.append(ast.arg(arg=instance_name))
                
                call_indices = defaultdict(int)
                # walk through and modify function call nodes if present
                for node in ast_walk(function_def, ast.Expr):
                    if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)):
                        func_name = node.value.func.id
                        if func_name not in self.extractor.call_within_sub[self.subroutine_name]:
                            continue

                        method = cls_info[module_name][instance_name]['methods'].get(func_name)
                        if not method:
                            continue

                        i = call_indices[func_name] # since by default all the elements are at 0 
                        call_indices[func_name] += 1  # increment for next time when we encounter the function itself

                        indexes = []
                        for arg in method.args.args:
                            if arg.arg in self.extractor.dummy_arg_list[func_name]:
                                index = self.extractor.dummy_arg_list[func_name].index(arg.arg)
                                indexes.append(index)

                        try:
                            args = [
                                ast.Name(id=self.extractor.actual_arg_spec_list[func_name][i][idx], ctx=ast.Load())
                                for idx in indexes
                            ]
                            node.value.args = args
                            # print(ast.unparse(ast.fix_missing_locations(node)))

                        except (IndexError, KeyError) as e:
                            logging.error(f"Error mapping arguments for call to '{func_name}' at index {i}: {e}")

                scalar_variables = set()
                if parent_mode:
                    func_name = function_def.name 
                    method = cls_info[module_name][instance_name]['methods'].get(func_name)
                    if not method:
                        continue

                    for arg in method.args.args:
                        if arg.arg in self.extractor.scalar_variables[func_name]:
                            scalar_variables.add(arg.arg)

                # This is to add the timer decorator in the case we need to measure the execution time of the function
                function_def.decorator_list = [ast.Name(id=next(ast_walk(timer_tree,ast.FunctionDef)).name,ctx=ast.Load())] if timer_tree else []
            
                # Return list: we need to verify if the function def requires a return stmt since the function defintion has been converted from 
                # fortran directly the return could not have been added as such we need to add it inside. To do so we directly use the var_dummy
                # to scout out the output variables.            
                for variables in self.extractor.var_dummy[self.subroutine_name]: # We use var_dummy which contains the arguments sent to the subroutine 
                    if any([var for var in walk(variables,F23.Intent_Spec) if var.tostr() in ["OUT","INOUT"]]):
                        name = walk(walk(variables,F23.Entity_Decl),F23.Name)[0].string
                        variables_output.append(name)
            
                if variables_output: # If THERE ARE variable output send as arguments
                    var_modified = set(variables_output) & set(self.extractor.var_modif[self.subroutine_name]) # Retreive the output variables that might be modified
                    if var_modified: # This is to find if the variables modified need to have a return statement
                        # Filter out global variables
                        non_global_vars = var_modified - set(global_attr)
                            
                        # Filter out arrays (those with 'DIMENSION' in their type info) which allows us to return only the scalars
                        # And arrays that are present locally within the main file and not the global attribute file will not be returned 
                        # but will still be updated due to the fact that these variables are sent as arguments and the instance of the global
                        # module being sent as arguments
                        scalars_to_return = {
                            var for var in non_global_vars
                            if 'DIMENSION' not in self.extractor.var_modif_info[self.subroutine_name].get(var, [])
                        }
                        
                        if scalars_to_return:
                            return_list.extend(scalars_to_return)
                            
            # Now we need to take care of the subscript element so that we don't have a index error ex:mc[:,nlsm,ins] here ins is 3 and nslm 11
            # Thus ins and nslm makes it go out of bounds since python use zero based indexation,
            # First retrieve all the loop variables within the for loop usually present in cls.loop_dict
            cons_var = set()
            for values in self.extractor.loop_dict.values():
                value = sorted(values)
                if len(value)>1:
                    cons_var.add(value[1])
                else:
                    cons_var.add(value[0])

            # Now we visit each node and adjust the subscripts 
            sub_name = subroutine_key if subroutine_key else self.subroutine_name
            # We need to send the all_array_info for two reasons: one being the fact that it also contains the local prsent arrays and the dimesnion info
            # and the second being that these arrays are sent as another when we call the anotehr function locally inside another function thus ru_infilt becomes ru_infilt_ns when 
            # we call hydrol_soil_infilt inside the hydrol_soil
            kwargs = {"exclude_index": scalar_variables} if scalar_variables else {} # This is to exclude the variables that are sent as arguments(scalars) and don't need to be modified
            # But when we isolate only the children only then we need to ensure that we the arguments which might be used inside the arguments 
            adjust_indices = AdjustIndices(cons_var,self.extractor.all_array_info[sub_name],cls_info[module_names[-1]][instance_name],**kwargs)
            for element in function_def.body:
                adjust_indices.visit(element)
            
            # print(ast.unparse(ast.fix_missing_locations(function_def)))

            if return_list:
                ret_stmts = []
                    
                for ret_stmt in return_list:
                    ret_stmts.append(ast.Name(id = ret_stmt,ctx = ast.Load()))
                # print(ast.dump(ret_stmts[0],indent=4))
                return_node = ast.Return()
                    
                if len(return_list) > 1:
                    return_node.value = ast.Tuple(
                                            elts = ret_stmts,
                                            ctx = ast.Load()
                                        )
                    function_def.body.append(return_node)
                else:
                    return_node.value = ret_stmts[0]
                    function_def.body.append(return_node)
        except Exception:
            logging.exception(f'Exception in correct_function')
            raise

    def out_module_python(self) -> ast.Module:
        """
        In charge of retrieving the python global module template on either the simple python type or in class format based on the self.cls_mode

        Returns
        -------
        code : ast.Module
            AST of the global code template either in normal python script or class format
        """
        try:
            templates = self.load_code_templates(self.config_path)
            if templates is None:
                raise ValueError("Templates could not be loaded due to a prior error.")
            
            if self.cls_mode:
                code = templates["Python_templates"]["Python_global_class_template"]["template"]
            else:
                code = templates["Python_templates"]["Python_global_normal_template"]["template"] 

            if code is None:
                raise ValueError(f"The code template for out_module_python wasn't retreived")
                
            parsed_ast = self.python_parser(code)
            if parsed_ast is None:
                raise ValueError(f'Parsed AST is None due to prior error')
            
            return parsed_ast
            
        except Exception:
            logging.exception(f"Exception occurred while loading out_module_python")
            return None
    
    def out_main_python(self) -> ast.Module:
        """
        In charge of retrieving the python main module template. 

        Returns
        -------
        code : ast.Module
            AST of the main empty code template 
        """
        try:
            templates = self.load_code_templates(self.config_path)
            if templates is None:
                raise ValueError("Templates could not be loaded due to a prior error.")
            
            code = templates["Python_templates"]["Python_main_template"]["template"]
            if code is None:
                raise ValueError(f"The code template for out_main_python wasn't retreived")

           
            parsed_ast =  self.python_parser(code)
            if parsed_ast is None:
                raise ValueError(f'Parsed AST is None due to prior error')
            
            return parsed_ast
            
        except Exception:
            logging.exception(f"Exception occurred while loading out_main_python")
            return None

    def retreive_variable_order(self) -> None:
        """
        Retrieve the access order of variables from binary files.

        Determines the sequence in which variables are accessed or stored in the binary
        file format. This order is important as it reflects how data is laid out in the binary.

        The resulting order is stored in the `variable_order` attribute of the instance.

        Returns
        -------
        None
            This method updates the `variable_order` attribute in place.
        """
        self.variable_order = []
        try:
            for read_dec in [self.isolator.processor.reads_in_decleration_routine,self.isolator.processor.reads_in_read_routine]:
                read_stmt = walk(read_dec,F23.Input_Item_List)
                for item in read_stmt:
                    self.variable_order.append(item.children[0].string)
        except Exception:
            logging.exception(f'Exception in retrieve_variable_order')
            raise

    def convert_SPECIFICATION_PART(self,declaration_stmts:List,fix_loc:bool=False, cls_mode:bool=False) -> List:
        """
        Convert FORTRAN specification statements into Python assignment statements.

        This method transforms FORTRAN declaration statements (e.g., variable types and initializations)
        into corresponding Python `ast.Assign` nodes. It also handles `use` statements by generating
        appropriate `Procedure` or import-related nodes.

        Parameters
        ----------
        declaration_stmts : list
            A list of FORTRAN AST nodes representing declaration statements to be converted.

        fix_loc : bool, optional
            If True, adjusts location info to enable the use of `ast.unparse` on the resulting AST nodes.
            Useful when calling this method independently.

        cls_mode : bool, optional
            If True, generates assignment statements suitable for inclusion within a class context.

        Returns
        -------
        ast_nodes : list of ast.AST
            A list of Python AST nodes including assignment statements and, if applicable, procedure/import nodes.
        
        """
        ast_nodes = []

        kind_map = {
            'REAL': 'np.float64',  
            'INTEGER': 'np.int32',
            'LOGICAL': 'np.bool'
        }

        target = None
        try:
            for declarations in declaration_stmts:
                # First is that we verify if we have a type declaration stmt and it's allocation stmt -> requires combine_allocate_declarations
                # in order to retrieve a combined and a proper allocation table variable

                # ANOTHER possibility with the first condition is that if we have intent within the declarations and also a length of 2 the
                # combine_allocate_declaration method will just remove it since and return a new formatted variable( look at the return value
                # of the method)
                if len(declarations) == 2:
                    declarations = Processor().combine_allocate_declaration(declarations)
                    # print(declarations)
                else: 
                    # Need to verify if the one of the declarations has an INTENT/SAVE/PUBLIC among them thus we need to remove it before
                    # transformation.
                    declarations = Processor().remove_intent_and_save(declarations)
                    # print(declarations[0])
                    
                # cls.dec_global finds not only the retreives the variables but also procedures, found case: hydrol_split_soil
                # Which means they might have use statements as well.
                for nodes in walk(declarations, (F23.Type_Declaration_Stmt,F23.Use_Stmt)):
                    if isinstance(nodes,F23.Use_Stmt): # THESE are for the procedure call
                        # Transform onto the python ast and add it to ast_nodes
                        # The first three elements of the use stmt gives out the module name which we nned to import
                        _,_,module_name,_,only_stmt = nodes.children
                        
                        if only_stmt:
                            # We need to see if we have an instance where we have something similar to this : USE my_module, ONLY : func,var1,var2
                            names_ast_list = []
                            for elements in only_stmt.children:
                                if isinstance(elements,F23.Name):
                                    if elements.string in ['ipslerr_p']:
                                        continue
                                    names_ast_list.append(ast.alias(name=elements.string)) 
                                elif isinstance(elements,F23.Rename):
                                    _,name, asname = elements.children
                                    if name.string in ['ipslerr_p'] or asname.string in ['ipslerr_p']:
                                        continue
                                    names_ast_list.append(ast.alias(name=name.string,asname=asname.string))
                            if names_ast_list:
                                import_node = ast.ImportFrom(
                                    module=module_name.string,
                                    names = names_ast_list,
                                    level=0
                                )
                                ast_nodes.append(import_node)
                        else:
                            import_node = ast.ImportFrom(
                                module=module_name.string,
                                names = ast.alias(name='*'),
                                level=0
                            )
                            ast_nodes.append(import_node)

                    elif isinstance(nodes,F23.Type_Declaration_Stmt): # These are for the variables 

                        value = None
                        np_dtype = None
                        idx,attr = None, None
                        
                        attr_spec = [param.string for param in walk(nodes,F23.Attr_Spec)] # This looks for the Attr specification such as ALLOCATBLE or PARAMETER or SAVE 
                        allocation_spec = [param.children[0] for param in walk(nodes,F23.Dimension_Attr_Spec) if param.children[0] == 'DIMENSION']
                        # print(allocation_spec)
                        kind_selec = any([param.string for param in walk(nodes, F23.Kind_Selector)])

                        # print(kind_selec)
                        intrinsic_type_spec,_,entity_decl_list = nodes.children # This gives out a tuple
                        entity_decls = entity_decl_list.children

                        # Allows us to create the numpy type based on the intrinisic type specification part 
                        np_dtype = kind_map.get(intrinsic_type_spec.children[0], 'np.float64')
                        idx,attr = np_dtype.split('.')

                        for entity_decl in entity_decls:
                            var_name, _,_, initialization = entity_decl.children

                            if cls_mode:
                                target = ast.Attribute(
                                        value = ast.Name(id="self",ctx=ast.Load()),
                                        attr = var_name.string, ctx = ast.Store()
                                )
                            else:
                                target = ast.Name(id=var_name.string, ctx=ast.Store())
                            
                            if initialization is not None:
                                _,value = initialization.children
                            
                            if 'PARAMETER' in attr_spec:
                                if intrinsic_type_spec.children[0] in ['INTEGER', 'REAL'] and not kind_selec: 
                                    # These are only for element thats has parameters and 
                                    # has no kind(KIND argument) inside
                                    # Example case : INTEGER, PARAMETER :: a = 6
                                    # Perhaps removable since even with/without PARAMETER argument in attribute specification this will changed similarly to INTEGER :: a = 6
                                    if value is not None:
                                        if not isinstance(value,F23.Name):
                                            num_var = float(value.string) if intrinsic_type_spec[0].children[0] == "REAL" else int(value.string)
                                            assign = ast.Assign(
                                                        targets=[target],
                                                        value=ast.Call( func=ast.Attribute(value=ast.Name(id=idx, ctx=ast.Load()), attr=attr, ctx=ast.Load()),
                                                                    args=[ast.Constant(value=num_var)],
                                                                    keywords = []
                                                                )
                                                    )
                                        else:
                                            if cls_mode:
                                                name_val = ast.Attribute(
                                                    value=ast.Name(id="self",ctx=ast.Load()),
                                                    attr=value.string,
                                                    ctx= ast.Load()
                                                )
                                            else:
                                                name_val = ast.Name(id=value.string,ctx=ast.Load())
                                            assign = ast.Assign(
                                                        targets=[target],
                                                        value= name_val
                                                    )
                                    else:
                                        
                                        assign = ast.Assign(
                                                    targets=[target],
                                                    value=ast.Call( func=ast.Attribute(value=ast.Name(id=idx, ctx=ast.Load()), attr=attr, ctx=ast.Load()),
                                                                args=[ast.Constant(value=0)],
                                                                keywords = []
                                                            )
                                                )
                                    ast_nodes.append(assign)
                                    # print(f'Dtype:{intrinsic_type_spec}, value:{entity_decl_list}')
                                elif intrinsic_type_spec.children[0] in ['INTEGER', 'REAL'] and kind_selec: # IF the keyword KIND is present  
                                    # np_dtype = kind_map.get(intrinsic_type_spec.children[0], 'np.float64') 
                                    # Create: var = np.array(value, dtype=np_dtype)
                                    # idx,attr = np_dtype.split('.')
                                    value_ = None
                                    val = None
                                    if value is not None:

                                        if 'DIMENSION' in allocation_spec:
                                            # Parse array constructor
                                            elements = []
                                            for val in walk(value,F23.Real_Literal_Constant):  
                                                num_str = val.string.split('_')[0] 
                                                num_val = float(num_str) if intrinsic_type_spec.children[0] == "REAL" else int(num_str)
                                                elements.append(ast.Constant(value=num_val))
                                            
                                            val = ast.Call(
                                                func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='array', ctx=ast.Load()),
                                                args=[ast.List(elts=elements, ctx=ast.Load())],
                                                keywords=[ast.keyword(arg='dtype',
                                                        value=ast.Attribute(value=ast.Name(id=idx, ctx=ast.Load()), attr=attr, ctx=ast.Load()))]
                                            )
                                            assign = ast.Assign(
                                                targets=[target],
                                                value=val
                                            )
                                        else:
                                            if len(value.children) > 2:
                                                num1,_, num2 = value.children
                                                value_ = ast.BinOp(
                                                    left=ast.Constant(value=float(num1.string)),
                                                    op=ast.Mult(),
                                                    right=ast.Constant(value=float(num2.string))
                                                )
                                            else:
                                                if not isinstance(value,F23.Name):
                                                    if attr == "int32":
                                                        value_ = ast.Constant(value=int(value.children[0]))
                                                    else:
                                                        value_ = ast.Constant(value=float(value.children[0]))
                                                else:
                                                    if cls_mode:
                                                        val = ast.Attribute(
                                                            value=ast.Name(id="self",ctx=ast.Load()),
                                                            attr=value.string,
                                                            ctx= ast.Load()
                                                        )
                                                    else:
                                                        val = ast.Name(id=value.string,ctx=ast.Load())

                                            val = ast.Call(
                                                func=ast.Attribute(value=ast.Name(id=idx, ctx=ast.Load()), attr=attr, ctx=ast.Load()),
                                                args=[value_],
                                                keywords = []
                                                # keywords=[
                                                #     ast.keyword(arg='dtype', value=ast.Attribute(value=ast.Name(id=idx, ctx=ast.Load()), attr=attr, ctx=ast.Load()))
                                                # ]
                                            )
                                    else:
                                        val = ast.Call(func=ast.Attribute(value=ast.Name(id=idx, ctx=ast.Load()), attr=attr, ctx=ast.Load()),
                                                        args=[ast.Constant(value=0)],
                                                        keywords = []
                                                    )
                                    
                                    assign = ast.Assign(
                                        targets=[target],
                                        value=val
                                    )
                                    
                                ast_nodes.append(assign)

                            elif 'DIMENSION' in allocation_spec:
                                dimensions_spec_list = walk(walk(nodes,F23.Dimension_Attr_Spec),F23.Explicit_Shape_Spec_List)
                                shape = []
                                left,right = None,None
                                constant_right = None
                                arg_shape = None
                                
                                for child in dimensions_spec_list[0].children:
                                    limits = child.tostr().split(' : ')
                                    lb = limits[0].strip(" ")
                                    
                                    if len(limits) > 1:
                                        lb_child,ub_child = child.children # THis will allow to separate the upper bound and lower who appears as integers 
                                        ub = limits[1].strip(" ")
                                        if lb:
                                            constant_right = ast.Constant(1)
                                        
                                        if cls_mode:
                                            if ub_child is not None and ub_child.string.isdigit():
                                                left = ast.Constant(int(ub))
                                            else:
                                                left = ast.Attribute( # Upper bound 
                                                    value = ast.Name(id = 'self', ctx = ast.Load()),
                                                    attr = ub,
                                                    ctx = ast.Load())
                                            
                                            if lb_child is not None and lb_child.string.isdigit():
                                                
                                                right = ast.Constant(int(lb))
                                            else:
                                                right = ast.Attribute( # Lower bound 
                                                    value = ast.Name(id = 'self', ctx = ast.Load()),
                                                    attr = lb,
                                                    ctx = ast.Load())
                                            
                                        else:
                                            if ub_child is not None and ub_child.string.isdigit():
                                                left = ast.Constant(int(ub))
                                            else:
                                                left = ast.Name(id = ub, ctx = ast.Load())
                                            if lb_child is not None and lb_child.string.isdigit():
                                                right = ast.Constant(int(lb))
                                            else:
                                                right = ast.Name(id = lb, ctx = ast.Load())
                                    
                                        arg_shape = ast.BinOp(
                                            left = ast.BinOp(
                                                left = left,
                                                op = ast.Sub(),
                                                right = right),
                                            op = ast.Add(),
                                            right = constant_right)
                                        # print(ast.dump(arg_shape,indent=4))
                                        shape.append(arg_shape)
                                    else:
                                        if cls_mode:
                                            if lb.isdigit():
                                                arg_shape = ast.Constant(int(lb))
                                            else:
                                                arg_shape = ast.Attribute(
                                                    value = ast.Name(id = 'self',ctx = ast.Load()),
                                                    attr = f"{lb}",
                                                    ctx = ast.Load()
                                                )
                                        else:
                                            if lb.isdigit():
                                                arg_shape = ast.Constant(int(lb))
                                            else:
                                                arg_shape = ast.Name(f"{lb}")
                                        shape.append(arg_shape)

                                # Why are we doing np.empty is due to the fact that empty doesn't intialize the arrays but keeps in memory (randomly created values) which might be really small 1e-300
                                # to really large but also these values are not random but values what was present in the memory: https://www.reddit.com/r/learnpython/comments/wgexrf/comment/iizmx5d/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button
                                # Thus using np.zeros better use case than the np.empty since this also helpful in the case we have logical arrays which means it has either zeros or one we use zeros dimensions
                                np_call = ast.Call(  
                                    func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='zeros', ctx=ast.Load()),
                                    args=[ast.Tuple(elts=shape, ctx=ast.Load())],
                                    keywords=[ ast.keyword(arg='dtype', value = ast.Attribute(value=ast.Name(id=idx, ctx=ast.Load()), attr=attr, ctx=ast.Load()))]
                                )
                                    
                                assign = ast.Assign(
                                    targets=[target],
                                    value=np_call
                                )
                                ast_nodes.append(assign)

                            else: # Cases where the variable is not a PARAMETER nor ALLOCATABLE present, either just a INTEGER, LOGICAL
                                
                                if intrinsic_type_spec.children[0] == 'LOGICAL' and not kind_selec: # SET AS AN EXCEPTIONAL CASE
                                    bool_val = False 
                                    if value is not None:
                                        if value.string.split(".")[1] == "TRUE":
                                            bool_val = True
                                        
                                        bool_call = ast.Call(
                                                        func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='bool', ctx=ast.Load()),
                                                        args=[ast.Constant(value=bool_val)],
                                                        keywords=[]
                                                    )
                                    else:
                                        bool_call = ast.Call(
                                                        func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='bool', ctx=ast.Load()),
                                                        args=[ast.Constant(value=bool_val)],
                                                        keywords=[]
                                                    )
                                    assign = ast.Assign(
                                                    targets=[target],
                                                    value=bool_call
                                                )
                                    ast_nodes.append(assign)
                                
                                # Need to handle case where there could be a value if the kind_selec is None but still has a value
                                elif not kind_selec:
                                    
                                    if value is None: # Example cases : INTEGER :: ier 
                                        assign = ast.Assign(
                                                targets=[target],
                                                value=ast.Call( func=ast.Attribute(value=ast.Name(id=idx, ctx=ast.Load()), attr=attr, ctx=ast.Load()),
                                                                args=[ast.Constant(value=0)],
                                                                keywords = []
                                                            )
                                                )
                                    
                                    else:
                                        if not isinstance(value, F23.Name):
                                            num_val = int(value.string) if intrinsic_type_spec.children[0] == "INTEGER" else float(value.string)
                                            assign = ast.Assign(
                                                targets=[target],
                                                value=ast.Call( func=ast.Attribute(value=ast.Name(id=idx, ctx=ast.Load()), attr=attr, ctx=ast.Load()),
                                                                args=[ast.Constant(value=num_val)],
                                                                keywords = []
                                                            )
                                            )
                                        else:
                                            if cls_mode:
                                                name_val = ast.Attribute(
                                                    value=ast.Name(id="self",ctx=ast.Load()),
                                                    attr=value.string,
                                                    ctx= ast.Load()
                                                )
                                            else:
                                                name_val = ast.Name(id=value.string,ctx=ast.Load())
                                            assign = ast.Assign(
                                                targets=[target],
                                                value=name_val
                                                )
                                        
                                    ast_nodes.append(assign)
                
                                else:
                                    if value is None:
                                        assign = ast.Assign(
                                                targets=[target],
                                                value=ast.Call( func=ast.Attribute(value=ast.Name(id=idx, ctx=ast.Load()), attr=attr, ctx=ast.Load()),
                                                                args=[ast.Constant(value=0)],
                                                                keywords = []
                                                            )
                                                )
                                    else:
                                        # Verify that the value is a string
                                        if not isinstance(value, F23.Name):
                                            num_val = int(value.string) if intrinsic_type_spec.children[0] == "INTEGER" else float(value.string)
                                            assign = ast.Assign(
                                                    targets=[target],
                                                    value=ast.Call( func=ast.Attribute(value=ast.Name(id=idx, ctx=ast.Load()), attr=attr, ctx=ast.Load()),
                                                                    args=[ast.Constant(value=num_val)],
                                                                    keywords = []
                                                                )
                                                    )
                                        else:
                                            if cls_mode:
                                                name_val = ast.Attribute(
                                                    value=ast.Name(id="self",ctx=ast.Load()),
                                                    attr=value.string,
                                                    ctx= ast.Load()
                                                )
                                            else:
                                                name_val = ast.Name(id=value.string,ctx=ast.Load())
                                            assign = ast.Assign(
                                                    targets=[target],
                                                    value=name_val
                                                    )
                                        
                                    ast_nodes.append(assign)
            if fix_loc:    
                ast_nodes = [ast.fix_missing_locations(node) for node in ast_nodes]

            return ast_nodes
        
        except Exception as e:
            logging.error(f'Exception error in convert_Specification_part')
            return None

    def pre_init_variables(self,code_template:ast.Module) -> None:
        """
        Extract variables that are pre-declared and initialized within the given code template.

        This method identifies all variables that have both been declared and initialized in the input code template.
        It stores their names in the `pre_init` attribute as a list.

        Parameters
        ----------
        code_template : ast.Module
            Code template upon which variables or other elements will be added.

        Returns
        -------
        None
            This method modifies the `pre_init` attribute directly.
        """
        
        self.pre_init = []
        class_exist = any(ast_walk(code_template,ast.ClassDef))
        try:
            if class_exist:
                functions_spec = ast_walk(code_template,ast.FunctionDef)
                for functions in functions_spec:
                    if functions.name == "__init__":
                        assign_stmt = ast_walk(functions,ast.Assign)
                        for assign_ in assign_stmt:
                            if isinstance(assign_.targets[0],ast.Name):
                                self.pre_init.append(assign_.targets[0].id)
                            elif isinstance(assign_.targets[0],ast.Attribute):
                                self.pre_init.append(assign_.targets[0].attr)
                            
            else:
                nodes = ast_walk(code_template,ast.Assign)
                for node in nodes:
                    self.pre_init.append(node.targets[0].id)
        except Exception:
            logging.exception(f'Exception in pre_init_variables')
            raise

    def search_dependant_variables(self,declaration_stmts:List) -> None:
        """
        Analyze dependencies between arrays and scalars using the FORTRAN AST.

        This method identifies which variables (dependents) rely on others (dependees) by traversing the FORTRAN AST.
        It creates an attribute `dependant_variables`, a dictionary where each key is a dependent variable and the corresponding value is a list of its dependees.

        Parameters
        ----------
        declaration_stmts : List
            List of declaration statements within which to find dependents and dependees.

        Returns
        -------
        None
            This method modifies the `dependant_variables` attribute directly.
        """
        
        self.dependant_variables = {} # THe variable that is the dependant of the other which has the key as the dependant and the values 
        # the different dependees 
        combined_stmt = None
        try:
            for declarations in declaration_stmts:
                dependees = []
                alloc_spec = any(alloc for alloc in walk(declarations, F23.Attr_Spec) if alloc.string == "ALLOCATABLE")
                if len(declarations) == 2 and alloc_spec:
                    combined_stmt = Processor().combine_allocate_declaration(declarations)
                else:
                    combined_stmt = declarations
                
                if combined_stmt and alloc_spec:
                    dimensions_spec_list = walk(walk(combined_stmt,F23.Dimension_Attr_Spec),F23.Explicit_Shape_Spec_List) 
                    entity_dec_name = walk(combined_stmt,F23.Entity_Decl)[0].children[0]
                    
                    # Now we verify if one of these variables has initialization as None
                    for arg in dimensions_spec_list[0].children: # This allows to handle cases such as imax:imin type 
                        
                        limits = arg.tostr().split(' : ')
                        lb = limits[0] 
                        
                        ub = limits[1] if len(limits) > 1 else None
                        # print(lb,ub)
                        # we now verify that the upper bound/lower bound 's shape is present within the pre init variables and the declarations
                        if lb is not None:
                            dec = [elements for elements in declaration_stmts 
                                if walk(walk(elements,F23.Entity_Decl),F23.Name) and walk(walk(elements,F23.Entity_Decl),F23.Name)[0].string == lb ]
                            if not dec and lb in self.pre_init: # This means that the variables is present in the pre init variables
                                continue 
                            elif dec and lb not in self.pre_init:
                                # we verify the initalization of these shapes
                                for elements in dec:
                                    entity_decl_list = walk(elements,F23.Entity_Decl_List)[0]
                                    for entity_dec in entity_decl_list.children:
                                        _, _,_, initialization = entity_dec.children
                                        if initialization is None:
                                            dependees.append(lb)

                        if ub is not None:
                            dec = [elements for elements in declaration_stmts 
                                if walk(walk(elements,F23.Entity_Decl),F23.Name) and walk(walk(elements,F23.Entity_Decl),F23.Name)[0].string == ub ]
                            if not dec and ub in self.pre_init: # This means that the variables is present in the pre init variables
                                continue 
                            elif dec and ub not in self.pre_init:
                                # we verify the initalization of these shapes
                                for elements in dec:
                                    entity_decl_list = walk(elements,F23.Entity_Decl_List)[0]
                                    for entity_dec in entity_decl_list.children:
                                        _, _,_, initialization = entity_dec.children
                                        if initialization is None:
                                            dependees.append(ub)
                                
                    if len(dependees) > 0:
                        self.dependant_variables[entity_dec_name.string] = dependees
        except Exception:
            logging.exception(f'Exception in search dependant variable')
            raise
    
    def _find_init_pattern(self, function: ast.FunctionDef, class_name: str, method_name: str) -> int | None:
        """
        Identifies the position after a method call on a class instance within a function.

        This helper function is responsible for identifying the class instance and locating 
        the method call on that instance. It is used to retrieve variables or elements 
        necessary for subsequent operations. By identifying the location of such method 
        calls, we can determine where to insert or place elements in the function body.

        Parameters
        ----------
        function : ast.FunctionDef
            The AST (Abstract Syntax Tree) node representing the function in which 
            the class instance and method call should be identified.
        class_name : str
            The name of the class whose instance we are trying to identify.
        method_name : str
            The name of the method called on the class instance.

        Returns
        -------
        int or None
            The index position after the method call where elements can be inserted. 
            Returns `None` if no suitable position is found.
        """
        for i in range(len(function.body) - 1):
            stmt1 = function.body[i]
            stmt2 = function.body[i + 1]

            # Look for assignment like: var = ClassName() or self.var = ClassName()
            if (
                isinstance(stmt1, ast.Assign)
                and isinstance(stmt1.value, ast.Call)
                and isinstance(stmt1.value.func, ast.Name)
                and stmt1.value.func.id == class_name
                and len(stmt1.targets) == 1
                and isinstance(stmt1.targets[0], (ast.Name, ast.Attribute))
            ):  
                # THis is to handle both self and non self aspects
                target = stmt1.targets[0]
                if isinstance(target, ast.Name):
                    var_name = target.id
                elif isinstance(target, ast.Attribute):
                    if isinstance(target.value, ast.Name) and target.value.id == "self":
                        var_name = target.attr
                    else:
                        continue
                
                # Look for method call: var.method_name() or self.var.method_name inside class 
                if (isinstance(stmt2, ast.Expr) and isinstance(stmt2.value, ast.Call) and isinstance(stmt2.value.func, ast.Attribute)):
                    if isinstance(stmt2.value.func.value, ast.Name): # Here we have an instance of a.method() 
                        if stmt2.value.func.value.id == var_name and stmt2.value.func.attr == method_name:
                            
                            # Check if any of the following statements is an assignment
                            for stmt in function.body[i + 2:]:
                                if isinstance(stmt, ast.Assign):
                                    return None  # Found an assignment later, reject it since the follwoing elements could be variables that might depend on this until 
                                    # we stumble upon another class instance and method call 
                            return i + 2  # Plus two since we are removing -1 from range 
                    elif isinstance(stmt2.value.func.value, ast.Attribute): # Here we have an instance of self.a.method()
                        
                        if stmt2.value.func.value.attr == var_name and stmt2.value.func.attr == method_name:
                            for stmt in function.body[i + 2:]:
                                if isinstance(stmt, ast.Assign):
                                    return None 
                            return i + 2 
                            
        return None

    def _check_position_within_function(self,function: ast.FunctionDef,class_name:str, method_name:str):
        """
        Determines the appropriate insertion position within a function body.

        This method analyzes the function body to find the appropriate position to insert new code. 
        It first attempts to find a pattern where a class instance is created and a specific method is called. 
        If not found, it falls back to placing the insertion after the last assignment. It also ensures that 
        the insertion does not occur after a return statement.

        Parameters
        ----------
        function : ast.FunctionDef
            The AST node representing the function to analyze.
        class_name : str
            The name of the class whose instance should be detected in the function.
        method_name : str
            The name of the method to detect on the class instance.

        Returns
        -------
        tuple of (int, int or None)
            A tuple containing:
            - `insert_pos` (int): The position in the function body where code should be inserted.
            - `return_stmt_pos` (int or None): The position of the last return statement, if any.
        """

        # This is to handle cases when we a class instance followed by a method call
        insert_pos = self._find_init_pattern(function, class_name=class_name, method_name=method_name)
        # If not found, default to after last assignment
        if insert_pos is None:
            assign_positions = [pos for pos, stmt in enumerate(function.body) if isinstance(stmt, ast.Assign)]
            insert_pos = assign_positions[-1] + 1 if assign_positions else len(function.body)

        # Check return statement position
        return_positions = [i for i, stmt in enumerate(function.body) if isinstance(stmt, ast.Return)]
        return_stmt_pos = return_positions[-1] if return_positions else None

        if return_stmt_pos is not None and insert_pos > return_stmt_pos:
            insert_pos = return_stmt_pos  # Insert just before return

        return insert_pos, return_stmt_pos

    def insert_at(self, idx:int, ast_node:ast.AST, python_template:ast.Module,method_name:str = None,**kwargs) -> None:
        """
        Inserts an AST node into a Python AST module at a specified location based on context.

        This method performs context-aware insertion of AST nodes into the provided Python AST
        (`ast.Module`). Depending on the type of node and the provided context, it determines 
        the correct insertion point.

        Supported behavior:
        - `Import` and `ImportFrom` nodes are inserted at the top of the module.
        - `FunctionDef` nodes are inserted at the specified index, or after the last import/function.
        - `Assign` nodes are inserted inside the specified method (`method_name`), defaulting to `__init__`
        if none is provided. If no method is found, it falls back to the module level.
        - (Planned/placeholder) `For` loops may be placed based on method name and reference node.
        - If `idx` is not provided, insertion happens after the last relevant node of the same type.

        Parameters
        ----------
        idx : int
            The index at which to insert the AST node within the relevant scope.
        ast_node : ast.AST
            The AST node to be inserted. Supported types include `Import`, `ImportFrom`, 
            `FunctionDef`, `Assign`, and `For`.
        python_template : ast.Module
            The Python AST (module or subtree) where the node will be inserted.
        method_name : str, optional
            The name of the method or function where the node should be inserted. 
            Required for contextual placement of `Assign` or `For` nodes.

        Other Parameters
        ----------------
        **kwargs
            Additional keyword arguments for context-aware insertion.

        Returns
        -------
        None
            This method modifies the `python_template` in place and returns nothing.
        """

        try:
            class_exists = any(isinstance(n, ast.ClassDef) for n in python_template.body) # any(ast_walk(python_template,ast.ClassDef))
            if class_exists:
                if isinstance(ast_node, (ast.Import, ast.ImportFrom)):
                    
                    import_stmts = [pos for pos, node in enumerate(python_template.body) if isinstance(node, (ast.Import, ast.ImportFrom))]
                    if import_stmts:
                        python_template.body.insert(import_stmts[-1] + 1,ast_node)
                    else:
                        python_template.body.insert(0,ast_node)
                
                # If the given ast_node is that of a function defintion
                elif isinstance(ast_node,ast.FunctionDef):
                    function_pos = [pos for pos, node in enumerate(python_template.body[0].body) if isinstance(node,ast.FunctionDef)]
                    # function_pos[-1] + 1 since if we have __init__ method, we place rest of the function after the init method or any other method
                    insert_pos = function_pos[-1] + 1 if function_pos else len(python_template.body[0].body)
                    if idx and idx > insert_pos:
                        python_template.body[0].body.insert(idx,ast_node)
                    else:
                        python_template.body[0].body.insert(insert_pos, ast_node)
                        
                elif isinstance(ast_node,ast.Assign): # If the given ast_node is that of a variable assignement statement
                    functions_spec = ast_walk(python_template,ast.FunctionDef)
                    if functions_spec:
                        for functions in functions_spec:
                            if method_name: 
                                logging.info(f"Since argument method_name is: {method_name}, placing the assign statement inside of this method")
                                if functions.name == method_name:
                                    class_name = kwargs.get('class_name')
                                    method = kwargs.get('method')
                                    insert_pos,return_stmt_pos = self._check_position_within_function(functions,class_name=class_name,method_name=method)
                                    
                                    if idx and (return_stmt_pos is None or idx < return_stmt_pos) and idx > insert_pos:
                                        functions.body.insert(idx,ast_node)
                                    else:
                                        logging.info(f'Since no index is given, WILL BE USING previous known ast Assign position with this method: {method_name}')
                                        functions.body.insert(insert_pos, ast_node)
                            else: # By default we will place it inside the __init__ method 
                                if functions.name == "__init__":
                                    class_name = kwargs.get('class_name')
                                    method = kwargs.get('method')
                                    
                                    assign_statement = [pos for pos, assign in enumerate(functions.body) if isinstance(assign, ast.Assign)]
                                    insert_pos = assign_statement[-1] + 1 if assign_statement else len(functions.body)
                                    if idx:
                                        functions.body.insert(idx,ast_node)
                                    else:
                                        logging.info(f'Since no index is given, WILL BE USING previous known ast Assign position')
                                        functions.body.insert(insert_pos , ast_node)
                    else:
                        logging.info(f'Since no method name is given, the ASSIGN ast will be placed inside the parent body')
                        python_template.body.append(ast_node)
                                
                elif isinstance(ast_node,ast.Expr):
                    if method_name:
                        if isinstance(python_template, ast.FunctionDef):
                            target_func = python_template
                        else:
                            target_func = [func for func in ast_walk(python_template, ast.FunctionDef)if func.name == method_name][0] 

                        if target_func:
                            class_name = kwargs.get('class_name')
                            method = kwargs.get('method')
                            insert_pos,return_stmt_pos = self._check_position_within_function(target_func,class_name=class_name,method_name=method)
                            if idx and (return_stmt_pos is None or idx < return_stmt_pos) and idx > insert_pos:
                                target_func.body.insert(idx,ast_node)
                            else:
                                target_func.body.insert(insert_pos, ast_node)
                    else:
                        python_template.body.append(ast_node) 
                            
            else:
                if isinstance(ast_node, (ast.Import, ast.ImportFrom)):
                    import_stmts = [pos for pos, node in enumerate(python_template.body) if isinstance(node, (ast.Import, ast.ImportFrom))]
                    if import_stmts:
                        python_template.body.insert(import_stmts[-1] + 1,ast_node)
                    else:
                        python_template.body.insert(0,ast_node)
                
                elif isinstance(ast_node, ast.ClassDef):
                    import_positions = [ pos for pos, stmt in enumerate(ast.iter_child_nodes(python_template))
                            if isinstance(stmt, (ast.Import, ast.ImportFrom))]
                    if idx:
                        if import_positions:
                            last_import_pos = import_positions[-1] + 1
                            
                            if idx <= last_import_pos:
                                logging.info( f'The given idx ({idx}) is before or within the import statements. ' 
                                                f'Correcting and placing it after the last import at position {last_import_pos}.'
                                )
                                python_template.body.insert(last_import_pos, ast_node)
                            else:
                                python_template.body.insert(idx, ast_node)
                    else:
                        # Perhaps check for any other elemnts especially if the __name__ == '__main__' type elements is present which can be used as 
                        # an anchor point to to place the class or another class or function can be used to place
                    
                        function_positions = [pos for pos, stmt in enumerate(ast.iter_child_nodes(python_template))
                            if isinstance(stmt, ast.FunctionDef)]
                        
                        _name_format = [pos for pos, stmt in enumerate(ast.iter_child_nodes(python_template)) # This is for the __name__ format if 
                            if isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Compare)]
                        
                        if function_positions:
                            last_func_pos = function_positions[-1]
                            python_template.body.insert(last_func_pos, ast_node)
                        
                        elif not function_positions and _name_format:
                            last_name_format_pos = _name_format[0]
                            python_template.body.insert(last_name_format_pos,ast_node)
                        elif not (function_positions and _name_format) and import_positions:
                            last_import_pos = import_positions[-1]
                            python_template.body.insert(last_import_pos,ast_node)
                        
                elif isinstance(ast_node, ast.FunctionDef):
                    if idx:
                        # Gather positions of all import statements
                        import_positions = [ pos for pos, stmt in enumerate(ast.iter_child_nodes(python_template))
                            if isinstance(stmt, (ast.Import, ast.ImportFrom))]
                
                        if import_positions:
                            last_import_pos = import_positions[-1] + 1
                            if idx <= last_import_pos:
                                logging.info( f'The given idx ({idx}) is before or within the import statements. ' 
                                                f'Correcting and placing it after the last import at position {last_import_pos}.'
                                )
                                python_template.body.insert(last_import_pos, ast_node)
                            else:
                                python_template.body.insert(idx, ast_node)
                        else:
                            # No import statements found, safe to insert at the given idx
                            python_template.body.insert(idx, ast_node)
                    else:
                        # No idx provided; insert before the last function definition if present and after the imports
                        function_positions = [pos for pos, stmt in enumerate(ast.iter_child_nodes(python_template))
                            if isinstance(stmt, ast.FunctionDef)]
                
                        if function_positions:
                            last_func_pos = function_positions[-1]
                            python_template.body.insert(last_func_pos, ast_node)
                        else:
                            # No functions yet; append at the end
                            python_template.body.append(ast_node)
                            
                elif isinstance(ast_node, ast.Assign):
                    if method_name: # Inside a method/function
                        # print([func for func in ast_walk(python_template,ast.FunctionDef) if func.name == method_name])
                        if isinstance(python_template, ast.FunctionDef):
                            target_func = python_template
                        else:
                            target_func = [ func for func in ast_walk(python_template, ast.FunctionDef)if func.name == method_name][0] 
                        if target_func:
                            class_name = kwargs.get('class_name')
                            method = kwargs.get('method')
                            insert_pos,return_stmt_pos = self._check_position_within_function(target_func,class_name=class_name,method_name=method)
                            
                            # Use idx only if it's between assign and return positions
                            if idx and (return_stmt_pos is None or idx < return_stmt_pos) and idx > insert_pos:
                                target_func.body.insert(idx, ast_node)
                            else:
                                logging.info(f'Inserting after last assign at position {insert_pos} inside the function : {method_name}')
                                target_func.body.insert(insert_pos, ast_node)
                    else:
                        assign_positions = [i for i, stmt in enumerate(python_template.body) if isinstance(stmt, ast.Assign)]
                        insert_pos = assign_positions[-1] + 1 if assign_positions else len(python_template.body)

                        if idx:
                            python_template.body.insert(idx, ast_node)
                        else:
                            logging.info(f'Inserting after last assign at position {insert_pos} inside the parent body')
                            python_template.body.insert(insert_pos, ast_node)

                elif isinstance(ast_node, ast.Expr):
                    if method_name: # Method inside which we need to place the call statement
                        if isinstance(python_template,ast.FunctionDef):
                            target_func = python_template
                        else:
                            target_func = [func for func in ast_walk(python_template,ast.FunctionDef) if func.name == method_name][0]
                            
                        if target_func:
                            # First we will handle the special case where the call statement is of read_dummy to see if the any of the 
                            # arguments sent to it isn't applied before.
                            is_read_dummy = (
                                isinstance(ast_node, ast.Expr) and isinstance(ast_node.value, ast.Call)
                                and isinstance(ast_node.value.func, ast.Name)
                                and ast_node.value.func.id == "read_dummy"
                            )
                            if is_read_dummy:
                                # Retrieve the arguments that are sent to the read_dummy
                                args = [args.id for args in ast_node.value.args]
                                assign_statements = [stmt.targets[0].id for stmt in target_func.body if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name)]
                                
                                if any(set(args) | set(assign_statements)): # IF there are arguments before that needs to be intiialized
                                    class_name = kwargs.get('class_name')
                                    method = kwargs.get('method')
                                    insert_pos,return_stmt_pos = self._check_position_within_function(target_func,class_name=class_name,method_name=method)
                                        
                                    if idx and (return_stmt_pos is None or idx < return_stmt_pos) and idx > insert_pos:    
                                        target_func.body.insert(idx,ast_node)
                                    else:
                                        logging.info(f'SPECIAL CASE, READ dummy method needs to be placed after all the assign statement(variables), insert position: {insert_pos}')
                                        target_func.body.insert(insert_pos, ast_node)
                            else:
                                class_name = kwargs.get('class_name')
                                method = kwargs.get('method')
                                insert_pos,return_stmt_pos = self._check_position_within_function(target_func,class_name=class_name,method_name=method)
                                if idx and (return_stmt_pos is None or idx < return_stmt_pos) and idx > insert_pos:
                                    target_func.body.insert(idx,ast_node)
                                else:
                                    target_func.body.insert(insert_pos, ast_node)
                    else:
                        python_template.body.append(ast_node)
        except Exception:
            logging.exception(f'Exception error in insert_at')
            raise

    def _is_scalar_var(self, dec_statement) -> str | None:
        """
        Determines if a declaration statement corresponds to a scalar or logical variable.

        This helper method is used to distinguish scalar or logical variables from arrays.
        A variable is considered scalar/logical if:
        - It does **not** have an initialization aspect (i.e., no default value assigned), and
        - It does **not** have a `DIMENSION` attribute (i.e., not declared as an array).

        Parameters
        ----------
        dec_statement : Any
            A declaration statement (typically a parsed representation of a Fortran declaration).
            Expected to support inspection for initialization and dimension attributes.

        Returns
        -------
        str or None
            The variable name if it is identified as a scalar or logical, otherwise `None`.
        """

        var = walk(dec_statement, F23.Entity_Decl)[0].string
        init_spec = any(walk(dec_statement, F23.Initialization))
        alloc_spec = any(walk(dec_statement, F23.Dimension_Attr_Spec))
        if not init_spec and not alloc_spec:
            return var
        return None
        
    def separate_scalar(self,dec_stmts:List=None) -> None:
        """
        Identifies and separates scalar or logical variables from declaration statements.

        This method populates `self.scalar` with variable names identified as scalars or logicals.
        It operates in three modes depending on the `dec_stmts` input and the value of `self.global_state`.

        Modes
        -----
        1. If `dec_stmts` is provided:
            - Iterates over the provided declaration statements.
            - Checks for variables with `INTENT(IN)` or `INTENT(INOUT)` attributes.
            - Adds scalar or logical variable names to `self.scalar`.

        2. If `dec_stmts` is not provided and `self.global_state` is True:
            - Iterates over global declarations (`self.dec_global`).
            - Identifies scalar or logical variables and adds their names to `self.scalar`.

        3. If `dec_stmts` is not provided and `self.global_state` is False:
            - Examines dummy arguments (`self.var_dummy`) for the current subroutine.
            - Filters variables with `INTENT(IN)` or `INTENT(INOUT)`.
            - Identifies scalar or logical variables and adds their names to `self.scalar`.

        Parameters
        ----------
        dec_stmts : list of str or None, optional
            A list of declaration statements (e.g., parsed Fortran declarations).
            If provided, only these statements will be processed. If `None`, the method
            uses internal structures (`self.dec_global` or `self.var_dummy`) based on 
            the value of `self.global_state`.

        Returns
        -------
        None
            Modifies `self.scalar` in place with the names of identified scalar/logical variables.
        """
        try:
            self.scalar = []
            if dec_stmts is None:
                if self.global_state:
                    for var in self.variable_order:
                        dec_statement = self.extractor.dec_global[self.subroutine_name][var]
                        varname = self._is_scalar_var(dec_statement)
                        if varname:
                            self.scalar.append(var)
                else: # This is used to separate scalar present in the var_dummy based on their INTENT(IN,INOUT)
                    for dec_statement in self.extractor.var_dummy[self.subroutine_name]:
                        if any(i.tostr() in ["IN", "INOUT"] for i in walk(dec_statement, F23.Intent_Spec)):
                            varname = self._is_scalar_var(dec_statement)
                            if varname:
                                self.scalar.append(varname)
            else:
                for dec_statement in dec_stmts:
                    # Check if the statement has an Intent_Spec with IN or INOUT
                    has_intent = any(i.tostr() in ["IN", "INOUT"] for i in walk(dec_statement, F23.Intent_Spec))

                    if has_intent:
                        varname = self._is_scalar_var(dec_statement)
                        if varname:
                            self.scalar.append(varname)
                    else:
                        varname = self._is_scalar_var(dec_statement)
                        if varname:
                            self.scalar.append(varname)
        except Exception as e:
            logging.error(f'Exception in separate_scalar')
            raise 
    
    def read_file_ast(self,assign_nodes:List) -> List:
        """
        Generates code lines to read variables from a Fortran binary file.

        This method constructs individual lines of code required to read each variable 
        from a binary file, based on provided assignment AST nodes. The resulting lines 
        typically resemble: `ffile.read_ints(np.int32)[0]`, depending on the data type.

        Parameters
        ----------
        assign_nodes : list of ast.Assign
            A list of AST assignment nodes representing the variables to be read 
            from the binary file.

        Returns
        -------
        var_list : list of str
            A list of code lines (as strings) that perform reading operations 
            for each variable from the binary file.
        """
        var = None
        var_name = None
        target = None
        var_list = []
        
        for assign_node in assign_nodes:
            if isinstance(assign_node,ast.Assign) and isinstance(assign_node.targets[0], ast.Name):
                var_name = assign_node.targets[0].id
                target = ast.Name(id=var_name,ctx = ast.Store())
            elif isinstance(assign_node,ast.Assign) and isinstance(assign_node.targets[0], ast.Attribute):
                var_name = assign_node.targets[0].attr
                target = ast.Attribute(
                    value = ast.Name(id='self',ctx=ast.Load()),
                    attr = var_name,
                    ctx = ast.Store()
                )
            
            if len(assign_node.value.keywords) == 0: # This means they are just intergers,reals or logical values mostly scalars 
                attr_type = assign_node.value.func.attr
                read_type = 'read_reals' if attr_type == "float64" else 'read_ints'
                if attr_type == "bool":
                    # Logical values representation: .TRUE. is mostly respresent with -1 because all bits are set to 1
                    # .FALSE. is represented by 0
                    # https://stackoverflow.com/a/39454385
                    subscript_format = ast.Call(
                                        func=ast.Attribute(
                                            value = ast.Name(id='np',ctx=ast.Load()),
                                            attr="bool",
                                            ctx = ast.Load()
                                            ),
                                        args=[
                                            ast.Subscript(
                                                    value = ast.Call(
                                                        func = ast.Attribute(
                                                            value= ast.Name(id = 'ffile',ctx=ast.Load()),
                                                            attr='read_ints',
                                                            ctx = ast.Load()),
                                                        args = [
                                                            ast.Attribute(
                                                                value = ast.Name(id = 'np',ctx=ast.Load()),
                                                                attr = "int32",
                                                                ctx=ast.Load()) ],
                                                        keywords = []),
                                                    slice = ast.Constant(value=0),
                                                    ctx=ast.Load()
                                                )
                                            ],
                                        keywords=[])
                    
                else:
                    subscript_format = ast.Subscript(
                                        value = ast.Call(
                                            func = ast.Attribute(
                                                value= ast.Name(id = 'ffile',ctx=ast.Load()),
                                                attr=read_type,
                                                ctx = ast.Load()),
                                            args = [
                                                ast.Attribute(
                                                    value = ast.Name(id = 'np',ctx=ast.Load()),
                                                    attr = attr_type,
                                                    ctx=ast.Load()) ],
                                            keywords = []),
                                        slice = ast.Constant(value=0),
                                        ctx=ast.Load()
                                            
                                    )
                
                var = ast.Assign(
                    targets = [target],
                    value =  subscript_format
                )
                    
            else: # This means that they all are arrays
                attr_type = assign_node.value.keywords[0].value.attr
                read_type = 'read_reals' if attr_type == "float64" else 'read_ints'
                # print(arr_shape)
                call_stmt = ast.Call(
                        func= ast.Attribute(
                            value= ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id='ffile', ctx=ast.Load()),
                                    attr=read_type,
                                    ctx=ast.Load()),
                                args=[
                                    ast.Attribute(
                                        value=ast.Name(id='np', ctx=ast.Load()),
                                        attr=attr_type,
                                        ctx=ast.Load())],
                                keywords=[]),
                            attr='reshape',
                            ctx=ast.Load()),
                        args=[assign_node.value.args[0]],
                        keywords=[ ast.keyword( arg='order', value=ast.Constant(value='F'))])
                    
                var = ast.Assign(
                    targets=[
                    ast.Subscript(
                        value=target,
                        slice=ast.Slice(),
                        ctx=ast.Store())],
                    value = call_stmt 
                )
            var_list.append(var)
            
        return var_list
    
    def init_dependant_variables(self,read_ast:ast.Module,assign_nodes:List) -> List: 
        """
        Initializes variables that depend on previously read input attributes.

        This method should be called after critical dependencies (e.g., configuration 
        parameters, dimensions) have been read from an input file. It ensures that 
        dependent variables are only initialized after all required 'dependee' variables 
        are available.

        The initialization order is determined using a dependency mapping (`self.dependant_variables`), 
        which specifies which variables depend on others. For each dependent variable, 
        the method finds the latest initialization point of its dependencies and inserts 
        the initialization code at the appropriate position in the AST.

        Parameters
        ----------
        read_ast : ast.Module
            The AST representing the read template from which variable values are extracted.
        assign_nodes : list of ast.Assign
            A list of assignment nodes representing variables that require dependency-based initialization.

        Returns
        -------
        list of ast.AST
            A list of AST nodes representing the updated body of `read_ast` with inserted initialization logic.
        """
        var_name = None
        for key in list(self.dependant_variables.keys()):
            try:
                # Get all assign statements from the read_ast
                assign_stmts = [(i,element) for i, element in enumerate(read_ast.body) if isinstance(element,ast.Assign)]

                dependees = self.dependant_variables[key]
                max_pos = -1 # This will allows us to find after which variable should we place the init of the dependant variables
                # Since each depandant variables might have multiple dependee variables which are init at different locations as such we try to find the
                # furthest/last positin of the variable dependee 

                # Find the max position among all dependee variables
                for i, stmt in assign_stmts:
                    if not stmt.targets:
                        raise ValueError("assign_node has no targets")
                    target = stmt.targets[0]
                    if isinstance(target, ast.Name):
                        var_name = target.id
                    elif isinstance(target, ast.Attribute):
                        var_name = target.attr
                    
                    if var_name is None:
                        raise AttributeError(f"node doesn't have either attribute or id or attr :{ast.unparse(ast.fix_missing_locations(stmt))} ")
                    
                    if var_name in dependees:
                        max_pos = max(max_pos, i)

                # We insert AFTER the latest dependency
                position = max_pos + 1
                for assign_node in assign_nodes:
                    try:
                        if not assign_node.targets:
                            raise ValueError("assign_node has no targets")
                        target = assign_node.targets[0]
                        if isinstance(target, ast.Name):
                            var_name = target.id
                        elif isinstance(target, ast.Attribute):
                            var_name = target.attr
                        else:
                            raise TypeError(f"Unsupported assignment target type: {type(target).__name__}")
                        
                        if var_name is None:
                           raise AttributeError(f"node doesn't have either attribute or id or attr :{ast.unparse(ast.fix_missing_locations(assign_node))} ")

                        if var_name is not None and var_name == key:
                            read_ast.body.insert(position,assign_node)
                    except (AttributeError,IndexError,ValueError) as e:
                        raise 
            except Exception:
                logging.exception(f'Exception in init_dependant_variable')
                    
        # read_ast = ast.fix_missing_locations(read_ast)
        return read_ast.body

    
    def transfer_to_pyfile(self, tree:ast.Module, folder_name:str="hydrol",python_file_type:Literal["module_global","main"] = "module_global") -> None:
        """
        Writes the finalized Python AST to a Python file based on its type.

        This method takes a finalized Python AST and writes it to a file, depending 
        on the specified file type (`module_global` or `main`). The output file is placed 
        in a designated folder, typically named `python_benchmark`, and the structure 
        of the generated Python file is based on the AST content and file type.

        Parameters
        ----------
        tree : ast.Module
            The finalized Python AST to be written to a file.
        folder_name : str, optional
            The name of the directory where the Python files will be saved.
            Defaults to `'python_benchmark'`.
        python_file_type : str, optional
            The type of Python file to generate. Can be either `'module_global'` or `'main'`.
            Defaults to `'module_global'`.

        Returns
        -------
        None
            This method performs file I/O.
        """
        try:
            current_dir = os.getcwd()

            path_to_folder = find_folder(current_dir,target_folder=folder_name)
            
            if path_to_folder is None:
                raise ValueError(f"For the given folder, it couldn't be found: {path_to_folder}")
            
            subroutine_path = os.path.join(path_to_folder, self.subroutine_name)
            file_path = os.path.join(subroutine_path, f"{python_file_type}.py")

            # First create python benchmark directory which will contain the directories of each subroutines dir within which contains the output of the subroutines test
            logging.info("Creating benchmark directory...")
            os.makedirs(path_to_folder, exist_ok=True)

            # Then the subroutine directory within the benchmark
            logging.info("Creating subroutine directory...")
            os.makedirs(subroutine_path, exist_ok=True)

            logging.info(f"Writing Python file: {file_path}")
            with open(file_path, "w") as f:
                f.write("#!/usr/bin/env python3\n")
                f.write(ast.unparse(tree))

            rights = stat.S_IRWXU
            os.chmod(file_path,rights)

            logging.info("File successfully written.")

        except Exception:
            logging.exception(f"Exception in transfer_to_pyfile")

    @staticmethod
    def python_parser(code:str) -> ast.Module:
        """
        Parses a Python code string into an AST module.

        Attempts to convert the given Python source code string into an abstract syntax tree (AST)
        using the built-in `ast` module. Logs an informational message on successful parsing, 
        or an error message if a `SyntaxError` is encountered during parsing.

        Parameters
        ----------
        code : str
            The Python source code to parse.

        Returns
        -------
        tree : ast.Module or None
            The parsed AST module if the code is syntactically valid, otherwise `None`.
        """
        try:
            tree = ast.parse(code)
            logging.info("INFO: Parsed python template is valid")
            return tree
        except SyntaxError as e:
            logging.error(f'ERROR: Syntax error: {e}')
            return None
    
    def insert_all_assign_nodes(self,assign_nodes:List,code_tree:ast.Module,method_name:str,**kwargs) -> None:
        """
        Inserts all assignment nodes into a specified method within a global code template.

        This method inserts a list of `ast.Assign` nodes into the body of a method (typically `__init__` 
        or another subroutine) within the given AST module (`code_tree`). The insertion point 
        is determined based on the method name and optionally guided by additional keyword arguments 
        for more precise placement.

        Parameters
        ----------
        assign_nodes : list of ast.Assign
            A list of AST assignment nodes to be inserted into the method.
        code_tree : ast.Module
            The Python AST module representing the global code template where the nodes 
            will be inserted.
        method_name : str
            The name of the method inside which the assignment nodes should be inserted.
        **kwargs : dict, optional
            Additional context used for determining the correct insertion point (dependency order, placeholder positioning).

        Returns
        -------
        None
            This method modifies the `code_tree` in place and does not return a value.
        """

        try:
            name = None
            name_to_node = {}
            if self.global_state:
                diff = list(set(
                    assign.targets[0].id if isinstance(assign.targets[0], ast.Name) else assign.targets[0].attr
                    for assign in assign_nodes) - set(self.variable_order))
                # DO it in two steps first the declared and intializd variables and then the variable order
                # The declared and non intialized variables 
                if len(diff) != 0:
                    for assign_node in assign_nodes:
                        if not assign_node.targets:
                            raise ValueError("assign_node has no targets")
                        target = assign_node.targets[0]
                        if isinstance(target, ast.Name):
                            name = target.id
                        elif isinstance(target, ast.Attribute):
                            name = target.attr
                        else:
                            raise TypeError(f"Unsupported assignment target type: {type(target).__name__}")
                        
                        if name is None:
                            raise AttributeError(f"node doesn't have either attribute or id or attr :{ast.unparse(ast.fix_missing_locations(assign_node))} ")
                        
                        if name in diff:
                            # self.insert_at(None,assign_node,code_tree,method_name=method_name)
                            name_to_node[name] = assign_node
                    
                    ordered_vars = order_assignments(assign_nodes, diff)
                    # Insert assignment nodes in the resolved order
                    for var in ordered_vars:
                        if var in name_to_node:
                            self.insert_at(None, name_to_node[var], code_tree, method_name=method_name)

                # Now all the declared and not intialized variables
                # assign_node_names = [assign.targets[0].id if isinstance(assign.targets[0], ast.Name) else assign.targets[0].attr  for assign in assign_nodes]
                for var in self.variable_order:
                    for assign_node in assign_nodes:
                        if not assign_node.targets:
                            raise ValueError("assign_node has no targets")
                        target = assign_node.targets[0]
                        if isinstance(target, ast.Name):
                            name = target.id
                        elif isinstance(target, ast.Attribute):
                            name = target.attr
                        else:
                            raise TypeError(f"Unsupported assignment target type: {type(target).__name__}")
                        
                        if name is None:
                            raise AttributeError(f"node doesn't have either attribute or id or attr :{ast.unparse(ast.fix_missing_locations(assign_node))} ")
                        
                        if var == name and var not in list(self.dependant_variables.keys()):
                            self.insert_at(None,assign_node,code_tree,method_name=method_name)
            else:
                for assign_node in assign_nodes:
                    target = assign_node.targets[0]
                    if isinstance(target, ast.Name):
                        name = target.id
                    elif isinstance(target, ast.Attribute):
                        name = target.attr
                    else:
                        raise TypeError(f"Unsupported assignment target type: {type(target).__name__}")
                    
                    name_to_node[name] = assign_node

                ordered_vars = order_assignments(assign_nodes,None)
                for var in ordered_vars:
                    if var in name_to_node:
                        self.insert_at(None,name_to_node[var],code_tree,method_name=method_name,**kwargs)

            # code_tree = ast.fix_missing_locations(code_tree)
        except Exception as e:
            logging.error(f'Exception in insert_all_assign_nodes')
            raise

    def create_test_function(self, cls_info) -> ast.FunctionDef:
        """
        Create a test function to compare the output of the Python code with the FORTRAN output saved in `output.bin`.

        Parameters
        ----------
        cls_info : dict
            Dictionary containing all the information of classes that some variable might depend on.

        Returns
        -------
        ast.FunctionDef
            Function AST to test the output of Python with that of FORTRAN.
        """

        try:
            for key in list(cls_info.keys()):
                try:
                    instance_name = list(cls_info[key].keys())[0]
                    attr = cls_info[key][instance_name]["attributes"]
                except (IndexError, KeyError, TypeError) as e:
                    logging.error(f"Error accessing attributes for key '{key}': {e}")
                    raise

                type_ = {"REAL": "float64",
                        "INTEGER": "int32"}
                args = []
                # Now to see if we need to an instance of the class 
                if any(set(attr) & self.extractor.var_modif[self.subroutine_name]):
                    args.append(ast.arg(arg=instance_name))
                    
                # THese are non global args that meant to be sent as args
                arg = [ast.arg(arg = arg) for arg in self.extractor.var_modif[self.subroutine_name] - set(attr)]
                args.extend(arg)
                
                # First create an empty function 
                function_def = ast.FunctionDef(
                    name = f"test_{self.subroutine_name}",
                    args=ast.arguments(
                        posonlyargs=[],
                        args=args,
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[]),
                    body=[],
                    decorator_list=[]
                )
                code = """
print('--- inside the test function for {subroutine_name} ---')
path = f'{benchmark_dir}/{subroutine_name}/output.bin'
ffile = FortranFile(path, 'r')
                """
                try:
                    code = code.format(benchmark_dir=self.benchmark_dir, subroutine_name=self.subroutine_name)
                    tree = ast.parse(code).body
                except (SyntaxError, KeyError) as e:
                    logging.error(f"Error in formatting/parsing test function body: {e}")
                    raise

                function_def.body = tree 
                
                # THe primary constraint is the fact that we have class attributes and local all together and we also need to see the variable
                # that's being read and compared, first use the var_modif_info.keys() to keep in check the variable read. 
                modif_var = ast.Assign(
                    targets = [ast.Name(id = "modif_var", ctx = ast.Store())],
                    value = ast.List(elts = [ast.Constant(value = variable) for variable in list(self.extractor.var_modif_info[self.subroutine_name].keys())],
                                    ctx = ast.Load()
                            )
                )
                # The for loop 
                for_loop = ast.For(
                    target = ast.Tuple(
                        elts = [ast.Name(id='variable', ctx=ast.Store()),
                            ast.Name(id='value', ctx=ast.Store())],
                        ctx = ast.Store()
                    ),
                    iter = ast.Call(
                        func = ast.Name(id = 'zip', ctx = ast.Load()),
                        args = [
                            ast.Name(id = 'modif_var', ctx = ast.Load()),
                            ast.List(elts = [ast.Name(id = variable,ctx=ast.Load()) for variable in list(self.extractor.var_modif_info[self.subroutine_name].keys())],
                                    ctx = ast.Load()
                            )
                        ],
                        keywords = []
                    ),
                    body=[],
                    orelse =[]
                    
                )
                # We send the for loop to replace the variables that has 
                try:
                    for_loop = ReplaceGlobals(cls_info).visit_For(for_loop)
                except Exception:
                    logging.exception(f"ReplaceGlobals failed")
                    raise
                
                # Now we need to create the core of the for loop which only does the matching of variables read from the ouput.bin of the Fortran values
                # with that of the python, which uses the known shape of the variables to ensure that we retrieve them in proper shape and then compare
                # For arrays, we will use the allclose to see if two arrays has the same shape and values: https://numpy.org/doc/2.3/reference/generated/numpy.allclose.html
                # For scalars, we will use isclose which is helpful when comparing floating points precision : https://numpy.org/devdocs/reference/generated/numpy.isclose.html
                
                templates = self.load_code_templates(self.config_path)
                if templates is None:
                    raise ValueError("Templates could not be loaded due to a prior error.")
                        
                code = templates["Python_templates"]["Python_test_output_template"]["template"]        

                if code is None:
                    raise ValueError(f'Test output template is None')
                
                try:      
                    core_step = ast.parse(code).body # we retreive only the body and which would allow us to add it to the the for loop body
                    for_loop.body = core_step

                except SyntaxError as e:
                    logging.error(f"Syntax error while parsing the code template: {e}")
                    raise

                # Now we append the modif_var_list and the for loop inside the test function 
                function_def.body.append(modif_var)
                function_def.body.append(for_loop)
                        
                # print(ast.unparse(ast.fix_missing_locations(function_def)))
                return function_def
        except Exception as e:
            logging.error(f'Exception in create_test_function')
            return None 
        
    ############################################################################################ Global python ############################################################################################
    
    def prepare_read_code_global_template(self, assign_nodes:List) -> ast.Module:
        """
        Generate and populate the read code template for reading a Fortran binary file.

        This method builds the structure for reading a Fortran binary file line by line,
        supporting both standalone Python scripts and class-based approaches. It inserts
        assignment nodes into the template to define how each variable should be read.

        Parameters
        ----------
        assign_nodes : list
            List of `ast.Assign` nodes for each variable.

        Returns
        -------
        ast.Module
            The modified read template filled with the assignment nodes.
        """

        try:
            templates = self.load_code_templates(self.config_path)
            if templates is None:
                raise ValueError("Templates could not be loaded due to a prior error.")
            
            template_str = templates["Python_templates"]["Python_read_global_template"]["template"]
            read_code_template =template_str.format(
                benchmark_dir=self.benchmark_dir,
                subroutine_name=self.subroutine_name
            )
            # print(read_code_template)
            read_ast = self.python_parser(read_code_template)
            if read_ast is None:
                raise ValueError(f'read ast for prepare read code global template is None due to prior error')
            # print(ast.dump(read_ast,indent=4))
            # We need to ensure that the assign_nodes follows the `self.variable_order`
            assign_nodes.sort(
                key=lambda node: self.variable_order.index(getattr(node.targets[0],'id',getattr(node.targets[0],'attr',None))) if getattr(node.targets[0],'id',getattr(node.targets[0],'attr',None)) in self.variable_order else float('inf') 
            )
            # Then we remove the nodes that are not necessary to be read since assign_nodes contains all the assign statement within the global python template
            assign_temp = []
            for assign_node in assign_nodes:
                target = assign_node.targets[0]
                name = getattr(target,'id', getattr(target,'attr',None))
                if name in self.variable_order:
                    assign_temp.append(assign_node)

            var_list = self.read_file_ast(assign_nodes=assign_temp)
            for variable in var_list:
                read_ast.body.append(variable)

            # read_ast = ast.fix_missing_locations(read_ast)
            return read_ast
        
        except Exception:
            logging.exception(f"Error from prepare_read_code_global_template method")
            return None

    def prepare_read_code_for_global_template(self,assign_nodes:List) -> ast.Module:
        """
        Initialize scalar variables by reading them line by line from a Fortran binary file.

        This method assumes that the necessary `for` loops for reading array data are already
        present in the `read_code_template`. It focuses on scalar variables, which are read
        individually and inserted into their appropriate positions within the code template.
        During this process, the names of the scalar variables are also added to the list
        of variables handled inside the `for` loop.

        Parameters
        ----------
        assign_nodes : list
            List of `ast.Assign` nodes for each variable.

        Returns
        -------
        ast.Module
            Python AST of the read template with the newly added elements.
        """

        try:
            template_name = "Python_read_for_loop_template" if not self.cls_mode else "Python_read_for_loop_class_template"
            templates = self.load_code_templates(self.config_path)
            if templates is None:
                raise ValueError("Templates could not be loaded due to a prior error.")
            
            for_template_str = templates["Python_templates"][template_name]["template"]

            read_code_template =for_template_str.format(
                benchmark_dir=self.benchmark_dir,
                subroutine_name=self.subroutine_name,
            )
            # print(read_code_template)
        
            read_ast = self.python_parser(read_code_template)
            if read_ast is None:
                raise ValueError(f'read_ast for the python read code for global template is None due to prior error')
            
            if assign_nodes:
                var_list = self.read_file_ast(assign_nodes)
                # In order to use for loop within the python script we will first read the scalars line by line and for the arrays we will read it using a for loop
                var_pos = [i for i,element in enumerate(ast.iter_child_nodes(read_ast)) if isinstance(element,ast.For)][0] 
                
                for variable in var_list:
                    read_ast.body.insert(var_pos, variable)
                    var_pos+=1               
            # The arrays are read through a loop instead of reading them line by line and Now fill up the list for the for loop with the read_ast

            # We apply the same prinicple for the class aspect but the primary differences situates within the self and thanks to the getattr and hasattr
            # methods which allows us to retrieve a class attribute and modify it dynamically allowing us to do a proper changement instead of using globals
            for_ast = next(iter(ast_walk(read_ast,ast.For)))
            # print(for_ast)
            if for_ast.iter.elts == []:
                difference = [item for item in self.variable_order if item not in self.scalar]
                for_ast.iter.elts = [ast.Constant(var) for var in difference]
            
            # read_ast = ast.fix_missing_locations(read_ast)
            return read_ast
        
        except Exception:
            logging.exception(f"Error from prepare_read_code_for_global_template method")
            return None

    def convert_global_read_subroutine(self,assign_nodes:List,code_template:ast.Module) -> None:
        """
        Populate the code template with all necessary elements for the `module_global` file.
        This method uses previously defined methods to assemble and insert all required components,
        ensuring that the `module_global` file is fully constructed.

        Parameters
        ----------
        assign_nodes : list
            List of assignment nodes (ast.Assign) for each variable.
        code_template : ast.Module
            AST tree of the code template that will be modified.

        Returns
        -------
        None
            This methods modifies directly the code_template.
        """
        try:
            # The variable order will be retrieved since we the instance of the isolator class which use the processor and extractor class
            function_def = ast_walk(code_template,ast.FunctionDef)
            # print(ast.dump(read_ast,indent=4))
            
            # Retrieved the scalar/Logical variables that will be read 
            self.separate_scalar()

            if self.for_loop:
                if self.scalar:
                    assign_map = {}
                    for assign_node in assign_nodes:
                        target = assign_node.targets[0]
                        name = target.id if isinstance(target, ast.Name) else target.attr
                        assign_map[name] = assign_node  
                    nodes = [assign_map[scalar] for scalar in self.scalar if scalar in assign_map]

                    read_ast = self.prepare_read_code_for_global_template(assign_nodes=nodes)
                    if read_ast is None:
                        raise ValueError(f'global read ast is None using for loop')
            else:
                read_ast = self.prepare_read_code_global_template(assign_nodes)
                if read_ast is None:
                    raise ValueError(f'global read ast is None without using for loop')

            read_ast_list = self.init_dependant_variables(read_ast,assign_nodes)
            
            for functions in function_def:
                if functions.name == "declaration_initialization": # IF we find the declaration intiailization method to read and fill tables
                    if self.scalar:
                        try:
                            tree = ast.parse(f"global {', '.join(self.scalar + list(self.dependant_variables.keys()))}") # self.scalar + list(self.dependant_variables.keys())
                            
                            functions.body.append(tree.body[0])
                        except (SyntaxError,AttributeError):
                            raise 
                        
                    # ast.iter_child_nodes(read_ast)
                    for elem in read_ast_list:
                        functions.body.append(elem)
                else:
                    print(functions.name)

            # What this does it fix the missing location(lineno,end_lineno,col_offset,end_col_offset) based on the parent node
            # https://docs.python.org/3/library/ast.html#ast.fix_missing_locations 
            code_template = ast.fix_missing_locations(code_template)

            return code_template 
        except Exception:
            logging.exception(f'Exception in convert_global_read_subroutine')
            return None
        
    def transform_to_python_script(self, ast_nodes:List) -> ast.Module:
        """
        Transform from AST Fortran to an AST Python script approach for a global module.

        Parameters
        ----------
        ast_nodes : list
            List of assignment nodes (ast.Assign) or import nodes (ast.Import or ast.ImportFrom).

        Returns
        -------
        code_tree : ast.Module
            The finalized Python script AST containing all elements of the transformation.
        """
        try:
            code_tree = self.out_module_python()
            if code_tree is None:
                raise ValueError(f'Code_tree is None')

            assign_nodes = []
            procedure_nodes = []
            
            for node in ast_nodes:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    procedure_nodes.append(node)
                elif isinstance(node, (ast.Assign, ast.Assign)):
                    assign_nodes.append(node)

            if procedure_nodes:
                for procedure_node in procedure_nodes:
                    self.insert_at(None,procedure_node,code_tree)

            # Specification part
            self.insert_all_assign_nodes(assign_nodes=assign_nodes,code_tree=code_tree,method_name=None)
            # This is for the functions part
            functions_spec = ast_walk(code_tree,ast.FunctionDef)

            for functions in functions_spec:
                if functions.name == "declaration_initialization":
                    code_tree = self.convert_global_read_subroutine(assign_nodes=assign_nodes,code_template=code_tree)
                    if code_tree is None:
                        raise ValueError(f'Code_tree is None')
                        
            return code_tree
        except Exception:
            logging.exception(f'Exception error in transform_to_python_script')
            raise
    
    def transform_to_class(self, ast_nodes:List) -> ast.Module:
        """
        Transform from AST Fortran to an AST Python class approach for a global module.

        Parameters
        ----------
        ast_nodes : list
            List of assignment nodes (ast.Assign) or import nodes (ast.Import or ast.ImportFrom).

        Returns
        -------
        code_tree : ast.Module
            The finalized Python class AST containing all elements of the transformation.
        """

        try:
            class_tree = self.out_module_python()
            if class_tree is None:
                raise ValueError(f'Class_tree is None')
            
            # We need to modify the class name since if we call the global module files with the same class name it might cause errors as such we change only the 
            # class name to make it correspond to the subroutine it self.
            class_defs = ast_walk(class_tree,ast.ClassDef)

            for class_def in class_defs:
                if class_def.name == 'Global_module':
                    class_def.name = "_".join(["Global_module", self.subroutine_name])

            functions_spec = ast_walk(class_tree,ast.FunctionDef)
            assign_nodes = []
            procedure_nodes = []

            for node in ast_nodes:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    procedure_nodes.append(node)
                elif isinstance(node, ast.Assign):
                    assign_nodes.append(node)
            # If the procedure is present then that we add them to the code_template
            if procedure_nodes:
                for procedure_node in procedure_nodes:
                    self.insert_at(None,procedure_node,class_tree)

            self.separate_scalar()
            for functions in functions_spec:
                if functions.name == "__init__":
                    self.insert_all_assign_nodes(assign_nodes,class_tree,method_name = functions.name)
                    
                elif functions.name == "declaration_initialization":
                    
                    if isinstance(functions.body[0], ast.Pass):
                        functions.body.pop(0)
                    
                    if self.for_loop:
                        # Here we will separate from the assign nodes, the scalar nodes
                        nodes = []
                        if self.scalar:
                            assign_map = {}
                            for assign_node in assign_nodes:
                                target = assign_node.targets[0]
                                name = target.id if isinstance(target, ast.Name) else target.attr
                                assign_map[name] = assign_node  

                            nodes = [assign_map[scalar] for scalar in self.scalar if scalar in assign_map]
                            
                        read_ast = self.prepare_read_code_for_global_template(nodes)
                        if read_ast is None:
                            raise ValueError(f'global read ast is None using for loop')
                    else:
                        read_ast = self.prepare_read_code_global_template(assign_nodes)
                        if read_ast is None:
                            raise ValueError(f'global read ast is None without using for loop')

                    read_ast_list = self.init_dependant_variables(read_ast,assign_nodes)

                    # ast.iter_child_nodes(read_ast)
                    for elem in read_ast_list:
                        functions.body.append(elem)
            # What this does it fix the missing location(lineno,end_lineno,col_offset,end_col_offset) based on the parent node
            # https://docs.python.org/3/library/ast.html#ast.fix_missing_locations     
            class_tree = ast.fix_missing_locations(class_tree)

            return class_tree
        except Exception:
            logging.exception(f'Exception error in transform_to_class')
            raise
    
    def update_global_python(self,subroutine_name:str,cls_mode:bool,for_loop:bool=True) -> ast.Module:
        """
        Update the global Python AST code through different steps as seen within the code.

        Parameters
        ----------
        subroutine_name : str
            Name of the isolated subroutine.
        cls_mode : bool
            If the global Python code AST should be in class mode or not.
        for_loop : bool
            Indicates whether to use a for loop within the code.

        Returns
        -------
        tree : ast.Module
            AST tree containing the finalized and updated elements.
        """
        try:
            
            self.subroutine_name = subroutine_name
            # THESE 3 attributes are set to create cls_mode, having for loops for the reading binary files or not and the global_state which allows
            # see which module we are currently working with 
            self.cls_mode = cls_mode
            self.for_loop = for_loop 
            self.global_state = True 
            tree = None
            # 1. Retreive the all the variables that will be declared but initialized yet/or just empty and the pre init variables such as kjipindex,nstlm etc...
            code_template = self.out_module_python()
            if code_template is None:
                raise ValueError(f'Code template is None')
            
            self.retreive_variable_order()
            
            self.pre_init_variables(code_template)

            # 2. Retrieve all the assignement python ast statements as well procesdure nodes(USE) for the global declarations
            declaration_stmts = list(self.extractor.dec_global[self.subroutine_name].values())
            ast_nodes = self.convert_SPECIFICATION_PART(declaration_stmts=declaration_stmts,cls_mode=cls_mode)
            if ast_nodes is None:
                raise ValueError(f'Ast_nodes are None')
            
            # 3. Search for variables that dependant one another between the variable_order and global declarations
            self.search_dependant_variables(declaration_stmts=declaration_stmts)

            # 4. Now we insert the variables and the read and initialization statement within the code template
            if cls_mode:
                tree = self.transform_to_class(ast_nodes=ast_nodes)
            else:
                tree = self.transform_to_python_script(ast_nodes=ast_nodes)
            
            #print(ast.unparse(tree))
            return tree 
        except Exception:
            logging.exception(f"Error in update_global_python method")
            return None 
            
    ############################################################################################ Main python ############################################################################################

    def prepare_read_code_for_main_template(self, var_dummy:List,assign_nodes:List[ast.AST]) -> ast.FunctionDef:
        """
        Prepare the `read_dummy` method by considering local variables declared in the main file, 
        global attributes it depends on, variables that need to be returned and updated, 
        and reading a binary file.

        Parameters
        ----------
        var_dummy : list
            List containing the argument names to be passed to the `read_dummy` function.
        assign_nodes : list of ast.AST
            List of AST assignment nodes that need to be initialized inside the function.

        Returns
        -------
        ast.FunctionDef
            An AST node representing the `read_dummy` function definition.
        """

        # The primary difference between this and that of the global read code template is that the variables are returned and these 
        # same variables are sent as arguments
        try:
            
            templates = self.load_code_templates(self.config_path)
            if templates is None:
                raise ValueError("Templates could not be loaded due to a prior error.")
            
            for_template_str = templates["Python_templates"]["Python_read_dummy_template"]["template"]  

            read_code_template =for_template_str.format(
                benchmark_dir=self.benchmark_dir,
                subroutine_name=self.subroutine_name,
            )    

            read_ast = self.python_parser(read_code_template).body[0]
            if read_ast is None:
                raise ValueError(f'read ast for main template is None due to prior error')

            # If immutable variables are sent as arguments they are resent back to be updated but for mutable variables if they are sent 
            # as arguments they don't need to be returned since these are sent as reference
            # Add the arguments to the function definition
            
            # read_ast has only one function definition and that's the read_dummy
            function_def = next(iter(ast_walk(read_ast,ast.FunctionDef)),None)
            if function_def is None:
                raise ValueError("No FunctionDef found in read_ast")
            
            dummy_list = []
            for node in var_dummy:
                intent_spec = walk(node,F23.Intent_Spec)
                if not F23.Intent_Spec('OUT') in intent_spec:
                    var_name = walk(walk(node,F23.Entity_Decl),F23.Name)[0]
                    arg_var = ast.arg(arg = var_name.string)
                    function_def.args.args.append(arg_var)
                    # We retrieve only the the input elements 
                    dummy_list.append(node)
                
            self.separate_scalar() # THis will allows us to retrieve the scalars and boolean varaibles 
            # Now we retrieve only the scalars following the order that is present in transformer.scalar 
            assign_map = {}
            for assign_node in assign_nodes:
                target = assign_node.targets[0]
                name = getattr(target, 'id', getattr(target, 'attr', None))
                assign_map[name] = assign_node  

            nodes = [assign_map[scalar] for scalar in self.scalar if scalar in list(assign_map.keys())]
            var_list = self.read_file_ast(nodes) # This will get the read stateemnt for scalars, boolean
            # BEfore adding this we need to verify that within the var_dummy that the scalars/booleans elements are read first and then the arrays
            # To do so we will check the position of these scalars/boolean among the arrays, in the case that they aren't read in this manner, need to take into account the reading positions

            dummy_var = [var.string for var in walk(dummy_list,F23.Entity_Decl)]
            arrays_to_add = []
            seen_arrays = set()
            seen_scalars = set()
            seen_names = set() # THis will allows us to avoid adding the same instance many times 
                
            # Get index of each scalar in dummy_var to determine order in which these scalar are present inside the dummyvar as such we also
            # need to handle the cases in which the arrays might be at different indexes. 
            scalar_positions = [(scalar, dummy_var.index(scalar)) for scalar in self.scalar if scalar in dummy_var]
            
            var_pos = next((i for i, node in enumerate(ast.iter_child_nodes(read_ast)) if isinstance(node, ast.For)), 0) - 1
            
            for scalar_name, scalar_pos in scalar_positions:
                # Get all arrays before this scalar that which are not present in the self.scalar and have not been previously seen/already read. 
                arrays_before_scalar = [name for name in dummy_var[:scalar_pos] if name not in self.scalar and name not in seen_arrays]
                # FIrst we add the arrays onto the read_ast 
                for assign in assign_nodes:
                    target = assign.targets[0]
                    name = getattr(target, 'id', getattr(target, 'attr', None))
            
                    if name in arrays_before_scalar:
                        new_node = self.read_file_ast([assign])[0]

                        if (isinstance(new_node.value, ast.Call) and new_node.value.args and isinstance(new_node.value.args[0], ast.Tuple)):
                            for elt in reversed(new_node.value.args[0].elts):
                                value = getattr(elt, 'value', None)
                                if isinstance(value, ast.Name):
                                    arg_name = value.id
                                    if arg_name not in seen_names:
                                        function_def.args.args.insert(0, ast.arg(arg=arg_name))
                                        seen_names.add(arg_name)
            
                        read_ast.body.insert(var_pos, new_node)
                        var_pos += 1
                        seen_arrays.add(name)
            
                # Add the scalar/booleans read AST from var_list
                if scalar_name not in seen_scalars:
                    for node in var_list:
                        if isinstance(node, ast.Assign) and node.targets:
                            target = node.targets[0]   
                            name = getattr(target, 'id', getattr(target, 'attr', None))
                            scalar_node = node if name == scalar_name else None 
                        if scalar_node:
                            read_ast.body.insert(var_pos, scalar_node)
                            var_pos += 1
                            seen_scalars.add(scalar_name)
                        else:
                            raise ValueError(f'Could not find scalar node for {scalar_name}')
                        
                arrays_to_add.extend(arrays_before_scalar) # THis is to preserve the order of binary reading files which will be used to retrieve
                # the arrays that will read through a loop.
                
            # We then add the variables from the to the list in the for loop 
            for_node = next(iter(ast_walk(read_ast,ast.For)),None)
            if for_node is None:
                raise ValueError(f'for_node is None')
            # EXCEPTIONAL CASE: in which all the arrays and scalars are being read line by line due to a scalar at the end of var dummy, we don't need the for loop
            # anymore thus could be removed or the fact we only have one element to read which could be just a scalar or boolean. 

            variables = [var.string for var in walk(dummy_list,F23.Entity_Decl)]
            table = self.scalar if not arrays_to_add else self.scalar + arrays_to_add
            difference = [item for item in variables if item not in table]
            if difference:
                for_node.iter.elts = [ast.Name(id = var,ctx = ast.Load()) for var in difference]
            else:
                # We will remove the for loop itself from the function
                for_pos = [i for i, node in enumerate(ast.iter_child_nodes(read_ast)) if isinstance(node, ast.For)][0] - 1
                read_ast.body.pop(for_pos)

            # add the return element, since we know that the scalars are immutable and arrays are mutables this means that the 
            # return element will only contain the return of the immutable elements.
            return_node = ast.Return()
            ret_stmts = []

            for ret_stmt in self.scalar :
                ret_stmts.append(ast.Name(id = ret_stmt,ctx = ast.Load()))
            
            if len(var_list) == 0: # IF there are not scalars to return just return the read_ast
                return read_ast
            elif len(var_list) > 1:
                return_node.value = ast.Tuple(
                                        elts = ret_stmts,
                                        ctx = ast.Load()
                                )
                read_ast.body.append(return_node)
            else:
                return_node.value = ret_stmts[0]
                read_ast.body.append(return_node)

            return read_ast
        except Exception:
            logging.exception(f'Exception in prepare_read_code_for_main_template')
            raise

    def update_main_python(self,out_module:ast.Module,parent_mode:bool=False):
        self.global_state = False 
        try:
            
            out_main_template = self.out_main_python()

            main_function_def = [function for function in ast_walk(out_main_template,ast.FunctionDef) if function.name == "main"]
            if main_function_def:
                main_function_def = main_function_def[-1]
            else:
                raise ValueError('main function is not present inside the out_main_template')
            
            idx = len(main_function_def.body)  # THis means that the idx represents internal counter when we add elements inside a function itself
            # In this case it will be inside the main()
            
            call_stmts = [] # Keeps the call statements of all the function
            function_stmts = [] # Keeps the all the functions ast in the order that they were created 
            
            # We will use the same approach used in the processor.update_main_program to add the elements onto the main function
            # 1. Create the global instance and then add them
            cls_info, import_nodes, instance_nodes = self.create_cls_info(out_module)
                
            if not all((cls_info,import_nodes,instance_nodes)):
                raise ValueError(f'ONe of these three elements is None:cls_info,import_nodes,instance_nodes')
            
            # Add the import node inside the main template
            for import_node in import_nodes:
                self.insert_at(idx = None,ast_node=import_node,python_template=out_main_template,method_name=None)
            
            # 2. Now we add the dummy arg variables onto the main function on which 
            declaration_stmts = [[elements] for elements in self.extractor.var_dummy[self.subroutine_name]]
                    
            ast_nodes = self.convert_SPECIFICATION_PART(declaration_stmts,fix_loc=True,cls_mode=parent_mode) # THis turns ast_nodes that contains
            # Assign statemetn and procedure statement(use, which appear as specification part in fortran) and we need to separate them into assign_nodes and precodure_nodes.
            if ast_nodes is None:
                raise ValueError(f'Ast_nodes is None')
            
            assign_nodes = []
            procedure_nodes = []
                    
            for node in ast_nodes:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    procedure_nodes.append(node)
                elif isinstance(node, (ast.Assign, ast.Assign)):
                    assign_nodes.append(node)
            
            # Now we need to ensure that we add the procedure_nodes if they exist
            if procedure_nodes:
                for procedure_node in procedure_nodes:
                    self.insert_at(None,procedure_node,out_main_template)
            
            if not parent_mode:
                
                # Since we add the global instance first since the following variables could depend on this global instance
                for instance_node in instance_nodes:
                    self.add_instance(idx,instance_node,cls_info,main_function_def,["declaration_initialization"])
                    
                idx = len(main_function_def.body)

                self.insert_all_assign_nodes(assign_nodes,main_function_def,method_name="main",class_name=list(cls_info)[-1],method="declaration_initialization")
                            
                # We need to then see if these variables have any dependencies of the global instance, and replace them if necessary 
                identify_replace_all(main_function_def.body,cls_info) # THis function allows us to identify and replace them recursivly. 
                idx = len(main_function_def.body) 
                
                # 3. Now we need to get the read template for these declared variables within the main function perhaps add a dummy only in the case if the input is present
                if os.path.exists(os.path.join(self.benchmark_dir,self.subroutine_name,'dummy.bin')):
                    read_dummy_ast = self.prepare_read_code_for_main_template(self.extractor.var_dummy[self.subroutine_name],assign_nodes) # THis will create the read_dummy function in Python AST
                    read_dummy_ast_call_stmt = self.create_call_statements(read_dummy_ast)

                    if read_dummy_ast_call_stmt is None:
                        raise ValueError(f'Read_ast_call_stmt is None')
                    
                    call_stmts.append(read_dummy_ast_call_stmt) # This will keep in the current order the list of call statements
                    function_stmts.append(read_dummy_ast)
                        
                # Need to add a timer for this subroutine to measure the time elapsed during the execution and since get_timer sends an AST tree for the 
                # the decorator method present in template.yaml(@timer)
                timer_tree = self.get_timer()
                if timer_tree is None:
                    raise ValueError("Timer tree(@timer) is None")
                
                function_stmts.append(timer_tree)

                # 4. Now we transform/translate the function definition of the subroutine    
                subroutine_tree = self.extractor.subroutines[self.subroutine_name]
                _,_,module_stack = self.f2np.recursive_ast(subroutine_tree) # THis will give out a list containing all the transformations done to the execution part
                # with respect to the hierarchy and nested loops, all of this being present inside the function parent body.
                if len(module_stack) == 1:
                    function_def = module_stack[0] # The list should only have one elements which should correspond to the function ast definintion itself.
                else:
                    raise ValueError(f'The length of module stack is greater than 1:{len(module_stack)}')
                # After this, we need to correct the function, which just means since we translated the from the subroutine tree the subrotuine itself
                # the arguments might have elements that depend on the global attribute or other class eleemnts, we perhaps need to add return stmts 
                # inside thus this requires some changes
                self.correct_function(function_def,cls_info,timer_tree=timer_tree)
                function_def_call_stmt = self.create_call_statements(function_def)
                if function_def_call_stmt is None:
                    raise ValueError(f'Function defintions call statement is None')
                
                call_stmts.append(function_def_call_stmt) # CREATE The call statement
                
                # Now we only need to ensure that the elements inside the function body which might depend on the global or other class attributes gets
                # replaced.
                identify_replace_all(function_def.body,cls_info)
                # print(ast.unparse(ast.fix_missing_locations(function_def)))
                function_stmts.append(function_def)
                # 5. Now we create the test function to test out the subroutine
                # Try to first find if the subroutines still has the benchmark

                if os.path.exists(os.path.join(self.benchmark_dir,self.subroutine_name,'output.bin')):
                    test_subroutine_function = self.create_test_function(cls_info) # This will create the TEST function to test the output of Python to that of 
                    # the FORTRAN ouptut
                    if test_subroutine_function is None:
                        raise ValueError(f'TEST subroutine {self.subroutine_name} is None')
                    
                    test_subroutine_function_call_stmt = self.create_call_statements(test_subroutine_function)
                    if test_subroutine_function_call_stmt is None:
                        raise ValueError(f'Test functions call statement is None')

                    call_stmts.append(test_subroutine_function_call_stmt)
                    function_stmts.append(test_subroutine_function)
                
                # 6. Now we can add the call onto the main function
                for call_stmt in call_stmts:
                    if isinstance(call_stmt,ast.AST):
                        self.insert_at(idx,call_stmt,main_function_def,"main")
                        idx+= 1
                    else:
                        main_function_def.body.extend(call_stmt)
                        idx += len(call_stmt)
                
                # 7. Now we add the functions created onto the main file python AST
                for functions in function_stmts:
                    self.insert_at(None,functions,out_main_template)
            else:
                main_class_template = self.out_module_python()

                for node in ast.iter_child_nodes(main_class_template):
                    if isinstance(node, ast.ClassDef):
                        node.body = [
                            item for item in node.body
                            if not (isinstance(item, ast.FunctionDef) and item.name == "declaration_initialization")
                        ]
                
                main_class_name = ast_walk(main_class_template,ast.ClassDef)
                class_def = next(iter(main_class_name))
                l = [self.subroutine_name[0].upper(), self.subroutine_name[1:]]
                class_def.name = "".join(l)

                # Use the instance_nodes to create the assign_node to be placed inside the __init__ as an arg and sent as argument
                out_module_class_def = list(ast_walk(out_module,ast.ClassDef))
                instance_nodes = self.create_instances(out_module_class_def,True)
                
                init_function_def = next(iter(ast_walk(class_def, ast.FunctionDef)))
                # Adding the object composition inside the main class template
                for instance_node in instance_nodes:
                    self.add_instance(None,instance_node,cls_info,init_function_def,method_name = ["declaration_initialization"])
                # Adding the attributes 
                self.insert_all_assign_nodes(assign_nodes,main_class_template,method_name='__init__',class_name=list(cls_info)[-1],method="declaration_initialization")
                
                # Now we transform each subroutine onto Python AST
                all_child_subroutines = []
                for subroutines in self.isolator.child_subroutine_call[self.subroutine_name]:
                    _,_,child_subroutine_ast = self.f2np.recursive_ast(subroutines)
                    if len(child_subroutine_ast) > 1:
                        raise ValueError(f'The length of module stack for children AST is greater than 1:{len(child_subroutine_ast)}')
                    all_child_subroutines.append(child_subroutine_ast[-1])

                # Now we transform the parent_subroutine
                subroutine_tree = self.extractor.subroutines[self.subroutine_name]

                _,_,parent_subroutine_ast = self.f2np.recursive_ast(subroutine_tree)
                if len(parent_subroutine_ast) > 1:
                    raise ValueError(f'The length of module stack for Parent AST is greater than 1:{len(child_subroutine_ast)}')
                else:
                    parent_subroutine_ast = parent_subroutine_ast[-1]
                    # all_subroutines.append(parent_subroutine_ast)
                
                # Now we add them inside the class as methods, since we still have the all the 
                class_def.body.extend(all_child_subroutines + [parent_subroutine_ast])

                # We will now create the cls information on this class defintion which has a composition type from another class which is why we are
                # sending the instance_nodes of the Global modules of the parent class 
                new_cls_info, _ , _ = self.create_cls_info(class_def,instance_nodes,self_mode=True)
                
                # Now we correct the read_ast created before
                read_dummy_ast = self.prepare_read_code_for_main_template(self.extractor.var_dummy[self.subroutine_name],assign_nodes)
                self.correct_function(read_dummy_ast,new_cls_info)
                identify_replace_all(read_dummy_ast.body,new_cls_info)
                class_def.body.extend([read_dummy_ast])
                
                # METHODS to call in the main fucntion : 
                methods_to_call = ["read_dummy", self.subroutine_name]
                new_cls_info = update_dict( # we update the parent class to keep in memory the information about the class is used internally
                    primary_dict=new_cls_info,
                    secondary_dict=cls_info,
                )

                if os.path.exists(os.path.join(self.benchmark_dir,self.subroutine_name,'output.bin')):
                    test_subroutine_function = self.create_test_function(new_cls_info) # This will create the TEST function to test the output of Python to that of 
                    self.correct_function(test_subroutine_function,new_cls_info)
                    # the FORTRAN ouptut
                    if test_subroutine_function is None:
                        raise ValueError(f'TEST subroutine {self.subroutine_name} is None')

                    class_def.body.extend([test_subroutine_function])
                    methods_to_call.append(f"test_{self.subroutine_name}")

                for child_idx, child_subroutine_key in enumerate(self.extractor.call_within_sub[self.subroutine_name]):
                    self.correct_function(all_child_subroutines[child_idx],new_cls_info,None,child_subroutine_key,parent_mode=parent_mode)
                    identify_replace_all(all_child_subroutines[child_idx].body,new_cls_info)
                
                #Now we correct the parent
                self.correct_function(parent_subroutine_ast,new_cls_info,None,self.subroutine_name)
                identify_replace_all(parent_subroutine_ast.body,new_cls_info)

                # We need to ensure thta the variables intialized inside the class doesn't depend on the variables from the composed class, if so we need to modify
                init_func = None
                for func in ast_walk(class_def, ast.FunctionDef):
                    if func.name == '__init__':
                        init_func = func

                identify_replace_all(init_func.body,new_cls_info)

                # NOw we insert the class node inside the out_main_template
                self.insert_at(idx = None,ast_node = class_def,python_template = out_main_template) 
                final_cls_info, _ , final_instance_nodes = self.create_cls_info(class_def)
                
                for instances in final_instance_nodes:
                    self.add_instance(len(main_function_def.body), instances, final_cls_info, main_function_def,methods_to_call)
                
            # print(ast.unparse(ast.fix_missing_locations(out_main_template)))
            return ast.fix_missing_locations(out_main_template)
        
        except Exception:
            logging.exception(f'Exception error in update_main_python')
            return None
        
    def compile_and_run(self, base_dir,modules_dir):
        target_module_dir_path = os.path.join(base_dir, modules_dir)

        if not os.path.isdir(target_module_dir_path):
            logging.error(f"Target module directory '{target_module_dir_path}' not found.")
            return 1

        for subdir in os.listdir(target_module_dir_path):
            subdir_path = os.path.join(target_module_dir_path, subdir)

            if not os.path.isdir(subdir_path):
                logging.warning(f"Skipping non-directory entry: {subdir}")
                continue

            logging.info(f"Processing module: {subdir}")

            # Python file checks
            main_file = os.path.join(subdir_path, 'main.py')
            global_module_file = os.path.join(subdir_path, 'module_global.py')

            missing_files = []
            if not os.path.exists(main_file):
                missing_files.append('main.py')
            if not os.path.exists(global_module_file):
                missing_files.append('module_global.py')

            if missing_files:
                logging.warning(f"Missing files in '{subdir}': {', '.join(missing_files)}")
                logging.info(f"Skipping '{subdir}' due to missing Python files.\n")
                continue
            else:
                logging.info(f"Required Python files found in '{subdir}'.")

            # Binary file checks
            benchmark_subdir = os.path.join(self.benchmark_dir, subdir)
            dummy_bin = os.path.join(benchmark_subdir, "dummy.bin")
            global_bin = os.path.join(benchmark_subdir, "global.bin")
            output_bin = os.path.join(benchmark_subdir, "output.bin")

            bin_missing = []
            for bin_file in [dummy_bin, global_bin, output_bin]:
                if not os.path.exists(bin_file):
                    bin_missing.append(os.path.basename(bin_file))

            if bin_missing:
                logging.warning(f"Missing binary files for '{subdir}': {', '.join(bin_missing)}")
                logging.info(f"Skipping '{subdir}' due to missing binaries.\n")
                continue

            logging.info(f"All binary files found for '{subdir}'. Running unit tests...")

            try:
                result = subprocess.run(['python3', main_file], check=True, capture_output=True, text=True)
                logging.info(f"Execution output for '{subdir}':\n{result.stdout}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Error running main.py for '{subdir}': {e.stderr}")
                continue

        