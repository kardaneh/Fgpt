from fparser.two import Fortran2003 as F23
from fparser.two.utils import walk
import logging
from typing import Dict, List,Literal,Any,Generator
import yaml
import os
import ast

from isolator import Isolator
from processor import Processor
from extractor import Extractor

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
        self.benchmark_dir = benchmark_dir      # THe benchmark directory
        self.subroutine_name = None             # Subroutine that will be isolated 
        self.ignore_case = ignore_case          # List of string of variables or functions names that are to be ignored
        self.isolator = isolator                # An instance of isolator class just used in the method retrieve_variable to retrieve variable order 
        self.extractor = extractor              # An instance of extractor class 
        self.cls_mode = None                    # Defines if we should create a class global module or not
        self.config_path = config_path          # Path to the template.yaml file 
        self.for_loop = False                   # If we want to create with either using using a for loop for the reading binary files 
        self.global_state = False               # Allows to define if the given code template is for global or not

    @staticmethod
    def load_code_templates(config_path:str) -> Dict:
        """
        Method used to read and retrieve all the templates available within a YAML file.  
        
        params
        ------
        - config_path(str): Path towards the template file 

        returns
        -------
        template(Dict) 

        """
        with open(config_path, 'r') as file:
            templates = yaml.safe_load(file)

        return templates

    def out_module_python(self) -> ast.Module:
        """
        In charge of retrieving the python global module template on either the simple python type or in class format based on the self.cls_mode

        returns
        -------
        code(ast.Module): of the global code template
        """
        templates = Transformer.load_code_templates(self.config_path)

        if self.cls_mode:
            code = templates["Python_templates"]["Python_global_class_template"]["template"]
        else:
            code = templates["Python_templates"]["Python_global_normal_template"]["template"] 

        if code is None:
            logging.error(f"The code template wasn't retreived")
            return None
        return Transformer.python_parser(code)

    def out_main_python(self) -> ast.Module:
        """
        In charge of retrieving the python main module template. 

        returns
        -------
        code(ast.Module): of the main code template
        """
        templates = Transformer.load_code_templates(self.config_path)
        code = templates["Python_templates"]["Python_main_normal_template"]["template"]
        return Transformer.python_parser(code)

    def retreive_variable_order(self) -> None:
        """
        Retrieves the order in which variables are accessed within the binary files.

        This order is significant because it reflects how the variables are stored in the binary format.
        The retrieved order is stored in the `variable_order` attribute.
        """
        self.variable_order = []
        
        for read_dec in [self.isolator.processor_sp.reads_in_decleration_routine,self.isolator.processor_sp.reads_in_read_routine]:
            read_stmt = walk(read_dec,F23.Input_Item_List)
            for item in read_stmt:
                self.variable_order.append(item.children[0].string)

    def convert_SPECIFICATION_PART(self,dec_global:Dict,fix_loc:bool=False, cls_mode:bool=False) -> List:
        """
        Method in charge of transforming FORTRAN specification statement into Python assignement statement

        params
        ------  
        - dec_global(Dict): Dictionnary containing all the global declarations statement for each subroutine  
        - fix_loc(bool):If True, allows the use of ast.unparse upon these assign nodes. Useful when this method is used independently.
        - cls_mode(bool): If True, generates assignments in the context of a class.

        returns
        -------
        ast_nodes(List) A list of AST nodes representing assignment statements for all variables.
        """
        ast_nodes = []

        kind_map = {
            'REAL': 'np.float64',  
            'INTEGER': 'np.int32',
            'LOGICAL': 'np.int32'
        }

        target = None
        # Verify is the dec_global is an instance of Dict of Dict 
        assert isinstance(dec_global[self.subroutine_name], Dict), "dec_global should be a Dict of Dict"

        for var_name, declarations in dec_global[self.subroutine_name].items():
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
                 
            for nodes in walk(declarations, F23.Type_Declaration_Stmt):
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
                                num_var = float(value.string) if intrinsic_type_spec[0].children[0] == "REAL" else int(value.string)
                                assign = ast.Assign(
                                            targets=[target],
                                            value=ast.Call( func=ast.Attribute(value=ast.Name(id=idx, ctx=ast.Load()), attr=attr, ctx=ast.Load()),
                                                        args=[ast.Constant(value=num_var)],
                                                        keywords = []
                                                    )
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
                                 
                                if len(value.children) > 2:
                                    num1,_, num2 = value.children
                                    value_ = ast.BinOp(
                                        left=ast.Constant(value=float(num1.string)),
                                        op=ast.Mult(),
                                        right=ast.Constant(value=float(num2.string))
                                    )
                                else:
                                    if attr == "int32":
                                        value_ = ast.Constant(value=int(value.children[0]))
                                    else:
                                        value_ = ast.Constant(value=float(value.children[0]))

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

                        np_call = ast.Call(
                            func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='empty', ctx=ast.Load()),
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
                                if value.split(".")[1] == "TRUE":
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
                                assign = ast.Assign(
                                    targets=[target],
                                    value=ast.Call( func=ast.Attribute(value=ast.Name(id=idx, ctx=ast.Load()), attr=attr, ctx=ast.Load()),
                                                    args=[ast.Constant(value=value.string)],
                                                    keywords = []
                                                )
                                )
                                
                            ast_nodes.append(assign)
        
                        else:
                                     
                            if value is None: # Example cases : INTEGER :: ier
                                assign = ast.Assign(
                                        targets=[target],
                                        value=ast.Call( func=ast.Attribute(value=ast.Name(id=idx, ctx=ast.Load()), attr=attr, ctx=ast.Load()),
                                                        args=[ast.Constant(value=0)],
                                                        keywords = []
                                                    )
                                        )
                            else:
                                # Verify that the value is a string
                                num_val = int(value.string) if intrinsic_type_spec.children[0] == "INTERGER" else float(value.string)
                                assign = ast.Assign(
                                        targets=[target],
                                        value=ast.Call( func=ast.Attribute(value=ast.Name(id=idx, ctx=ast.Load()), attr=attr, ctx=ast.Load()),
                                                        args=[ast.Constant(value=num_val)],
                                                        keywords = []
                                                    )
                                        )
                                
                            ast_nodes.append(assign)
            if fix_loc:
                for node in ast_nodes:
                    self.set_missing_locations(node)

        return ast_nodes
        
    # We can remove this and only keep the ast_nodes and then once we have added the nodes onto the primary node(code template) we can iterate through them to 
    # fix the missing lineno, col_offset using the ast.fix_missing_locations
    # This is primarily used to show the corrected assign statements when we unparse them
    def set_missing_locations(self, node:ast.AST, lineno=1, col_offset=0) -> ast.AST:
        """
        Adds line number and column offset metadata to an AST node. This method is used within `convert_SPECIFICATION_PART` to correct nodes before passing them to the `unparse` method.  
        It adds the `lineno` and `col_offset` metadata required by the AST.  
        Typically, this correction is handled using `ast.fix_missing_locations`, but this method provides more controlled metadata assignment.

        params
        ------
        - node: The node to correct 
        - lineno(int): By default to 1 Defines the starting row of a function or a variable, uses the 1-based index(1...10)
        - col_offset(int) By default to 0 Defines the staring column of a function or a variables, uses the 0-based index(0...9)

        returns
        -------
        node(ast.AST): The corrected AST node with added location metadata. 

        """
        if not hasattr(node, 'lineno'):
            node.lineno = lineno
        if not hasattr(node, 'col_offset'):
            node.col_offset = col_offset
    
        for n in ast.iter_child_nodes(node):
            self.set_missing_locations(n, lineno, col_offset)  # Recursive and set for all children iteratively
    
        return node       

    def pre_init_variables(self,code_template:ast.Module):
        """
        Extracts variables that are pre-declared and initialized within the given code template. This method identifies all variables that have both been declared and initialized in the input code template.
        It stores their names in the `pre_init` attribute as a list.
        
        params
        ------
        - code_template(ast.Module): Code template upon which will be added the variables or other elements
        """
        
        self.pre_init = []
        class_exist = any(self.ast_walk(code_template,ast.ClassDef))
        if class_exist:
            functions_spec = self.ast_walk(code_template,ast.FunctionDef)
            for functions in functions_spec:
                if functions.name == "__init__":
                    assign_stmt = self.ast_walk(functions,ast.Assign)
                    for assign_ in assign_stmt:
                        if isinstance(assign_.targets[0],ast.Name):
                            self.pre_init.append(assign_.targets[0].id)
                        elif isinstance(assign_.targets[0],ast.Attribute):
                            self.pre_init.append(assign_.targets[0].attr)
                        
        else:
            nodes = self.ast_walk(code_template,ast.Assign)
            for node in nodes:
                self.pre_init.append(node.targets[0].id)

    """
    # temp = [var for var in variable_order for arg in args_ if re.search(var,arg,re.IGNORECASE)]
    # if len(temp) > 0:
    # dependant_variables[entity_dec_name.string] = temp
    """
    def search_dependant_variables(self) -> None:
        """
        Analyzes dependencies between arrays and scalars using the FORTRAN AST. This method identifies which variables (dependents) rely on others (dependees) by traversing the FORTRAN AST.
        It creates an attribute `dependant_variables`, which is a dictionary where each key is a dependent variable and the corresponding value is a list of its dependees.
 
        """
        
        self.dependant_variables = {} # THe variable that is the dependant of the other which has the key as the dependant and the values 
        # the different dependees 
        combined_stmt = None
        for key in self.extractor.dec_global[self.subroutine_name].keys():
            dependees = []
            declarations = self.extractor.dec_global[self.subroutine_name][key]
            alloc_spec = any([alloc for alloc in walk(declarations, F23.Attr_Spec) if alloc.string == "ALLOCATABLE"])
            if len(declarations) == 2 and alloc_spec:
                combined_stmt = Processor().combine_allocate_declaration(declarations)
            
            if combined_stmt:
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
                        dec = self.extractor.dec_global[self.subroutine_name].get(lb,None)
                        if dec is None and lb in self.pre_init: # This means that the variables is present in the pre init variables
                            continue 
                        elif dec is not None and lb not in self.pre_init:
                            # we verify the initalization of these shapes
                            for elements in dec:
                                entity_decl_list = walk(elements,F23.Entity_Decl_List)[0]
                                for entity_dec in entity_decl_list.children:
                                    _, _,_, initialization = entity_dec.children
                                    if initialization is None:
                                        dependees.append(lb)

                    if ub is not None:
                        dec = self.extractor.dec_global[self.subroutine_name].get(ub, None)
                        if dec is None and ub in self.pre_init: # This means that the variables is present in the pre init variables
                            continue 
                        elif dec is not None and ub not in self.pre_init:
                            # we verify the initalization of these shapes
                            for elements in dec:
                                entity_decl_list = walk(elements,F23.Entity_Decl_List)[0]
                                for entity_dec in entity_decl_list.children:
                                    _, _,_, initialization = entity_dec.children
                                    if initialization is None:
                                        dependees.append(ub)
                            
                if len(dependees) > 0:
                    self.dependant_variables[entity_dec_name.string] = dependees
    
    def insert_at(self, idx:int, ast_node:ast.AST, python_template:ast.Module,method_name:str = None, place_after:ast.AST = None) -> None:
        """
        Inserts an AST node (such as Import, FunctionDef, Assign, or For) into a given Python AST module at the specified location based on context (e.g., inside a class, method, or at the global level).

        This function supports smart placement of:
        - Import and ImportFrom nodes at the top of the module.
        - FunctionDef nodes at a given index or after the last function/import.
        - Assign nodes inside a specified method (or __init__ by default), or at the module level.
        - (Placeholder) For loops, based on the method name and a reference AST node.
        - If the module contains a class, it assumes that insertions are within the first class's body.
        - When inserting Assign nodes and no method name is provided, it defaults to inserting inside `__init__`.
        - If `idx` is not provided, insertion happens after the last known relevant node of the same type.

        params:
        -------
        - idx(int): Position at which to place the ast node
        - ast_node(ast.AST): The AST node to add onto the python template
        - python_template(ast.Module): The python AST template within which the ast nodes are to be placed
        - method_name(str): Name of a function/method inside which a variable, or for loop or statement needs to be placed
        - place_after(ast.AST): Corresponds to the AST node after which the ast_node needs to be placed, primarily used for `for loops`
        """
        class_exists = any(isinstance(n, ast.ClassDef) for n in python_template.body) # any(ast_walk(python_template,ast.ClassDef))
        if class_exists:
            if isinstance(ast_node, (ast.Import, ast.ImportFrom)):
                python_template.body.insert(0,ast_node)
            
            # If the given ast_node is that of a function defintion
            elif isinstance(ast_node,ast.FunctionDef):
                if idx is not None:
                    python_template.body[0].body.insert(idx,ast_node)
                else:
                    # Else we just need to simply append it
                    function_pos = [pos for pos, node in enumerate(python_template.body[0].body) if isinstance(node,ast.FunctionDef)]
                    pos = function_pos[-1]
                    python_template.body[0].body.insert(pos + 1, ast_node)
                    
            elif isinstance(ast_node,ast.Assign): # If the given ast_node is that of a variable assignement statement
                functions_spec = self.ast_walk(python_template,ast.FunctionDef)
                for functions in functions_spec:
                    if not method_name:
                        logging.info(f"Since no method name is given, defaulting to the __init__ method")
                        if functions.name == "__init__":
                            assign_statement = [i for i, stmt in enumerate(ast.iter_child_nodes(functions)) if isinstance(stmt, ast.Assign)]
                            insert_pos = assign_statement[-1] + 1 if assign_statement else len(functions.body)
                            if idx is not None:
                                functions.body.insert(idx,ast_node)
                            else:
                                logging.info(f'Since no index is given, WILL BE USING previous known ast Assign position')
                                functions.body.insert(insert_pos, ast_node)
                    else:
                        logging.info(f"Since argument method_name is: {method_name}, placing the assign statement inside of the method")
                        if functions.name == method_name:
                            assign_statement = [pos for pos, assign in enumerate(ast.iter_child_nodes(functions)) if isinstance(assign, ast.Assign)]
                            insert_pos = assign_statement[-1] + 1 if assign_statement else len(functions.body)
                            if idx:
                                functions.body.insert(idx,ast_node)
                            else:
                                logging.info(f'Since no index is given, WILL BE USING previous known ast Assign position')
                                functions.body.insert(insert_pos, ast_node)
            elif isinstance(ast_node,ast.For):
                # Here we need to be given absolutely the method name within which the for loop needs to be placed upon
                # And the place_after argument which is the type of AST after which the foor loop might be needed to be added
                pass 
                        
        else:
            if isinstance(ast_node, (ast.Import, ast.ImportFrom)):
                python_template.body.insert(0,ast_node)
                
            elif isinstance(ast_node, ast.FunctionDef):
                if idx is not None:
                    # Gather positions of all import statements
                    import_positions = [ pos for pos, stmt in enumerate(ast.iter_child_nodes(python_template))
                        if isinstance(stmt, (ast.Import, ast.ImportFrom))]
            
                    if import_positions:
                        last_import_pos = import_positions[-1]
                        if idx <= last_import_pos:
                            logging.warning( f'The given idx ({idx}) is before or within the import statements. ' f'Correcting and placing it after the last import at position {last_import_pos}.'
                            )
                            python_template.body.insert(last_import_pos + 1, ast_node)
                        else:
                            python_template.body.insert(idx, ast_node)
                    else:
                        # No import statements found, safe to insert at the given idx
                        python_template.body.insert(idx, ast_node)
                else:
                    # No idx provided; insert after the last function definition if present
                    function_positions = [pos for pos, stmt in enumerate(ast.iter_child_nodes(python_template))
                        if isinstance(stmt, ast.FunctionDef)]
            
                    if function_positions:
                        last_func_pos = function_positions[-1]
                        python_template.body.insert(last_func_pos + 1, ast_node)
                    else:
                        # No functions yet; append at the end
                        python_template.body.append(ast_node)
            elif isinstance(ast_node, ast.Assign):
                if method_name:
                    # print([func for func in self.ast_walk(python_template,ast.FunctionDef) if func.name == method_name])
                    target_func = [func for func in self.ast_walk(python_template,ast.FunctionDef) if func.name == method_name][0]
                    if target_func:
                        assign_positions = [i for i, stmt in enumerate(target_func.body)
                            if isinstance(stmt, ast.Assign)]
                        # Insert pos depends on if the presence of other assign statement if not we insert it after the len(body) + 1
                        insert_pos = assign_positions[-1] + 1 if assign_positions else len(target_func.body)

                        if idx is not None:
                            target_func.body.insert(idx, ast_node)
                        else:
                            logging.info(f'Inserting after last assign at position {insert_pos}')
                            target_func.body.insert(insert_pos + 1, ast_node)
                else:
                    assign_positions = [i for i, stmt in enumerate(python_template.body)
                        if isinstance(stmt, ast.Assign)]
                    insert_pos = assign_positions[-1] + 1 if assign_positions else len(python_template.body)

                    if idx is not None:
                        python_template.body.insert(idx, ast_node)
                    else:
                        logging.info(f'Inserting after last global assign at position {insert_pos}')
                        python_template.body.insert(insert_pos, ast_node)
            elif isinstance(ast_node,ast.For):
                pass 
        

    def separate_scalar(self) -> None:
        """
        Separates `dec_global` entries into initialized and uninitialized variables, and identifies scalar or logical variables. This method distinguishes between variables that are only declared and those that require initialization.
        Scalar and logical (immutable) variables are separated and stored in `self.scalar`.
        """
        self.scalar = []
        if self.global_state:
            for var in self.variable_order:
                dec_statement = self.extractor.dec_global[self.subroutine_name][var]

                init_spec = any(walk(dec_statement, F23.Initialization))
                alloc_spec = any([alloc for alloc in walk(dec_statement, F23.Attr_Spec) if alloc.string == "ALLOCATABLE"])
                if not init_spec and not alloc_spec:
                    self.scalar.append(var)
        else:
            for dec_statement in self.extractor.var_dummy[self.subroutine_name]:
                var = walk(dec_statement,F23.Entity_Decl)[0].string
                init_spec = any(walk(dec_statement, F23.Initialization))
                alloc_spec = any([alloc for alloc in walk(dec_statement, F23.Dimension_Attr_Spec)])
                if not init_spec and not alloc_spec:
                    self.scalar.append(var)
        
    def retrieve_table_info(self) -> Dict:
        tables = {}
        args_ = None
        combined_stmt = None
        logging.info(f'Combining allocate statement together to retrieve dimension shape inside the retrieve_table_info method')
        for var in self.variable_order:
            dec_statement = self.extractor.dec_global[self.subroutine_name][var]
            alloc_spec = any([alloc for alloc in walk(dec_statement, F23.Attr_Spec) if alloc.string == "ALLOCATABLE"])
            if len(dec_statement) == 2 and alloc_spec:
                combined_stmt = Processor().combine_allocate_declaration(dec_statement)
            
                if combined_stmt:
                    dimensions_spec_list = walk(walk(combined_stmt,F23.Dimension_Attr_Spec),F23.Explicit_Shape_Spec_List) 
                    children = dimensions_spec_list[0].children
                    args_ = [var.string for var in children if var is not None]
        
                tables[var] = args_
        return tables
    
    def prepare_read_code_global_template(self, assign_nodes:List) -> ast.Module:
        """
        Generates and populates the read code template for reading a Fortran binary file. This method builds the structure for reading a Fortran binary file line by line, supporting both standalone Python scripts and class-based approaches.
        It inserts assignment nodes into the template to define how each variable should be read.

        params
        ------
        - assign_nodes(List): List of assign(ast.Assign) type nodes for each variables

        returns
        -------
        read_ast(ast.Module): The modified read template filled with the assignement nodes. 
        """
        try:
            template_str = Transformer.load_code_templates(self.config_path)["Python_templates"]["Python_read_template"]["template"]
            read_code_template =template_str.format(
                benchmark_dir=self.benchmark_dir,
                subroutine_name=self.subroutine_name
            )
            # print(read_code_template)
            read_ast = self.python_parser(read_code_template)
        except Exception as e:
            print(f"Error from prepare_read_code_global_template method: {e}")
        
        # print(ast.dump(read_ast,indent=4))
        var_list = self.read_file_ast(assign_nodes=assign_nodes)
        for variable in var_list:
            read_ast.body.append(variable)

        # read_ast = ast.fix_missing_locations(read_ast)
        return read_ast
    
    def read_file_ast(self,assign_nodes:List) -> List:
        """
        Generates the code lines used to read variables from a Fortran binary file. This method constructs the individual lines of code required to read each variable from the binary file.
        It returns a list of such lines, which may look like the following example: `ffile.read_ints(np.int32)[0]`

        params
        ------
        - assign_nodes(List): List of assign(ast.Assign) type nodes for each variables

        returns
        -------
        var_list(List): List of all the variable reading file 
        """
        var = None
        var_name = None
        target = None
        var_list = []
        for variable in self.scalar:
            for assign_node in assign_nodes:
                if isinstance(assign_node.targets[0], ast.Name):
                    var_name = assign_node.targets[0].id
                    target = ast.Name(id=var_name,ctx = ast.Store())
                else:
                    var_name = assign_node.targets[0].attr
                    target = ast.Attribute(
                        value = ast.Name(id='self',ctx=ast.Load()),
                        attr = var_name,
                        ctx = ast.Load()
                    )
                if var_name == variable:
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

    def prepare_read_code_for_global_template(self,assign_nodes:List) -> ast.Module:
        """
        Initializes scalar variables by reading them line by line from a Fortran binary file. This method assumes that the necessary `for` loops for reading array data are already present in the `read_code_template`.
        It focuses on scalar variables, which are read individually and inserted into their appropriate positions within the code template.
        During this process, the names of the scalar variables are also added to the list of variables handled inside the `for` loop.

        params
        ------
        - assign_nodes(List): List of assign(ast.Assign) type nodes for each variables

        returns
        -------
        read_ast(ast.Module): Python AST of the read template with the newly added elements 
        """

        try:
            template_name = "Python_read_for_loop_template" if not self.cls_mode else "Python_read_for_loop_class_template"
            for_template_str = Transformer.load_code_templates(self.config_path)["Python_templates"][template_name]["template"]
            
            read_code_template =for_template_str.format(
                benchmark_dir=self.benchmark_dir,
                subroutine_name=self.subroutine_name,
            )
            # print(read_code_template)
        except Exception as e:
            print(f"Error from prepare_read_code_global_template method: {e}")

        try:
            read_ast = self.python_parser(read_code_template)
        except Exception as e:
            print(f"Error from prepare_read_code_global_template method: {e}")

        var_list = self.read_file_ast(assign_nodes)
        # In order to use for loop within the python script we will first read the scalars line by line and for the arrays we will read it using a for loop
        var_pos = [i for i,element in enumerate(ast.iter_child_nodes(read_ast)) if isinstance(element,ast.For)][0] 
        
        for variable in var_list:
            read_ast.body.insert(var_pos, variable)
            var_pos+=1               
        # The arrays are read through a loop instead of reading them line by line and Now fill up the list for the for loop with the read_ast

        # We apply the same prinicple for the class aspect but the primary differences situates within the self and thanks to the getattr and hasattr
        # methods which allows us to retrieve a class attribute and modify it dynamically allowing us to do a proper changement instead of using globals
        for_ast = next(iter(self.ast_walk(read_ast,ast.For)))
        # print(for_ast)
        if for_ast.iter.elts == []:
            difference = [item for item in self.variable_order if item not in self.scalar]
            for_ast.iter.elts = [ast.Constant(var) for var in difference]
        
        # read_ast = ast.fix_missing_locations(read_ast)
        return read_ast
    
    def init_dependant_variables(self,read_ast:ast.Module,assign_nodes:List) -> List: 
        """
        Initializes variables that depend on other attributes which are set only after reading input values from a file.

        This method is designed to be called after certain critical dependencies(such as configuration parameters or dimensions) have been read from an input file. It ensures that dependent variables are not initialized until all 
        required 'dependee' variables are available.

        The initialization sequence is managed based on a dependency mapping(self.dependant_variables) that specifies which variables rely on which.For each dependent variable, the method determines the latest initialization
        point of its dependencies and inserts the initialization code accordingly.

        params
        ------
        - read_ast: AST tree of the read template 
        - assign_nodes(List): List of assign(ast.Assign) type nodes for each variables

        returns
        -------
        read_ast(List): List containing all the Python ast from the read_ast body
        """

        var_name = None
        for key in list(self.dependant_variables.keys()):
            pos = 0 # This will allows us to find after which variable should we place the init of the dependant variables
            # Since each depandant variables might have multiple dependee variables which are init at different locations as such we try to find the
            # furthest/last positin of the variable dependee 

            # Get all assign statements
            assign_stmts = self.ast_walk(read_ast,ast.Assign)
            dependees = self.dependant_variables[key]
            for i, stmt in enumerate(assign_stmts):
                if isinstance(stmt.targets[0], ast.Name):
                    if stmt.targets[0].id in dependees:
                        if i > pos: # THis way we retrieve the maximum dependee position
                            pos = i
                elif isinstance(stmt.targets[0], ast.Attribute):
                    # print(stmt.targets[0])
                    if stmt.targets[0].attr in dependees:
                        if i > pos:
                            pos = i
            position = pos + 1
            for assign_node in assign_nodes:
                if isinstance(assign_node.targets[0], ast.Name):
                    var_name = assign_node.targets[0].id
                elif isinstance(assign_node.targets[0], ast.Attribute):
                    var_name = assign_node.targets[0].attr

                if var_name == key and var_name is not None:
                    read_ast.body.insert(position,assign_node)

        # read_ast = ast.fix_missing_locations(read_ast)
        return read_ast.body

    def convert_global_read_subroutine(self,assign_nodes:List,code_template:ast.Module) -> None:
        """
        Populates the entire code template with all necessary elements for the `module_global` file. This method utilizes previously defined methods to assemble and insert all required components into the code template,
        ensuring that the `module_global` file is fully constructed.

        params
        ------
        - assign_nodes(List): List of assign(ast.Assign) type nodes for each variables
        - code_template(ast.Module): AST tree of the code template that will be modified
        """
        
        # The variable order will be retrieved since we the instance of the isolator class which use the processor and extractor class
        function_def = self.ast_walk(code_template,ast.FunctionDef)
        # print(ast.dump(read_ast,indent=4))
        
        # Retrieved the scalar/Logical variables that will be read 
        self.separate_scalar()
        
        # read_ast = self.prepare_read_code_global_template(assign_nodes)
        if self.for_loop:
            read_ast = self.prepare_read_code_for_global_template(assign_nodes=assign_nodes)
        else:
            read_ast = self.prepare_read_code_global_template(assign_nodes)

        read_ast_list = self.init_dependant_variables(read_ast,assign_nodes)
        
        for functions in function_def:
            if functions.name == "declaration_initialization": # IF we find the declaration intiailization method to read and fill tables
                if self.scalar:
                    tree = ast.parse(f"global {', '.join(self.scalar + list(self.dependant_variables.keys()))}") # self.scalar + list(self.dependant_variables.keys())
                    
                    functions.body.append(tree.body[0])
                    
                # ast.iter_child_nodes(read_ast)
                for elem in read_ast_list:
                    functions.body.append(elem)
            else:
                print(functions.name)

        # What this does it fix the missing location(lineno,end_lineno,col_offset,end_col_offset) based on the parent node
        # https://docs.python.org/3/library/ast.html#ast.fix_missing_locations 
        code_template = ast.fix_missing_locations(code_template)

        return code_template 
    
    def transform_to_python_script(self, assign_nodes) -> ast.Module:
        """
        Transform from AST Fortran to AST python script approach for global module.  

        params
        ------
        - assign_nodes(List): List of assign(ast.Assign) type nodes for each variables

        returns
        -------
        code_tree(ast.Module): The finalized python script in which contains all the elements of the transformation
        """
        code_tree = self.out_module_python()
        # Specification part
        self.insert_all_assign_nodes(assign_nodes=assign_nodes,code_tree=code_tree,method_name=None)
        # This is for the functions part
        functions_spec = self.ast_walk(code_tree,ast.FunctionDef)

        for functions in functions_spec:
            if functions.name == "declaration_initialization":
                code_template = self.convert_global_read_subroutine(assign_nodes=assign_nodes,code_template=code_tree)

        return code_template
    
    def transform_to_class(self, assign_nodes:List) -> ast.Module:
        """
        Transform from AST Fortran to AST python class approach for global module.  

        params
        ------
        - assign_nodes(List): List of assign(ast.Assign) type nodes for each variables

        returns
        -------
        code_tree(ast.Module): The finalized python class in which contains all the elements of the transformation

        """
        class_tree = self.out_module_python()
        functions_spec = self.ast_walk(class_tree,ast.FunctionDef)

        self.separate_scalar()

        for functions in functions_spec:
            if functions.name == "__init__":
                self.insert_all_assign_nodes(assign_nodes,class_tree,method_name = functions.name)
                
            elif functions.name == "declaration_initialization":
                if isinstance(functions.body[0], ast.Pass):
                    functions.body.pop(0)
                
                if self.for_loop:
                    read_ast = self.prepare_read_code_for_global_template(assign_nodes)
                else:
                    read_ast = self.prepare_read_code_global_template(assign_nodes)

                read_ast_list = self.init_dependant_variables(read_ast,assign_nodes)

                # ast.iter_child_nodes(read_ast)
                for elem in read_ast_list:
                    functions.body.append(elem)

        # What this does it fix the missing location(lineno,end_lineno,col_offset,end_col_offset) based on the parent node
        # https://docs.python.org/3/library/ast.html#ast.fix_missing_locations     
        class_tree = ast.fix_missing_locations(class_tree)

        return class_tree

    def insert_all_assign_nodes(self,assign_nodes:List,code_tree:ast.Module,method_name:str) -> None:
        """
        Inserts all the assign nodes inside the given global code template. 

        params
        ------
        - assign_nodes(List): List of assign type AST nodes 
        - code_tree(ast.Module): AST of the code template to insert the nodes onto
        - method_name(str): Name of the method inside which to fill it in

        """
        if self.global_state:
            diff = list(set(
                assign.targets[0].id if isinstance(assign.targets[0], ast.Name) else assign.targets[0].attr
                for assign in assign_nodes) - set(self.variable_order))
            name = None
            # DO it in two steps first the declared and intializd variables and then the variable order
            # The declared and intialized variables 
            if len(diff) != 0:
                for assign_node in assign_nodes:
                    if isinstance(assign_node.targets[0],ast.Name):
                        name = assign_node.targets[0].id 
                    else:
                        name = assign_node.targets[0].attr
                    if name in diff:
                        # self.insert_assign_at(None,assign_node,code_tree,method_name=method_name)
                        self.insert_at(None,assign_node,code_tree,method_name=method_name)
            # Now all the declared and not intialized variables
            # assign_node_names = [assign.targets[0].id if isinstance(assign.targets[0], ast.Name) else assign.targets[0].attr  for assign in assign_nodes]
            for var in self.variable_order:
                for assign_node in assign_nodes:
                    if isinstance(assign_node.targets[0],ast.Name):
                        name = assign_node.targets[0].id 
                    else:
                        name = assign_node.targets[0].attr
                        
                    if var == name and var not in list(self.dependant_variables.keys()):
                        # self.insert_assign_at(None,assign_node,code_tree,method_name=method_name)
                        self.insert_at(None,assign_node,code_tree,method_name=method_name)
        else:
            for assign_node in assign_nodes:
                # self.insert_assign_at(None,assign_node,code_tree,method_name)
                self.insert_at(None,assign_node,code_tree,method_name=method_name)
        # code_tree = ast.fix_missing_locations(code_tree)
        

    def update_global_python(self,subroutine_name:str,cls_mode:bool,for_loop:bool=True) -> ast.Module:
        """
        Update the global code python AST through the different steps as seen within the code
        
        params
        ------
        - subroutine_name(str): Name of the isolated subroutine
        - cls_mode(bool): If the global python code ast should be in class mode or not
        - for_loop(bool): Defines if we want to use for loop within the 

        returns
        -------
        tree(ast.Module): AST tree containing the finalized and updated elements
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
            self.retreive_variable_order()
            code_template = self.out_module_python()
            self.pre_init_variables(code_template)

            # 2. Retrieve all the assignement python ast statements for the global declarations
            assign_nodes = self.convert_SPECIFICATION_PART(self.extractor.dec_global,cls_mode=cls_mode)

            # 3. Search for variables that dependant one another between the variable_order and global declarations
            self.search_dependant_variables()

            # 4. Now we insert the variables and the read and initialization statement within the code template
            if cls_mode:
                tree = self.transform_to_class(assign_nodes=assign_nodes)
            else:
                tree = self.transform_to_python_script(assign_nodes=assign_nodes)
            
            #print(ast.unparse(tree))
            return tree 
        except Exception as e:
            logging.error(f"Error in update_global_python method: {e}")

    def update_main_python(self):
        self.global_state = False 
    
    def transfer_to_pyfile(self, tree:ast.Module, folder_name:str="hyrdol",python_file_type:Literal["module_global","main"] = "module_global") -> None:
        """
        Method to transfer the FINALIZED python ast onto a python file based on it's type either a module_global or main. 
        
        params
        ------
        - tree(ast.Module): The python AST tree
        - folder_name(str): By default the name is set to python_benchmark which defines the directory within which we will have the python per subroutines
        - python_file_type(str): By default the type is module_global, which defines the type of python file either a main or module_global file

        """
        try:
            current_dir = os.getcwd()
            benchmark_path = os.path.join(current_dir, folder_name)
            subroutine_path = os.path.join(benchmark_path, self.subroutine_name)
            file_path = os.path.join(subroutine_path, f"{python_file_type}.py")

            # First create python benchmark directory which will contain the directories of each subroutines dir within which contains the output of the subroutines test
            logging.info("Creating benchmark directory...")
            os.makedirs(benchmark_path, exist_ok=True)

            # Then the subroutine directory within the benchmark
            logging.info("Creating subroutine directory...")
            os.makedirs(subroutine_path, exist_ok=True)

            logging.info(f"Writing Python file: {file_path}")
            with open(file_path, "w") as f:
                f.write(ast.unparse(tree))

            logging.info("File successfully written.")

        except Exception as e:
            logging.error(f"Exception in transfer_to_pyfile: {e}", exc_info=True)

    @staticmethod
    def python_parser(code:str) -> ast.Module:
        """
        Parses a Python code string into an AST module. Attempts to parse the given Python code string into an abstract syntax tree (AST). 
        Logs an info message on success, or an error message if a syntax error occurs.

        params
        ------
        code(str): The Python code to parse.

        returns
        -------
        tree(ast.Module or None): The parsed AST module if successful, otherwise None.
        """
        try:
            tree = ast.parse(code)
            logging.info("INFO: Parsed python template is valid")
            return tree
        except SyntaxError as e:
            logging.error(f'ERROR: Syntax error: {e}')
        return None
        
    def ast_walk(self, node, node_type:Any) -> Generator:
        """
        Recursively walks an AST tree, yielding all nodes or nodes of a specific type.

        params
        ------
        - node: The node to walk through
        - node_type: The type of which to find within the node

        returns
        -------
        Generator
        """
        if node_type is None or isinstance(node, node_type):
            yield node
        for child in ast.iter_child_nodes(node):
            yield from self.ast_walk(child, node_type)
    