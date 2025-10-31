import numpy as np
from fparser.two.utils import walk
from fparser.two import Fortran2003 as F23
from fparser.two import Fortran2008 as F28
from extractor import Extractor
from typing import List, Dict, Optional,Union
import re
import ast
import copy
import itertools
from utils import ast_walk,adjust_loop_variables
from logger import Logger

class F2NP:
    """
    F2NP is a class for converting Fortran code into NumPy-based Python code.

    This class takes Fortran constructs and translates them into equivalent
    Python code using NumPy for numerical operations. It handles various
    Fortran statements, including subroutine calls, type declarations, 
    control statements (if, do), and intrinsic functions.

    Attributes:
        result (list): A list to store the results of the translation.
        indentation_level (int): The current level of indentation for nested
                                 structures.
        npcode (str): The resulting Python code as a string.
        replacements (dict): A mapping of Fortran logical and arithmetic 
                             operators to their Python equivalents.
        intrinsic_replacements (dict): A mapping of Fortran intrinsic 
                                        functions to their NumPy equivalents.

    Methods:
        recursive(block): Recursively processes the Fortran block.
        handle_subroutine_stmt(stmt): Translates Fortran subroutine statements.
        handle_call_stmt(stmt): Translates Fortran subroutine call statements.
        handle_type_declaration_stmt(stmt): Translates type declaration statements.
        simplify_limits(expression): Simplifies loop limits and expressions.
        handle_end_stmt(stmt): Handles end statements for control structures.
        handle_print_stmt(stmt): Translates print statements.
        handle_assignment(stmt): Handles variable assignments.
        handle_do_stmt(stmt): Translates Fortran do loops to Python for loops.
        handle_if_condition(condition): Translates Fortran if conditions.
        handle_part_ref(stmt_str, part_ref): Handles array references.
        handle_intrinsic_function_reference(stmt_str, intrinsic_function_reference): 
            Translates intrinsic function calls.
    """
    def __init__(self,extractor:Optional[Extractor]=None):
        self.result = []
        self.ast_mode = False
        self.extractor = extractor
        self.loop_variables = {} # This is to ensure that each time we loop, the loop variables which are inserted in the order of apparation is properly used
        self.indentation_level = 0
        self.npcode = ""
        self.replacements = {
                r'\bELSE IF\b': 'elif',
                r'\bIF\b': 'if',
                r'\bELSE\b': 'else:',
                r'\.LT\.': '<',
                r'\.LE\.': '<=',
                r'\.GT\.': '>',
                r'\.GE\.': '>=',
                r'\.NE\.': '!=',
                r'\.EQ\.': '==',
                r'\.AND\.': 'and',
                r'\.OR\.': 'or',
                r'\.NOT\.': 'not',
                r'\bTHEN\b': ':'
                }
        self.intrinsic_replacements = {
                r'\bINT\b': 'int',
                r'\bREAL\b': 'float',
                r'\bMIN\b': 'np.minimum', 
                r'\bMAX\b': 'np.maximum', # https://medium.com/@amit25173/understanding-element-wise-maximum-in-numpy-43916b1c2002 but perhaps go with np.fmax since it handles NaN values too. 
                r'\bMAXVAL\b': 'np.max',
                r'\bMINVAL\b': 'np.min',
                r'\bMINLOC\b': 'np.argmin',
                r'\bMAXLOC\b': 'np.argmax',
                r'\bABS\b': 'np.abs',
                r'\bSQRT\b': 'np.sqrt',
                r'\bEXP\b': 'np.exp',
                r'\bLOG\b': 'np.log',
                r'\bSIN\b': 'np.sin',
                r'\bCOS\b': 'np.cos',
                r'\bTAN\b': 'np.tan',
                r'\bASIN\b': 'np.arcsin',
                r'\bACOS\b': 'np.arccos',
                r'\bATAN\b': 'np.arctan',
                r'\bATAN2\b': 'np.arctan2',
                r'\bAINT\b': 'np.trunc',
                r'\bMOD\b': 'np.mod',
                r'\bCEILING\b': 'np.ceil',
                r'\bFLOOR\b': 'np.floor',
                r'\bSUM\b': 'np.sum',
                r'\bPRODUCT\b': 'np.prod',
                r'\bDOT_PRODUCT\b': 'np.dot',
                r'\bMATMUL\b': 'np.matmul',
                r'\bRESHAPE\b': 'np.reshape',
                r'\bALLOCATE\b': 'np.empty',
                r'\bSIZE\b': 'np.size'
            }
        
        self.conditional_ops_map = {
                '>' : ast.Gt(),
                '>=': ast.GtE(),
                '<': ast.Lt(),
                '<=': ast.LtE(),
                '!=': ast.NotEq(),
                '==': ast.Eq(),
                'not': ast.Not(),
                'and': ast.And(),
                'or': ast.Or()
            }

        self.logger = Logger()
        self.logger.show_header("F2NP")

    def recursive(self, block):
        """
        Recursively processes a block of Fortran code, identifying and handling different types 
        of Fortran statements such as subroutines, type declarations, DO loops, IF conditions, 
        and assignments, converting them to equivalent Python code with NumPy.
        
        Parameters:
        block: A Fortran block of code to be analyzed and transformed.
        """
        if hasattr(block, "content"):
            idx = 0
            while idx < len(block.content):
                child = block.content[idx]
                if isinstance(child, F23.Subroutine_Stmt):
                    print('\033[34m' + f"{child}" + '\033[0m')
                    child = self.handle_subroutine_stmt(child)
                    self.npcode += f"{self.indentation_level * '    '}{child}\n"
                    self.indentation_level += 1
                    print('\033[32m' + f"{child}" + '\033[0m\n')
                elif isinstance(child, F23.Type_Declaration_Stmt):
                    print('\033[34m' + f"{child}" + '\033[0m')
                    child = self.handle_type_declaration_stmt(child)
                    print('\033[32m' + f"{child}" + '\033[0m\n')
                    if child is not None:
                        self.npcode += f"{self.indentation_level * '    '}{child}\n"
                    else:
                        del block.content[idx]
                        continue
                elif isinstance(child, F23.Nonlabel_Do_Stmt):
                    print('\033[34m' + f"{child}" + '\033[0m')
                    child = self.handle_do_stmt(child)
                    print('\033[32m' + f"{child}" + '\033[0m\n')
                    self.npcode += f"{self.indentation_level * '    '}{child}\n"
                    self.indentation_level += 1
                elif isinstance(child, F23.If_Then_Stmt):
                    print('\033[34m' + f"{child}" + '\033[0m')
                    if walk(child, F23.Part_Ref):
                        child = self.handle_assignment(child)
                    else:
                        child = child.tostr()
                    child = self.handle_if_condition(child)
                    print('\033[32m' + f"{child}" + '\033[0m\n')
                    self.npcode += f"{self.indentation_level * '    '}{child}\n"
                    self.indentation_level += 1
                elif isinstance(child, (F23.Else_If_Stmt, F23.Else_Stmt)):
                    print('\033[34m' + f"{child}" + '\033[0m')
                    self.indentation_level -= 1
                    if isinstance(child, F23.Else_If_Stmt) and walk(child, F23.Part_Ref):
                        child = self.handle_assignment(child)
                    else:
                        child = child.tostr()
                    child = self.handle_if_condition(child)
                    print('\033[32m' + f"{child}" + '\033[0m\n')
                    self.npcode += f"{self.indentation_level * '    '}{child}\n"
                    self.indentation_level += 1
                elif isinstance(child, (F23.End_If_Stmt, F23.End_Do_Stmt, F23.End_Subroutine_Stmt)):
                    print('\033[34m' + f"{child}" + '\033[0m')
                    self.indentation_level -= 1
                    del block.content[idx]
                    continue
                elif isinstance(child, F23.Print_Stmt):
                    print('\033[34m' + f"{child}" + '\033[0m')
                    child = self.handle_print_stmt(child)
                    print('\033[32m' + f"{child}" + '\033[0m\n')
                    self.npcode += f"{self.indentation_level * '    '}{child}\n"
                elif isinstance(child, F23.Assignment_Stmt):
                    print('\033[34m' + f"{child}" + '\033[0m')
                    child = self.handle_assignment(child)
                    print('\033[32m' + f"{child}" + '\033[0m\n')
                    self.npcode += f"{self.indentation_level * '    '}{child}\n"
                else:
                    self.recursive(child)
                idx += 1
    
    def append_to_current_parent(self, stmt, control_stack:List):
        try:
            if not control_stack:
                return 
            current_parent = None
            if control_stack and len(control_stack) > 0:
                current_parent = control_stack[-1]
            else:
                current_parent = control_stack
            if current_parent is not None and (hasattr(current_parent, 'body') or isinstance(current_parent, list)):
                if hasattr(current_parent, 'body') and isinstance(current_parent.body, list):
                    current_parent.body.append(stmt)
                    
                elif isinstance(current_parent, list): # This for the case of ELSE statemnt in which the control_stack will contain the 
                    # orelse list 
                    current_parent.append(stmt)
                
                else:
                    raise RuntimeError("Expected parent with 'body' attribute for nested control block")
                
            elif current_parent is not None and isinstance(current_parent, dict):
                    if_chain = current_parent['if_chain']
                    if hasattr(if_chain, 'body') and isinstance(if_chain.body, list):
                        if_chain.body.append(stmt)
            else:
                control_stack.append(stmt)
        except Exception:
            raise

    def recursive_ast(self, block, ast_mode:bool=True, control_stack:List=None, counters:Dict=None, module_stack:List= None):
        """
        Recursively traverse and transform a Fortran AST block into a Python AST.

        Parameters
        ----------
        block : object
            A node or block from the Fortran abstract syntax tree (AST) to be transformed.
        ast_mode : bool, optional
            If True, perform the transformation in AST mode; otherwise, may apply alternate processing to python string.
            Default is True.
        control_stack : list, optional
            Stack used to keep track of control flow constructs (loops, conditionals) during recursion.
            If None, an empty list is initialized.
        counters : dict, optional
            Dictionary to track counters or indices related to AST nodes during traversal.
            If None, an empty dictionary is initialized.
        module_stack : list, optional
            Stack to track the primary module contexts in the Fortran AST.
            If None, an empty list is initialized.

        Returns
        -------
        control_stack : list, optional
            Stack used to keep track of control flow constructs (loops, conditionals) during recursion.
            If None, an empty list is initialized.
        counters : dict, optional
            Dictionary to track counters or indices related to AST nodes during traversal.
            If None, an empty dictionary is initialized.
        module_stack : list, optional
            Stack to track the primary module contexts in the Fortran AST.
            If None, an empty list is initialized.

        Notes
        -----
        This method recursively walks the Fortran AST, transforming nodes into their Python AST
        equivalents. The stacks and counters assist in maintaining contextual information throughout
        the traversal, supporting accurate translation of control flow and modular constructs.
        """
        self.ast_mode = ast_mode
        if control_stack is None: # THis will now be used for the loops and conditional elements
            control_stack = []

        if module_stack is None: # This is the primary stack in which we will contain all the converted/transformed ast code
            module_stack = []
            
        if counters is None:
            counters = {'do': 0, 'if': 0, 'elif':0, 'ifwhere':0, 'elifwhere':0, 'case': 0}

        if hasattr(block, "content"):
            idx = 0
            while idx < len(block.content):
                try:
                    child = block.content[idx]
                    if isinstance(child, F23.Nonlabel_Do_Stmt):
                        for_loop = self.handle_do_stmt(child)
                        self.append_to_current_parent(for_loop, control_stack)
                        control_stack.append(for_loop)  # for_loop has a body
                        counters['do'] += 1
                        # print(ast.unparse(ast.fix_missing_locations(for_loop)))
                    # Handle IF-THEN
                    elif isinstance(child, F23.If_Then_Stmt):
                        if walk(child, F23.Part_Ref):
                            child = self.handle_assignment(child)
                            
                        if_stmt = self.handle_if_condition(child)
                        self.append_to_current_parent(if_stmt, control_stack)
                        control_stack.append(if_stmt)  # if_stmt has body
                        counters['if'] += 1
                    
                    elif isinstance(child,F23.If_Stmt):
                        if_condition = child.children[0]
                        condition_stmt = child.children[1]
                        if_condition_ast = self.handle_expr(if_condition)
                        condition_stmt_ast = self.handle_expr(condition_stmt)
                        if_stmt =  ast.If(
                                test=if_condition_ast,
                                body=[condition_stmt_ast],
                                orelse=[]
                            ) 
                        if counters["if"] == 0 and counters["do"] == 0:
                            module_stack.append(if_stmt)
                        else:
                            self.append_to_current_parent(if_stmt, control_stack)
                            # control_stack.append(if_stmt)

                    elif isinstance(child,F23.Assignment_Stmt):
                        stmt = self.handle_assignment(child)
                        # print(counters, ast.unparse(ast.fix_missing_locations(stmt)))
                        if counters["if"] == 0 and counters["do"] == 0 and counters['case'] == 0: # We don't need to check for the counters['elif'] since the if the `if` counters is empty then elif is also empty 
                            # since elif can't exist without the other. 
                            # control_stack.append(stmt)
                            if counters['ifwhere'] > 0 or counters["elifwhere"] > 0:
                                # Need to create a deepcopy if not they will share the same address, found out the hard way during the
                                # rest of the process
                                stmt_copy = copy.deepcopy(stmt)
                                # Now we need to modify the stmt itself
                                stmt = ast.Assign( 
                                    targets= [ast.Subscript(
                                        value = stmt_copy.targets[0],
                                        slice = ast.Name(id='mask',ctx=ast.Load()),
                                        ctx = ast.Store()
                                    )],
                                    value=self.apply_mask_to_rhs(stmt_copy.value)
                                )
        
                                if module_stack and isinstance(module_stack[-1], (ast.If,list)):
                                    self.append_to_current_parent(stmt, control_stack=module_stack)
                                else:
                                    module_stack.append(stmt)
                            else:
                                # print(ast.unparse(ast.fix_missing_locations(stmt)))
                                module_stack.append(stmt)
                        else:
                            if counters['ifwhere'] > 0:
                                # Need to create a deepcopy if not they will share the same address, found out the hard way during the
                                # rest of the process
                                stmt_copy = copy.deepcopy(stmt)

                                stmt = ast.Assign( 
                                    targets= [ast.Subscript( # LHS SIDE adjustemeent
                                        value = stmt_copy.targets[0],
                                        slice = ast.Name(id='mask',ctx=ast.Load()),
                                        ctx = ast.Store()
                                    )],
                                    # value=stmt_copy.value
                                    value=self.apply_mask_to_rhs(stmt_copy.value) # We need to check the RHS to see if the target name is present and and apply the mask
                                )

                                if control_stack and isinstance(control_stack[-1], (ast.If,list)):
                                    self.append_to_current_parent(stmt, control_stack=control_stack)
                                else:
                                    control_stack.append(stmt)

                                # self.append_to_current_parent(stmt,control_stack=control_stack)
                            else:
                                self.append_to_current_parent(stmt, control_stack)

                    elif isinstance(child, (F23.Else_If_Stmt, F23.Else_Stmt)):
                        if not control_stack or not isinstance(control_stack[-1], ast.If):
                            # print(ast.unparse(ast.fix_missing_locations(control_stack[-1][0])))
                            raise RuntimeError("Else/Else If without a preceding If")

                        parent_if = control_stack[-1] # We go back to the parent if of the current else/else if statement

                        if isinstance(child, F23.Else_If_Stmt):
                            # Create new ast.If node for Else If
                            if isinstance(child, F23.Else_If_Stmt) and walk(child, F23.Part_Ref): 
                                child = self.handle_assignment(child)
                                
                            elif_node = self.handle_if_condition(child)
                            while control_stack and not isinstance(control_stack[-1], ast.If): 
                                control_stack.pop()
                            # Attach to orelse of previous If the new instance IF 
                            parent_if.orelse = [elif_node] 
                            
                            # But we move on to the newly created elif_node
                            control_stack.append(elif_node)
                            counters["elif"] += 1

                        if isinstance(child, F23.Else_Stmt):
                            # print(child)
                            # https://stackoverflow.com/questions/44728436/difference-between-nested-if-else-and-elif
                            if not control_stack or not isinstance(control_stack[-1], ast.If):
                                raise RuntimeError("Else without a preceding If")

                            control_stack.append(parent_if.orelse)
                            
                    elif isinstance(child, (F23.End_Do_Stmt, F23.End_If_Stmt)):
                        if ast_mode and control_stack:
                            if isinstance(child,F23.End_Do_Stmt) and counters["do"] != 0:
                                counters['do'] -= 1
                                #self.loop_variables.popitem() # Pops the last inserted loop variables key and value 
                            elif isinstance(child,F23.End_If_Stmt) and counters["if"] != 0:
                                counters['if'] -= 1

                            if len(control_stack) > 1:
                                # Special handling for ELSE and ELSE IF:
                                # In these cases, we temporarily pushed the `orelse` list (a Python list) onto the stack so `popped` may be a list instead of an AST node.
                                # If the popped element is a list and the current top of stack is an ast.If node,
                                # then we are finishing an ELSE/ELSE IF(which is basically a IF) block, and we may also need to pop the corresponding IF.
                                # print(control_stack[-1],control_stack[-2],counters["if"])
                                if isinstance(control_stack[-1],list) and isinstance(control_stack[-2],ast.If): # and counters["if"] != 0
                                    # This is primarily used for removing the else if present inside the if loop
                                    control_stack.pop()
                                if counters["elif"] > 0: 
                                    # Case: nested IF/ELIF chains
                                    # Fortran uses a single END IF to close a chain of IF / ELSE IF / ELSE, whereas Python AST uses nested `if` statements in `orelse`.
                                    # We need to pop all the nested `ast.If` nodes representing the ELSE IF chain and thanks to the elif statement
                                    # we can ensure that we remoeve only the corresponding the number of if(elif). 
                                    while counters["elif"] > 0:
                                        if isinstance(control_stack[-1], ast.If):
                                            control_stack.pop()
                                            counters["elif"] -= 1
                                        else:
                                            break  
                                # print(control_stack) 
                                if len(control_stack) > 1:
                                    control_stack.pop() # Now we pop the if corresponding to the parent if 
                                # print(control_stack)
                        if counters["do"] == 0 and counters["if"] == 0: # we don't take into account the elif since we primarily based on 
                            # when the end do or end if appear
                            # print(control_stack,counters)
                            module_stack.extend(control_stack)
                            control_stack.clear()
                            
                    elif isinstance(child,F23.Comment):
                        # https://pypi.org/project/ast-comments/
                        pass
                        
                    elif isinstance(child,(F23.Print_Stmt,F23.Write_Stmt)):
                        # FOR NOW WE will treat it as a print since numout = 6 in the class https://docs.oracle.com/cd/E19957-01/805-4940/6j4m1u7oh/index.html
                        # https://stackoverflow.com/questions/28620899/difference-between-write-and-write6-in-fortran
                        # Since 1363 is to write into the files. '(a,i2.2,"|",F13.4,"|",F13.4,"|",3(F9.6))' in split soil
                        # print(child.children)
                        if not any(walk(walk(child,F23.Io_Control_Spec),F23.Int_Literal_Constant)):
                            stmt = self.handle_print_stmt(child)
                            if counters["do"] == 0 and counters["if"] == 0:
                                module_stack.append(stmt)
                            else:
                                self.append_to_current_parent(stmt, control_stack)
                        else:
                            raise NotImplementedError(f"When 1363 is present, the approach hasn't be implemented yet")
                            
                    elif isinstance(child,F23.Call_Stmt):
                        stmt = self.handle_call_stmt(child)
                        if stmt is None:
                            raise ValueError(f'Call statement is None due to prior error')
                        if counters["do"] == 0 and counters["if"] == 0 and counters['case'] == 0:
                            if not isinstance(stmt,ast.Pass):
                                module_stack.append(stmt)
                        else:
                            if not isinstance(stmt,ast.Pass):
                                self.append_to_current_parent(stmt, control_stack)
                    
                    elif isinstance(child,F23.Where_Stmt):
                        condition_stmt_ast = self.handle_expr(child.children[0])
                        value_stmt_ast = self.handle_assignment(child.children[1])

                        stmt_copy = copy.deepcopy(value_stmt_ast)
                                # Now we need to modify the stmt itself
                        stmt = ast.Assign( 
                            targets= [ast.Subscript(
                                value = stmt_copy.targets[0],
                                slice = condition_stmt_ast,
                                ctx = ast.Store()
                            )],
                            value=stmt_copy.value
                        )

                        if counters["do"] == 0 and counters["if"] == 0:
                            module_stack.append(stmt)
                        else:
                            self.append_to_current_parent(stmt, control_stack)

                        
                    elif isinstance(child,F23.Where_Construct_Stmt):
                        # This corresponds to the IF format
                        stmt = self.handle_where_stmt(child)
                        counters['ifwhere'] += 1
                        if counters["do"] == 0 and counters["if"] == 0:

                            self.append_to_current_parent(stmt,module_stack)
                            module_stack.append(stmt)
                        else:
                            self.append_to_current_parent(stmt, control_stack)
                            control_stack.append(stmt)

                    elif isinstance(child,(F23.Masked_Elsewhere_Stmt,F23.Elsewhere_Stmt)):
                        # This corresponds to the ELSEIF format look at replace_where in modifier.py
                        if counters["do"] == 0 and counters["if"] == 0:
                            stack_to_check = module_stack
                        else:
                            stack_to_check = control_stack

                        if not stack_to_check or not isinstance(stack_to_check[-1], ast.If):
                            # print(ast.unparse(ast.fix_missing_locations(control_stack[-1][0])))
                            raise RuntimeError("Else/Else If for where stmt without a preceding If")
                        
                        parent_if = stack_to_check[-1] # We go back to the parent if of the current else/else if statement
                        # print(child)
                        if isinstance(child, F23.Masked_Elsewhere_Stmt):
                            # Create new ast.If node for Else If
                            if isinstance(child, F23.Masked_Elsewhere_Stmt) and walk(child, F23.Part_Ref): 
                                child = self.handle_assignment(child)
                            
                            elif_node = self.handle_where_stmt(child)
                            # while stack_to_check and not isinstance(stack_to_check[-1], ast.If):
                            #     stack_to_check.pop()
                            # Attach to orelse of previous If the new instance IF 
                            parent_if.orelse.append(elif_node) 
                            # print(ast.dump(parent_if,indent=4))
                            # But we move on to the newly created elif_node
                            stack_to_check.append(elif_node)
                            counters["elifwhere"] += 1

                        if isinstance(child, F23.Elsewhere_Stmt):
                            stack_to_check.append(parent_if.orelse)

                    elif isinstance(child,F23.End_Where_Stmt):
                        
                        if counters["do"] == 0 and counters["if"] == 0:
                            stack_to_check = module_stack
                        else:
                            stack_to_check = control_stack

                        if stack_to_check and counters["ifwhere"] != 0:
                            counters['ifwhere'] -= 1

                        if stack_to_check:
                            if len(stack_to_check) > 1:
                                if isinstance(stack_to_check[-1],list) and isinstance(stack_to_check[-2],ast.If): # and counters["if"] != 0
                                    # This is primarily used for removing the else if present inside the if loop
                                    stack_to_check.pop()
                                if counters["elifwhere"] > 0: 
                                    while counters["elifwhere"] > 0:
                                        if isinstance(stack_to_check[-1], ast.If):
                                            stack_to_check.pop()
                                            counters["elifwhere"] -= 1
                                        else:
                                            break  
                                # print(control_stack) 
                                if len(stack_to_check) > 1:
                                    stack_to_check.pop()

                    elif isinstance(child,F23.Implicit_Stmt):
                        pass

                    elif isinstance(child,F23.Subroutine_Stmt):
                        stmt = self.handle_subroutine_stmt(child)
                        if stmt is None:
                            raise ValueError(f'AST subroutine function statement is None due to prior error')
                        module_stack.append(stmt)
                    
                    elif isinstance(child,F23.Type_Declaration_Stmt):
                        if walk(child,F23.Explicit_Shape_Spec) or walk(child,F23.Attr_Spec):
                            if walk(walk(child,F23.Entity_Decl),F23.Name)[0].string not in self.arg_list :
                                stmt = self.handle_type_declaration_stmt(child)
                                if counters["do"] == 0 and counters["if"] == 0:
                                    module_stack.append(stmt)
                                else:
                                    self.append_to_current_parent(stmt, control_stack)

                    elif isinstance(child,(F23.End_Function_Stmt,F23.End_Subroutine_Stmt)):
                        func_def = module_stack[0]  
                        if hasattr(func_def, 'body'):
                            func_body = func_def.body
                            for node in module_stack[1:]:
                                func_body.append(node)
                                
                            module_stack[:] = [func_def] # In the case, we send the function definition, we keep only the function def since
                            # we have appended the execution part inside the body
                            if isinstance(child,F23.End_Function_Stmt):
                                # Try to check if teh eleemnt SUFFIX is present or not, which usually means that we have a function and not a subroutine 
                                return_stmt = walk(walk(child.parent,F23.Suffix),F23.Name)[0]
                                return_node = ast.Return()
                                if return_stmt:
                                    return_node.value = ast.Name(id=return_stmt.string,ctx=ast.Load())
                            
                                    func_def.body.append(return_node)
                        else:
                            raise AttributeError("Function definition does not have a 'body' attribute")
                    
                    elif isinstance(child, F23.Function_Stmt):
                        stmt = self.handle_subroutine_stmt(child)
                        if stmt is None:
                            raise ValueError(f'AST function statement is None due to prior error')
                        
                        module_stack.append(stmt)

                    elif isinstance(child, F23.Cycle_Stmt): # The equivalent of this in Python is 'continue'
                        stmt = ast.Continue()

                        if counters['do'] == 0:
                            raise ValueError(f'Continue(CYCLE in Fortran) stmt needs to be placed inside a For loop')
                        else:
                            self.append_to_current_parent(stmt, control_stack)

                    
                    elif isinstance(child,F23.Exit_Stmt):
                        stmt = ast.Break()

                        if counters['do'] == 0:
                            raise ValueError(f'Break(EXIT in Fortran) stmt needs to be placed inside a For loop')
                        else:
                            self.append_to_current_parent(stmt, control_stack)

                    elif isinstance(child, F23.Select_Case_Stmt):
                        switch_expr = self.handle_expr(child.children[0]) 
                        case_stack = {
                            'type': 'select_case',
                            'switch_expr': switch_expr, 
                            'if_chain': None}
                        control_stack.append(case_stack)
                        counters['case'] = counters.get('case', 0) + 1
                    
                    elif isinstance(child, F23.Case_Stmt):
                        # Each CASE (...) or CASE DEFAULT
                        select_info = next(
                            (s for s in reversed(control_stack) if isinstance(s, dict) and s.get('type') == 'select_case'),
                            None
                        )
                        if not select_info:
                            raise RuntimeError("CASE statement found without an enclosing SELECT CASE")

                        switch_expr = select_info['switch_expr']

                        # Extract the selector (which can be None for DEFAULT)
                        selector_node = child.children[0] 
                        # This may have Case_Value_Range_List or None
                        value_list = getattr(selector_node, 'children', [None])[0]

                        if value_list is None:
                            # CASE DEFAULT
                            prev_if = select_info['if_chain']
                            if prev_if is None:
                                raise RuntimeError("CASE DEFAULT without any preceding CASE")
                            # Default → orelse list of the last if
                            prev_if.orelse = []
                            control_stack.append(prev_if.orelse)
                        else:
                            # Extract value from Case_Value_Range_List
                            case_value_node = walk(selector_node, F23.Name)[0]
                            case_value = self.handle_expr(case_value_node)

                            case_if = ast.If(
                                test=ast.Compare(
                                    left=switch_expr,
                                    ops=[ast.Eq()],
                                    comparators=[case_value]
                                ),
                                body=[],
                                orelse=[]
                            )

                            if select_info['if_chain'] is None:
                                # First CASE — attach to module or enclosing control
                                select_info['if_chain'] = case_if
                                if counters["do"] == 0 and counters["if"] == 0:
                                    module_stack.append(case_if)
                                else:
                                    self.append_to_current_parent(case_if, control_stack)
                            else:
                                # Subsequent CASE — attach to orelse of previous one
                                prev_if = select_info['if_chain']
                                while prev_if.orelse and isinstance(prev_if.orelse[0], ast.If):
                                    prev_if = prev_if.orelse[0]
                                prev_if.orelse = [case_if]
                                select_info['if_chain'] = case_if

                            # # Push this CASE to stack (so body statements append correctly)
                            # control_stack.append(case_if)
                    
                    elif isinstance(child, F23.End_Select_Stmt):
                        if counters.get('case', 0) > 0:
                            counters['case'] -= 1

                        while control_stack and not (isinstance(control_stack[-1], dict) and control_stack[-1].get('type') == 'select_case'):
                            control_stack.pop()
                        if control_stack and isinstance(control_stack[-1], dict):
                            control_stack.pop()

                    elif isinstance(child, F23.Return_Stmt):
                        args = []
                        for child in child.children:
                            if child:
                                args.append(self.handle_expr(child))

                        if args:
                            if len(args) == 1:
                                stmt = ast.Return(value= args[0])
                            else:
                                stmt = ast.Return(value=ast.Tuple(elts = args))
                        else:
                            stmt = ast.Return()
                            
                        if counters["do"] == 0 and counters["if"] == 0:
                            module_stack.append(stmt)
                        else:
                            self.append_to_current_parent(stmt, control_stack)
                    else:   
                        self.recursive_ast(child, ast_mode=ast_mode, control_stack=control_stack,counters=counters,module_stack=module_stack)
                
                except Exception as e:
                    self.logger.exception(f"Exception in recursive block at index {idx}, block type: {type(child).__name__}", e)
                    raise 
                    
                idx += 1
                # print(counters, module_stack, child)
        else:
            raise AttributeError(f"Block doesn't have the `content` attribute for the block : {block}, {type(block)}")
        
        return control_stack,counters,module_stack

    def handle_subroutine_stmt(self, stmt) -> Union[str,ast.FunctionDef]:
        """
        Handles a Fortran subroutine statement, extracting its name and arguments,
        and converting it into a Python function definition.
        
        Parameters
        ----------
        stmt 
            A Fortran subroutine statement to be converted.
        
        Returns
        -------
        str|ast.FunctionDef
            Python function definition corresponding to the subroutine or Python AST based on the `ast_mode` attribute. 
        """
        self.arg_list = []
        for child in stmt.children:
            if child is None:
                continue
            if isinstance(child, F23.Name):
                subroutine_name = child.tostr()
            elif isinstance(child, F23.Dummy_Arg_List):
                arg_list = child.tostr()
                for gchild in child.children:
                    self.arg_list.append(gchild.tostr())
        if not self.ast_mode:
            return f"def {subroutine_name}({arg_list}):"
        else:
            try:                
                args = [ast.arg(arg) for arg in self.arg_list]                
                function_def = ast.FunctionDef(
                    name = subroutine_name,
                        args = ast.arguments(
                            posonlyargs=[],
                            args = args,
                            kwonlyargs=[],
                            kw_defaults=[],
                            defaults=[]
                        ),
                        body = [],
                        decorator_list=[]
                    ) 
                return function_def
            except Exception as e:
                self.logger.exception(f'Exception in handle_subroutine_stmt', e)
                raise 

    def handle_call_stmt(self, stmt) -> Union[str,ast.Expr]:
        """
        Handles a Fortran CALL statement, converting it to a Python function call or AST based on the `ast_mode` attribute
        
        Parameters
        ----------
        stmt
            A Fortran CALL statement.
        
        Returns
        -------
        str|ast.Expr
            Python function call equivalent to the Fortran CALL or Expression based on the `ast_mode` attribute.
        """
        if not self.ast_mode:
            for child in stmt.children:
                if child is None:
                    continue
                if isinstance(child, F23.Name):
                    subroutine_name = child.tostr()
                elif isinstance(child, F23.Actual_Arg_Spec_List):
                    arg_list = child.tostr()
            return f"{subroutine_name}({arg_list})"
        else:
            try:
                if not hasattr(stmt,'children'):
                    raise AttributeError(f'stmt has no children')
                
                if len(stmt.children) != 2:
                    raise ValueError('Expected two children: function_name and args_spec_list')
                
                function_name,args_spec_list = stmt.children
                args = []
                stmt = None
                special_skip_functions = {"xios_orchidee_send_field", "xios_orchidee_recv_field"}

                if not function_name.string in self.extractor.allowed_external_subroutines:
                    for arg in args_spec_list.children:
                        arg_ast = self.handle_expr(arg)
                        # if self.loop_variables:
                        #     adjust_loop_variables(arg_ast,self.loop_variables)
                        args.append(arg_ast)
                        
                    stmt = ast.Expr(value = ast.Call(
                                            func = ast.Name(id = function_name.string, ctx = ast.Load()),
                                            args = args,
                                            keywords = []
                                        )
                                    )
                else:
                    
                    for arg in args_spec_list.children:
                        if isinstance(arg, F23.Char_Literal_Constant):
                            value = arg.items[0].strip(" ' ") # this will remove the 'hydrol' and if there any extra '' inside
                            args.append(value)
                    
                    logging_method ='info' if function_name.string in special_skip_functions else 'error'

                    args.insert(0, "Exception:" if not function_name.string in special_skip_functions else f'INFO: {function_name.string}:')
                    final_message = " ".join(args)
                    
                    # logging.error AST call
                    stmt = ast.Expr(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id='logging', ctx=ast.Load()),
                                attr=logging_method,
                                ctx=ast.Load()
                            ),
                            args=[ast.Constant(value=final_message)],
                            keywords=[]
                        )
                    )
                return stmt
            except Exception:
                self.logger.exception(f'Exception in handle_call_stmt:')
                raise 

    def handle_type_declaration_stmt(self, stmt) -> Union[str,ast.Assign]:
        """
        Handles a Fortran type declaration, converting it to a NumPy array declaration.
        It determines the data type (e.g., REAL, INTEGER) and the dimensions of the array.

        Parameters
        ----------
        stmt
            A Fortran type declaration statement.
        
        Returns:
        str | ast.Assign
            NumPy array declaration corresponding to the Fortran type declaration or the AST based on the `ast_mode` attribute
        """
        if not self.ast_mode:
            var_part = []
            for child in stmt.children:
                if child is None:
                    continue
                if isinstance(child, F23.Intrinsic_Type_Spec):
                    if child.children[0]=='REAL':
                        dtype = 'float32'
                    elif child.children[0]=='INTEGER':
                        dtype = 'int'
                    elif child.children[0]=='LOGICAL':
                        dtype = 'bool'
                    else:
                        raise ValueError("unknown dtype")
                elif isinstance(child, F23.Entity_Decl_List):
                    entity_decls = walk(child, F23.Entity_Decl)
                    assert len(entity_decls) == 1,\
                            "walk(child, F23.Entity_Decl)!= 1, but got a different number."
                    if entity_decls[0].tostr() not in self.arg_list:
                        var_part = entity_decls[0].tostr()
                    else:
                        return None
            if walk(stmt, F23.Explicit_Shape_Spec):
                shape = []
                for dim in walk(stmt, F23.Explicit_Shape_Spec):
                    shape.append(dim.tostr())
                shape.reverse()
                dimensions = ', '.join([name for name in shape])
            else:
                return None

            return f"{var_part} = np.zeros(({dimensions}),dtype={dtype})"
        else:
            shape = []
            TYPE = {'REAL':'np.float64','INTEGER':'np.int32', 'LOGICAL':'np.bool'}
            ast_stmt = None
            try: 
                for dim in walk(stmt,F23.Explicit_Shape_Spec):
                    left,right = None,None
                    lb,ub = dim.children[0],dim.children[1]
                    if lb and ub:
                        # This for the formula: ub - lb + 1
                        right = self.handle_expr(lb)
                        left = self.handle_expr(ub)
                        
                        arg_shape = ast.BinOp(
                                left = ast.BinOp(
                                    left = left,
                                    op = ast.Sub(),
                                    right = right),
                                op = ast.Add(),
                                right = ast.Constant(1))
                        
                        shape.append(arg_shape)
                    elif lb:
                        # Only lower bound is given
                        shape.append(self.handle_expr(lb))

                    elif ub:
                        # Only upper bound is given
                        shape.append(self.handle_expr(ub))
                    else:
                        raise ValueError(f'Both of the, lower bound:{lb}, upper bound:{ub}')
                
                if shape:
                    name = walk(walk(stmt,F23.Entity_Decl),F23.Name)[0].string
                    
                    fdtype = walk(stmt,F23.Intrinsic_Type_Spec)[0].children[0]
                    np_dtype = TYPE.get(fdtype)
                    idx,attr = np_dtype.split('.')
                    if np_dtype is None:
                        raise KeyError("Non corresponding key given")
                    
                    if walk(stmt,F23.Array_Constructor):
                        array_list = walk(walk(stmt,F23.Array_Constructor),F23.Ac_Value_List)[0]
                        elements = []
                        for val in array_list.children:
                            elements.append(self.handle_expr(val))

                        val = ast.Call(
                                func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='array', ctx=ast.Load()),
                                args=[ast.List(elts=elements, ctx=ast.Load())],
                                keywords=[ast.keyword(arg='dtype',
                                    value=ast.Attribute(value=ast.Name(id=idx, ctx=ast.Load()), attr=attr, ctx=ast.Load()))]
                            )
                        ast_stmt = ast.Assign(
                                                targets=[ast.Name(id=name,ctx=ast.Store())],
                                                value=val
                                            )
                    else:
                        ast_stmt = ast.Assign(
                            targets = [ast.Name(id=name,ctx=ast.Store())],
                            value = ast.Call(func=ast.Attribute(value=ast.Name(id=idx,ctx=ast.Load()),attr='zeros',ctx=ast.Load()),
                                            args = [ast.Tuple(elts=shape,ctx=ast.Load())],
                                            keywords = [
                                                ast.keyword(arg='dtype',value = ast.Attribute(value=ast.Name(id = idx,ctx=ast.Load()),
                                                                                            attr=attr,
                                                                                            ctx=ast.Load()))
                                            ]))
                else:
                    intrinsic_type_spec,_,entity_decl_list = stmt.children # This gives out a tuple
                    entity_decls = entity_decl_list.children

                    np_dtype = TYPE.get(intrinsic_type_spec.string, 'np.float64')
                    idx,attr = np_dtype.split('.')

                    for entity_decl in entity_decls:
                        var_name, _,_, initialization = entity_decl.children

                        target = ast.Name(id=var_name.string, ctx=ast.Store())
                        value = None
                        if initialization is not None:
                            _,value = initialization.children
                        if not isinstance(value,F23.Name):
                            if intrinsic_type_spec.children[0] in ["REAL", "INTEGER"]:
                                num_var = self.handle_expr(value)
                                ast_stmt = ast.Assign(
                                            targets=[target],
                                            value=ast.Call( func=ast.Attribute(value=ast.Name(id=idx, ctx=ast.Load()), attr=attr, ctx=ast.Load()),
                                                        args=[num_var],
                                                        keywords = []
                                                    )
                                        )
                            elif intrinsic_type_spec.children[0] == "LOGICAL":
                                if value.string.split(".")[1] == "TRUE":
                                    bool_val = True
                                        
                                    bool_call = ast.Call(
                                                func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='bool', ctx=ast.Load()),
                                                args=[ast.Constant(value=bool_val)],
                                                keywords=[]
                                            )
                                    ast_stmt = ast.Assign(
                                        targets=[target],
                                        value = bool_call
                                    )
                        
                return ast_stmt
            except Exception:
                self.logger.exception(f'Exception in handle_type_declaration_stmt')
                raise 

    def simplify_limits(self, expression) -> str:
        """
        Simplifies expressions for loop bounds or array dimensions, combining constants 
        and variables in the expression.

        Parameters
        ----------
        expression
            The expression representing loop bounds or dimensions.
        
        Returns
        -------
        str
            Simplified version of the expression.
        """
        terms = re.split(r'\s*([+\-])\s*', expression)
        numbers = []
        variables = []
        for i in range(len(terms)):
            if i % 2 == 0:
                if terms[i].isdigit() or (terms[i].startswith('-') and terms[i][1:].isdigit()):
                    if i > 0 and terms[i-1] == '-':
                        numbers.append(-int(terms[i]))
                    else:
                        numbers.append(int(terms[i]))
                elif terms[i].strip():
                    variables.append(terms[i].strip())

        total = sum(numbers)
        new_expression = ' + '.join(variables)
        if total != 0:
            sign = '+' if total > 0 else '-'
            new_expression += f' {sign} {abs(total)}'
        return new_expression.lstrip('+ ').strip()

    def handle_end_stmt(self, stmt) -> str:
        assert isinstance(stmt, (F23.End_Do_Stmt, F23.End_If_Stmt, F23.End_Function_Stmt, F23.End_Subroutine_Stmt)), (
                f"Unexpected statement type: {type(stmt).__name__}. Expected one of: "
                f"End_Do_Stmt, End_If_Stmt, End_Function_Stmt, End_Subroutine_Stmt.")
        return ''

    def handle_specification(self, stmt) -> None:
        print(f"# Specification: {stmt}")
        self.result.append(f"# Specification: {stmt}")

    def handle_where_stmt(self, stmt:F23.Where_Construct_Stmt) -> Union[str,ast.Expr]:
        """
        Handles a Fortran WHERE construct statement, converting it into an equivalent Python representation or ast.Expr.

        Parameters
        ----------
        stmt : F23.Where_Construct_Stmt
            A Fortran WHERE construct statement.

        Returns
        -------
        str or ast.Expr
            Python equivalent of the Fortran WHERE construct, either as source code or Expression node(ast.Expr),depending on the `ast_mode` attribute.
        """

        if not self.ast_mode:
            condition = stmt.items[0].string
            print(f"np.where({condition})")
            self.result.append(f"np.where({condition})")
        else:
            try:
                mask_node = []
                for element in stmt.children:
                    if element:
                        mask_node.append(self.handle_expr(element))
                if len(mask_node) == 1:
                    if isinstance(mask_node[0],ast.Subscript):
                        expr_node = ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id='np',ctx=ast.Load()),
                                    attr='any',
                                    ctx=ast.Load()
                                ),
                                args = [mask_node[0]],
                                keywords=[]
                            )
                    elif isinstance(mask_node[0], ast.Compare):
                        expr_node = ast.Call(
                            func=ast.Attribute(
                                value=mask_node[0],
                                attr="any",
                                ctx=ast.Load()
                            ),
                            args=[],
                            keywords=[]
                        )
                    else:
                        expr_node = ast.Call(
                            func=ast.Attribute(
                                value=mask_node[0],
                                attr="any",
                                ctx=ast.Load()
                            ),
                            args=[],
                            keywords=[]
                        )
                        # raise NotImplementedError(f"Not implemented error for the where condition type:{type(mask_node[0])}")

                    where_if_stmt = ast.If(
                        test = expr_node,
                        body= [ast.Assign(
                                    targets=[ast.Name(id='mask',ctx=ast.Store())],
                                    value=mask_node[0]
                                )],
                        orelse=[]
                    )
                    return where_if_stmt
                else:
                    raise NotImplementedError("Not implemented handle_where_stmt with multiple conditions")
                
            except Exception:
                self.logger.exception(f'Exception in handle_where_stmt')
                raise 

    def get_conventional_var(self, candidate_var:str, upper_key:str) -> str:
        """
        Retrieve and correct the loop variable corresponding to the upper key.

        Parameters
        ----------
        candidate_var : str
            Loop variable that needs to be checked.
        upper_key : str
            Upper key to which the loop variable needs to correspond.

        Returns
        -------
        str
            Corresponding (and possibly corrected) loop variable.
        """
        normalized_upper_key = upper_key.replace(" ", "")
    
        mapping = {}
        for key, var_set in self.extractor.loop_dict.items():
            normalized_key = key.replace(" ", "")  # normalize keys from loop_dict

            if len(var_set) == 1:
                only_var = next(iter(var_set))
                mapping[(normalized_key, only_var)] = only_var
            elif len(var_set) == 2:
                var1, var2 = sorted(var_set)
                mapping[(normalized_key, var1)] = var2
                mapping[(normalized_key, var2)] = var2

        return mapping.get((normalized_upper_key, candidate_var))
    
    def handle_do_stmt(self, stmt) -> Union[str,ast.For]:
        """
        Handles a Fortran DO loop, converting it into a Python for loop or AST for loop based on the `ast_mode` attribute.

        Parameters
        ----------
        stmt
            A Fortran DO loop statement.

        Returns
        -------
        str|ast.For
            Python for loop equivalent to the Fortran DO loop or AST for loop based on the `ast_mode` attribute.
        """
        if not self.ast_mode:
            line_parts = stmt.tostr().split('=')
            loop_var = line_parts[0].split()[-1]
            start_end_stride_values = line_parts[1].split(',')
            start = start_end_stride_values[0].strip() + '-1'
            end = start_end_stride_values[1].strip()
            if len(start_end_stride_values)==2:
                stride = 1
            elif len(start_end_stride_values)==3:
                stride = start_end_stride_values[2].strip()
            else:
                raise ValueError("Loop control error!")
            lb = self.simplify_limits(start)
            if not lb:
                lb = 0
            return f"for {loop_var} in range({lb}, {end}, {stride}):"
        else:
            # WE directly use the stmt not as the string except for the start which corresponds to the lower bound 
            elements = walk(stmt,F23.Loop_Control)[0].children[1]
            if elements: # FOr loop DO in fortran 
                loop_var, start_end_stride_values = elements[0].string,elements[1]
                start,end = start_end_stride_values[0], start_end_stride_values[1]

                # Now we need to make sure that the start(lower bound) is compatible which we transform to string
                if isinstance(start,F23.Int_Literal_Constant):
                    value = self.simplify_limits(start.tostr() + '-1')
                    if value == '':
                        start = F23.Int_Literal_Constant('0')
                    else:
                        start = F23.Int_Literal_Constant(value)
                else:
                    start = F23.Level_2_Expr((start.tostr(),'-','1'))
                if len(start_end_stride_values) == 2:
                    stride = 1
                elif len(start_end_stride_values) == 3:
                    stride = start_end_stride_values[2]
                else:
                    raise ValueError("Loop control error!")
                
                arg = []
                lb = start

                if not lb:
                    lb = 0
                # FOr the lower bound control which is kept as a string 
                if type(lb) == int: # is of type int
                    arg.append(ast.Constant(value=int(lb)))
                else:
                    arg.append(self.handle_expr(lb))
                    
                # Before creating the ast node, we need to verify if the loop_var corresoponds the convention used 
                # based on the upper bound, THe candidate var is also in string format
                candidate_var = self.get_conventional_var(loop_var,end.string)
                # self.loop_variables[loop_var] = candidate_var
                # if candidate_var is not None:
                #     loop_var = candidate_var

                end_ast = None
                if isinstance(end,F23.Part_Ref):
                    end_ast = ast.Call(func=ast.Name(id='int',ctx=ast.Load()),
                                                    args =[self.handle_expr(end)],
                                                    keywords = [])
                else:
                    end_ast = self.handle_expr(end)
                
                arg.append(end_ast)
                # We need to verify the type of the stride since the stride could be an integer format or that of the unary format ex : -1 or -nslm 
                if type(stride) == int:
                    arg.append(ast.Constant(value = stride))
                else:
                    stride_ast = self.handle_expr(stride)
                    # Before appending we need to verify if that the if the stride is negative thus the end variable need to be negative too
                    if isinstance(stride_ast,ast.UnaryOp):
                        end_ast = arg[-1]
                        if isinstance(end_ast, ast.Constant) and isinstance(end_ast.value, int):
                            if end_ast.value > 0:
                                end_ast.value = -1 * end_ast.value
                    arg.append(stride_ast)

                for_loop = ast.For(
                    target = ast.Name(id=loop_var, ctx = ast.Store()),
                    iter = ast.Call(
                        func = ast.Name(id="range",ctx = ast.Load()),
                        args = arg,
                        keywords = []),
                    body = [],
                    orelse = []
                )       

            else: # While loop DO while 
                # print(stmt.children)
                loop_control = walk(stmt,F23.Loop_Control)[0]
                for cont in loop_control.items:
                    if cont is not None:
                        loop_control_ast = self.handle_expr(cont)
                    
                for_loop = ast.While(
                    test = loop_control_ast,
                    body=[],
                    orelse=[]
                )

            return for_loop

    def handle_if_condition(self, condition) -> Union[str,ast.If]:
        """
        Handles a Fortran IF condition, converting it to a Python IF statement or If AST based on the `ast_mode` attribute.

        Parameters
        ----------
        stmt
            A Fortran IF condition statement.
        
        Returns
        -------
        str|ast.If
            Python IF statement equivalent to the Fortran condition or If AST based on the `ast_mode` attribute.
        """
        if not self.ast_mode:
            for fortran_op, python_op in self.replacements.items():
                condition = re.sub(fortran_op, python_op, condition, flags=re.IGNORECASE)
            return condition
        else:
            try:
                if condition is None:
                    raise ValueError(f'condition argument is None')
                condition_stmt = None
                if isinstance(condition,ast.AST):
                    condition_stmt = ast.If(
                                test = condition,
                                body=[],
                                orelse=[]
                        )
                else:
                    if hasattr(condition, 'children') and condition.children[0] is not None: # These cases corresponds to the IF/ELSE IF 
                        if len(condition.children) == 1 and isinstance(condition.children[0],F23.Name): # This is for the logical case
                            condition = self.handle_expr(condition.children[0])
                            condition_stmt = ast.If(
                                test = condition,
                                body=[],
                                orelse=[]
                            )
                        else:
                            stmt = self.handle_assignment(condition)
                            condition_stmt = ast.If(
                                test = stmt,
                                body=[],
                                orelse=[]
                            )

                return condition_stmt
            except Exception:
                self.logger.exception(f'Exception in handle_if_condition')
                raise

    def handle_print_stmt(self, stmt) -> Union[str,ast.Expr]:
        """
        Handles a Fortran PRINT statement, converting it into a Python print statement or the AST based on the `ast_mode` attribute.

        Parameters
        ----------
        stmt
            A Fortran PRINT statement.
        
        Returns
        -------
        str|ast.Expr
            Python print statement equivalent to the Fortran PRINT or the AST based on the `ast_mode` attribute.
        """
        if not self.ast_mode:
            assert isinstance(stmt, F23.Print_Stmt), (
                    f"Unexpected statement type: {type(stmt).__name__}. Expected one of: "
                    f"Print_Stmt")
            output_item_list = ''
            for child in stmt.children:
                if isinstance(child, F23.Output_Item_List):
                    output_item_list = child.tostr()
            return f"print({output_item_list})"
        else:
            try:
                output_item_lists = []
                str_print = []
                for child in stmt.children:
                    if isinstance(child, F23.Output_Item_List):
                        for elements in child.children:
                            if isinstance(elements,F23.Char_Literal_Constant):
                                str_print.append(self.handle_expr(elements))
                            else:
                                output_item_lists.append(self.handle_expr(elements)) # THis to retrieve the elements that are not strings
                if self.ast_mode:
                    values = []
                    print_call = None
                    if output_item_lists:
                        # By doing so will allows us to handle case when we have str,variable,variable, variable and str,variable,str,variable printing
                        for i, element in enumerate(itertools.zip_longest(str_print,output_item_lists)):
                            str_print, item_list = element

                            if str_print is not None:
                                values.append(str_print)

                            if item_list is not None:
                                values.append( ast.FormattedValue( value = item_list,
                                        conversion= -1
                                    ))
                                if i < len(output_item_lists) - 1:
                                    values.append(
                                        ast.Constant(value=',')
                                    )
                        print_call = ast.Call(
                            func = ast.Attribute(value = ast.Name(id = 'logging', ctx = ast.Load()),
                                     attr = 'info',
                                     ctx = ast.Load()),
                            args=[ast.JoinedStr(values=values)],
                            keywords=[]
                        )
                        # print(ast.unparse(ast.fix_missing_locations(print_call))) 
                    else:
                        print_call = ast.Call(
                            func = ast.Attribute(value = ast.Name(id = 'logging', ctx = ast.Load()),
                                     attr = 'info',
                                     ctx = ast.Load()),
                            args=str_print,
                            keywords=[])
                        
                    return ast.Expr(value=print_call)
            except Exception:
                self.logger.exception(f'Exception in handle_print_statement')
                raise

    def handle_intrinsic_function_reference(self, stmt_str:str, intrinsic_function_reference:Union[F23.Intrinsic_Function_Reference, List]) -> Union[str,ast.Call]:
        """
        Handles a Fortran intrinsic function reference, converting it into a Python AST call or string representation.

        Parameters
        ----------
        stmt_str : str
            The Fortran statement string for context or diagnostics.

        intrinsic_function_reference : F23.Intrinsic_Function_Reference | list
            The parsed Fortran intrinsic function reference.

        Returns
        -------
        str | ast.Call
            Python equivalent of the Fortran intrinsic function as a string or AST node.
        """
        if not self.ast_mode:
            for func in intrinsic_function_reference:
                for child in func.children:
                    if child is None:
                        continue 
                    if isinstance(child, F23.Intrinsic_Name):
                        intrinsic_name = child.tostr()
                        pattern = rf'\b{intrinsic_name}\b'
                        assert pattern in self.intrinsic_replacements, f"{intrinsic_name} is not in the intrinsic replacements!"
                        np_func = self.intrinsic_replacements[pattern]
                        stmt_str = re.sub(pattern, np_func, stmt_str, flags=re.IGNORECASE)
            return stmt_str
        else:
            try:
                pattern = rf'\b{intrinsic_function_reference.items[0].string}\b'
                func_name = self.intrinsic_replacements.get(pattern,None) # THere are some intrinsic functions in python whose arguments might differ
                # from that of the FORTRAN, thus we will handle them here
                intrinsic_func = None 
                if func_name is None:
                    # in here : https://www.intel.com/content/www/us/en/docs/fortran-compiler/developer-guide-reference/2025-0/epsilon.html
                    # it says that the x that enters must be real for epsilon
                    # but for some pythonic functions you might not need to send an argument or any element such the case of epsilon, in python we 
                    # don't need argument as such, we can already predine the python AST for such case
                    instrinsic_exception = {
                        r'\bEPSILON\b': ast.Attribute(
                            value = ast.Call(func = ast.Attribute(value = ast.Name(id='np',ctx = ast.Load()),
                                                                attr = 'finfo',
                                                                ctx = ast.Load()),
                                            args = [
                                                ast.Attribute(value = ast.Name(id='np',ctx=ast.Load()),
                                                            attr = 'float64',
                                                            ctx = ast.Load())
                                            ],
                                            keywords = []),
                            attr = 'eps',
                            ctx = ast.Load())
                    }
                    intrinsic_func = instrinsic_exception.get(pattern,None)
                    if not intrinsic_func:
                        raise NotImplementedError(f"Not implemented intrinsic exception:{pattern}")
                        
                if func_name:
                    # Once we retieve the function name we need to identify if the it's just normal instrinics or numpy based intrinsic function
                    func = None 
                    if len(func_name.split(".")) > 1:
                        parts = func_name.split(".")
                        func = ast.Attribute(value = ast.Name(id=parts[0],ctx=ast.Load()), attr = parts[1], ctx = ast.Load())
                    else:
                        func = ast.Name(id=func_name.lower(), ctx=ast.Load())
                        
                    # FInd the intrinsic arguments, as such it would allow us to retrieve the the actual arguments inside the intrinsic parameter
                    intrinsic_args = walk(intrinsic_function_reference,F23.Actual_Arg_Spec_List)[0]
                    args = []
                    keywords = []
                    # Since we do this,it will recuresively call the handle_expr
                    # onto the elements inside the intrinsic function in question : max(mc[ji, jsl, ins] - mcr[ji,], zero), which will get transformed and
                    # onto a binary operation 
                    for iarg, arg in enumerate(intrinsic_args.children):
                        if func_name == "np.sum":
                            if isinstance(arg,F23.Actual_Arg_Spec): # THis is meant to work for the SUM since they might or might have DIM as argument 
                                keywords.append(self.handle_expr(arg))
                            elif not isinstance(arg, F23.Actual_Arg_Spec) and iarg > 0: # THis is the case where it might not have the DIM as argument 
                                expr = self.handle_expr(arg)
                                if isinstance(arg, F23.Int_Literal_Constant):
                                    adjusted_value = str(int(expr.value) - 1)
                                    expr = F23.Int_Literal_Constant(adjusted_value, None)
                                else:
                                    expr = F23.Level_2_Expr((expr,'-','1'))
                                
                                expr = self.handle_expr(expr)
                                keywords.append(ast.keyword(arg='axis',value = expr))

                            else:
                                args.append(self.handle_expr(arg))
                        else:
                            if isinstance(arg,F23.Actual_Arg_Spec): # THis is meant to retrieve the keywords(argument of instrinsic function) other than the variable
                                keywords.append(self.handle_expr(arg))
                            else:
                                args.append(self.handle_expr(arg))

                    intrinsic_func = ast.Call(
                        func=func,  
                        args=args,
                        keywords=keywords
                    )  
                    
                    
                return intrinsic_func
            except Exception:
                self.logger.exception(f'Exception in handle_intrinsic_function_reference')
                raise

    def handle_real_literal_constant(self, stmt_str:str, real_literal_constant:Union[F23.Int_Literal_Constant,List]) -> Union[str,ast.Constant]:
        """
        Handles a Fortran real literal constant, converting it into a Python constant or its string representation.

        Parameters
        ----------
        stmt_str : str
            The Fortran statement string for context or diagnostics.

        real_literal_constant : F23.Int_Literal_Constant | list
            The parsed Fortran real literal constant.

        Returns
        -------
        str | ast.Constant
            Python equivalent of the Fortran real constant as a string or AST node.
        """
        if not self.ast_mode:
            for item in real_literal_constant:
                pre = item.children[1]
                stmt_str = stmt_str.replace(f"_{pre}", '')
            return stmt_str
        else:
            return ast.Constant(value=float(real_literal_constant.items[0]))

    def handle_part_ref(self, stmt_str:str, part_ref:Union[List,F23.Part_Ref]) -> Union[str,ast.Subscript]:
        """
        Handles a Fortran part reference, converting it into a Python subscript or its string representation.

        Parameters
        ----------
        stmt_str : str
            The Fortran statement string for context or diagnostics.

        part_ref : list | F23.Part_Ref
            The parsed Fortran part reference.

        Returns
        -------
        str | ast.Subscript
            Python equivalent of the Fortran part reference as a string or AST subscript node.
        """
        if not self.ast_mode:
            for array in part_ref:
                name = array.children[0].tostr()
                shape = []
                for child in array.children:
                    if isinstance(child, F23.Section_Subscript_List):
                        for idim, dim in enumerate(child.children):
                            limits = dim.tostr().split(':')
                            lb = limits[0]
                            if len(limits) > 1:
                                ub = limits[1]
                                if lb:
                                    lb = lb + '-1'
                                lb = self.simplify_limits(lb)
                                ub = self.simplify_limits(ub)
                                shape.append(f"{lb}:{ub}")
                            elif len(limits)==1:
                                shape.append(f"{lb}")
                dimensions = ', '.join([sh for sh in shape])
                numpy_ref_str = f"{name}[{dimensions}]"
                stmt_str = stmt_str.replace(array.tostr(), numpy_ref_str)

            return stmt_str
        
        else:
            try:
                name = walk(part_ref, F23.Name)[0].string
                name_found = False

                # Check if name exists in any of the keys from all_array_info
                for elements in self.extractor.all_array_info.values():
                    if name in elements.keys():
                        name_found = True
                        break

                if not name_found:
                    _,args_spec_list = part_ref.children
                    # Name not found in any array info: which probabaly measn it's an functioin 
                    args = []
                    for arg in args_spec_list.children:
                        args.append(self.handle_expr(arg))
                        
                    subscript = ast.Call(
                                    func = ast.Name(id = name, ctx = ast.Load()),
                                    args = args,
                                    keywords = []
                                )
                else:
                    part_refs = walk(part_ref,F23.Part_Ref)
                    subscript = None
                    args = []
                    if len(part_refs) == 1:
                        for array in part_refs:
                            name = array.children[0].tostr()
                            shape = []
                            for child in array.children:
                                if isinstance(child, F23.Section_Subscript_List):
                                    for idim, dim in enumerate(child.children):                                        
                                        limits = dim.tostr().split(':')
                                        lb = limits[0]
                                        if len(limits) > 1:
                                            ub = limits[1]
                                            if lb:
                                                lb = lb + '-1'
                                            lb = self.simplify_limits(lb)
                                            ub = self.simplify_limits(ub)
                                            
                                            shape.append((f"{lb}:{ub}",dim))
                                        elif len(limits)==1:
                                            shape.append((f"{lb}",dim))

                            for sh,node in shape:
                                # print(sh,node)
                                if ':' in sh and isinstance(node, F23.Subscript_Triplet):
                                    # It's a slice
                                    if sh == ':':
                                        # Simple ':' slice
                                        slice_node = ast.Slice()
                                    else:
                                        # Possibly lb:ub
                                        lb_ub = sh.split(':')
                                        lb = lb_ub[0].strip() or None
                                        ub = lb_ub[1].strip() if len(lb_ub) > 1 else None
                            
                                        slice_node = ast.Slice(
                                            lower=self.handle_expr(node.children[0]) if lb else None, # ast.Name(id=lb, ctx=ast.Load())
                                            upper=self.handle_expr(node.children[1]) if ub else None # ast.Name(id=ub, ctx=ast.Load())
                                        )
                                        # if lb:  
                                        #     adjust_loop_variables(slice_node.lower,self.loop_variables)
                                        # if ub:
                                        #     adjust_loop_variables(slice_node.upper,self.loop_variables)

                                    args.append(slice_node)
                                else:
                                    # it's a direct index
                                    # expr_node = ast.parse(sh, mode='eval').body
                                    if isinstance(node,ast.AST):
                                        expr_node = node
                                        # if self.loop_variables:
                                        #     adjust_loop_variables(expr_node,self.loop_variables)

                                    else:
                                        expr_node = self.handle_expr(node)
                                        # if self.loop_variables:
                                        #     adjust_loop_variables(expr_node,self.loop_variables)

                                    args.append(expr_node)
                            
                            if len(args) == 1:
                                subscript = ast.Subscript(
                                    value=ast.Name(id=name, ctx=ast.Load()),
                                    slice=args[0],
                                    ctx=ast.Load()
                                )
                            else:
                                subscript = ast.Subscript(
                                    value=ast.Name(id=name, ctx=ast.Load()),
                                    slice=ast.Tuple(elts=args, ctx=ast.Load()),
                                    ctx=ast.Load()
                                )
                    else:
                        name = None
                        elts = []
                        for elements in part_ref.children:
                            if isinstance(elements,F23.Name):
                                name = ast.Name(id=elements.tostr(),ctx =ast.Load())
                            elif isinstance(elements,F23.Section_Subscript_List):
                                for child in elements.children:
                                    node = self.handle_expr(child)
                                    if isinstance(node,ast.Subscript) and not any(ast_walk(node,ast.Slice)):
                                        # Need to check if there is ast.Slice
                                        elts.append(ast.Call(func=ast.Name(id='int',ctx=ast.Load()),
                                                    args = [node],
                                                    keywords = []))
                                    else:
                                        if not isinstance(node,list):
                                            # adjust_loop_variables(node,self.loop_variables)
                                            elts.append(node)
                                        else:
                                            # for elem in node:
                                            #     adjust_loop_variables(elem,self.loop_variables)
                                            elts.extend(node)

                        subscript = ast.Subscript(
                                    value=name,
                                    slice=elts[0] if len(elts)== 1 else ast.Tuple(elts=elts,ctx=ast.Load()),
                                    ctx=ast.Load()
                                )
                        # print(ast.unparse(ast.fix_missing_locations(subscript)))
                return subscript
            except Exception:
                self.logger.exception(f'Exception in handle_part_ref')
                raise 
    
    def handle_level_4epr(self, stmt) -> Union[None,ast.Compare]:
        """
        Handles a Fortran level-4 expression, converting it into a Python comparison AST node.

        Parameters
        ----------
        stmt
            A parsed Fortran level-4 expression.

        Returns
        -------
        None | ast.Compare
            Python AST comparison node or None if not applicable.
        """
        try:
            level4_expr = walk(stmt,F23.Level_4_Expr)[0].children
            left_node, op, right_node = level4_expr

            left_ast = self.handle_expr(left_node)
            right_ast = self.handle_expr(right_node)
            
            if left_ast is None or right_ast is None:
                raise ValueError(f'Either left or right part of handle_or_and_operand, left:{left_ast}, right:{right_ast}')

            pattern = rf'\.{op.strip(".")}\.'
                                
            operator = self.replacements.get(pattern,None)
            # We will check if the conditional operator is present in the self.replacements dict if it's None then we check directly into the conditional_op_map 
            # which contains the ast format of each conditional operators directly.
            if operator is not None:
                ast_op = self.conditional_ops_map.get(operator,None)
            else:
                ast_op = self.conditional_ops_map.get(op,None)

            if ast_op is None:
                raise KeyError(f"Error in ast_mapping: {op} isn't available in the ast_map")

            ast_stmt = ast.Compare(
                left = left_ast,
                    ops = [
                        ast_op
                    ],
                    comparators = [
                        right_ast
                    ]
                )
            
            return ast_stmt
        
        except Exception:
            self.logger.exception(f'Exception in handle_level_4expr')
            raise

    def handle_OR_AND_Operand(self, stmt) -> Union[None,ast.UnaryOp]:
        """
        Handles a Fortran OR/AND operand, converting it into a Python unary operation AST node.

        Parameters
        ----------
        stmt
            A parsed Fortran OR or AND operand expression.

        Returns
        -------
        None | ast.UnaryOp
            Python AST unary operation node or None if not applicable.
        """
        try:
            or_and_stmt = None
            ast_stmt = None
            if any(walk(stmt,F23.Or_Operand)):
                or_and_stmt = walk(stmt,F23.Or_Operand)[0]
            elif any(walk(stmt,F23.And_Operand)):
                or_and_stmt = walk(stmt,F23.And_Operand)[0]
            # We need to check the length of the stmt
            if len(stmt.children) > 2:
                
                left_ast = self.handle_expr(or_and_stmt.items[0])
                op_token = or_and_stmt.items[1]
                right_ast = self.handle_expr(or_and_stmt.items[2])

                if left_ast is None or right_ast is None:
                    raise ValueError(f'Either left or right part of handle_or_and_operand, left:{left_ast}, right:{right_ast}')
                
                op_str = op_token.strip().strip('.')  
                op_map = {
                    'AND': ast.And(),
                    'OR': ast.Or(),
                }            
                op = op_map.get(op_str.upper())
                if not op:
                    raise NotImplementedError(f"Logical operator {op_token} not supported.")
                
                # Helper function to check if a node contains a Subscript with a Slice
                def contains_subscript_with_slice(node):
                    for sub_node in ast.walk(node):
                        if isinstance(sub_node, ast.Subscript):
                            if isinstance(sub_node.slice, ast.Slice):
                                return True
                    return False
                
                # When we an ast of type COmpare on either and both of them has an array with slice inside of them then we need to replace the operator AND/OR to bitwise comparaison
                if (isinstance(left_ast, ast.Compare) and isinstance(right_ast, ast.Compare) and contains_subscript_with_slice(left_ast) and contains_subscript_with_slice(right_ast)):
                    # Replace logical operator with bitwise equivalent
                    if isinstance(op, ast.And):
                        op = ast.BitAnd()
                    elif isinstance(op, ast.Or):
                        op = ast.BitOr()

                    ast_stmt = ast.BinOp(left=left_ast, op=op, right=right_ast)

                else:
                    ast_stmt = ast.BoolOp(op=op, values=[left_ast, right_ast])
                
            elif len(stmt.children) == 2 : # We are in the case of element such as NOT ok_freeze_cwr
                op,operand = stmt.children
                pattern = rf'\.{op.strip(".")}\.'
        
                operator = self.replacements.get(pattern,None)
                if operator is not None:
                    ast_op = self.conditional_ops_map.get(operator,None)
                else:
                    ast_op = self.conditional_ops_map.get(op,None)

                if ast_op is None:
                    raise KeyError(f"Error in ast_mapping: {op} isn't available in the ast_map")

                ast_stmt = ast.UnaryOp(
                    op = ast_op,
                    operand = self.handle_expr(operand)
                )
            
            return ast_stmt
            
        except Exception:
            self.logger.exception(f'Exception in handle_OR_AND_Operand')
            raise 
        
    def handle_assignment(self, stmt) -> Union[str,ast.Assign]:
        """
        Handles a Fortran assignment statement, converting it into a Python assignment.

        Parameters
        ----------
        stmt
            A Fortran assignment statement.
        
        Returns
        -------
        str | ast.Assign
            Python AST or Python string assignment equivalent to the Fortran statement.
        """
        stmt_str = stmt.tostr()
        part_ref = walk(stmt, F23.Part_Ref)
        intrinsic_function_reference = walk(stmt, F23.Intrinsic_Function_Reference)
        real_literal_constant = walk(stmt, F23.Real_Literal_Constant) 

        if not self.ast_mode:
            if intrinsic_function_reference:
                stmt_str = self.handle_intrinsic_function_reference(stmt_str, intrinsic_function_reference)

            if real_literal_constant:
                stmt_str = self.handle_real_literal_constant(stmt_str, real_literal_constant)

            if part_ref:
                stmt_str = self.handle_part_ref(stmt_str, part_ref)

            return stmt_str

        else:
            try:
                lhs_ast,rhs_ast = None,None 
                ast_stmt = None
                if len(stmt.children) == 3: # THese are usally meant for this such as a[i,j] = m[i,j] + ...
                    lhs_node, eq_sign, rhs_node = stmt.children
                    # Handle left side of the assignement
                    if isinstance(lhs_node, F23.Name):
                        # In some cases, we observed that arrays are assigned like this : a = TRUE within a function locally in this case
                        # Python will create a new local variables with the same name thus could pose a problem further down the code that might use the actual variable 
                        elem_found = None
                        for key,value in self.extractor.all_array_info.items():
                            if (lhs_node.string in value.keys()) and isinstance(rhs_node, (F23.Name,F23.Logical_Literal_Constant,F23.Real_Literal_Constant, F23.Int_Literal_Constant)):
                                elem_found = self.extractor.all_array_info[key][lhs_node.string]
                        
                        if elem_found:
                            if len(elem_found) == 1:
                                nb_slices = ast.Slice()
                            elif len(elem_found) > 1:
                                nb_slices = ast.Tuple(elts=[ast.Slice() for _ in range(len(elem_found))],ctx=ast.Load())
                            lhs_ast = ast.Subscript(
                                value=ast.Name(id=lhs_node.string,ctx = ast.Store()),
                                slice=nb_slices
                            )
                        else:
                            lhs_ast = ast.Name(id=lhs_node.string,ctx = ast.Store())
                    elif isinstance(lhs_node,F23.Part_Ref):
                        lhs_ast = self.handle_expr(lhs_node)
                        if lhs_ast is None:
                            raise ValueError(f'lhs_ast for handle assignement is None')
            
                    # Handle right side of the assignement
                    rhs_ast = self.handle_expr(rhs_node)
                    if rhs_ast is None:
                        raise ValueError(f'rhs_ast for handle assignement is None')
                    
                    # We need to add another check that when we have an array(rhs) to another array in left hand side thus we need to create copy
                    dim_found = None
                    left_name = None
                    right_name = None

                    if isinstance(lhs_ast,ast.Name) and isinstance(rhs_ast,ast.Name):
                        left_name = lhs_ast.id
                        right_name = rhs_ast.id
                    # Check if name exists in any of the keys from all_array_info
                    for key,value in self.extractor.all_array_info.items():
                        if (left_name in value.keys()) and (right_name in value.keys()):
                            dim_found = self.extractor.all_array_info[key][left_name]
                            break
                    # If rhs is a known array, wrap with .copy() which creates a litteral new copy of the attributes thus we will go with the a[:,:] = b which copies onto the array
                    if dim_found: # THIs is to ensrue that only the assignement of type a = b.copy() is affected 

                        if len(dim_found) == 1:
                            arg = ast.Slice()
                        else:
                            arg = ast.Tuple(elts=[])
                            for _ in range(len(dim_found)):
                                arg.elts.append(ast.Slice())

                        lhs_ast = ast.Subscript(
                            value = lhs_ast,
                            slice=arg
                        )
                        pass

                    ast_stmt = ast.Assign(
                        targets = [lhs_ast],
                        value = rhs_ast
                    )
                else:
                    # THis is for the case that might have a and b type elements ex: humrel[ji, jv] > min_sechiba and soiltile[ji, jst] * vegtot[ji] > min_sechiba
                    ast_stmt = self.handle_expr(stmt.items[0])
                    if ast_stmt is None:
                        raise ValueError(f'ast_stmt for handle assignement is None')
                return ast_stmt
            
            except Exception:
                self.logger.exception(f'Exception in handle_assignement')
                raise 
        
    def build_binop(self, left, op_token, right) -> ast.BinOp:
        """
        Builds a Python binary operation AST node from the given operands and operator token.

        Parameters
        ----------
        left
            The left-hand side operand.

        op_token
            The operator token representing the binary operation ('+', '-', '*', etc.).

        right
            The right-hand side operand.

        Returns
        -------
        ast.BinOp
            Python AST node representing the binary operation.
        """
        # Get operator symbol from token or string
        op_str = str(op_token).strip()

        op_map = {
            '+': ast.Add(),
            '-': ast.Sub(),
            '*': ast.Mult(),
            '/': ast.Div(),
            '**': ast.Pow(),
        }

        op = op_map.get(op_str)
        if not op:
            raise NotImplementedError(f"Operator {op_str} not supported.")
        
        return ast.BinOp(left=left, op=op, right=right)
    
    def handle_expr(self,expr_node):
        """
        Recursively handle and convert Fortran expression nodes into Python AST nodes.

        This function serves as a dispatcher that processes various types of
        expression nodes from a parsed Fortran AST (using F23) and converts them into
        corresponding Python `ast` nodes for code generation or analysis.

        Parameters
        ----------
        expr_node : object
            A node representing an expression from the Fortran AST.
            Can be one of several F23 classes such as `Real_Literal_Constant`,
            `Part_Ref`, `Intrinsic_Function_Reference`, `Level_2_Expr`, etc.

        Returns
        -------
        ast.AST
            A Python AST node representing the translated Fortran expression.
            Types can include `ast.Constant`, `ast.Name`, `ast.BinOp`, `ast.BoolOp`,
            `ast.UnaryOp`, or `ast.keyword`, depending on the type of the input node.

        Raises
        ------
        NotImplementedError
            If the given `expr_node` is of a type that is not yet supported,
            or if specific cases within supported types are not implemented.

        Notes
        -----
        This method handles:
            - Literal constants (integers, reals, logicals, characters)
            - Binary operations (e.g., `+`, `-`, `*`, logical `AND`, `OR`)
            - Unary operations (`+`, `-`)
            - Intrinsic function references (e.g., `MINLOC`, `MAXLOC`)
            - Nested expressions and parenthetical groupings
            - Named references and argument specifications
        """
        
        if isinstance(expr_node, F23.Real_Literal_Constant):
            return self.handle_real_literal_constant(None, expr_node)

        elif isinstance(expr_node, F23.Part_Ref):
            return self.handle_part_ref(None,expr_node)

        elif isinstance(expr_node, F23.Intrinsic_Function_Reference):
            return self.handle_intrinsic_function_reference(None,expr_node)

        elif isinstance(expr_node, F23.Level_2_Expr):  # Composite expression, which contains tuples of different other expressions 
            if isinstance(expr_node.items[0], F23.Intrinsic_Function_Reference) and expr_node.items[0].items[0].string in ['MINLOC', 'MAXLOC']:
                return self.handle_expr(expr_node.items[0])
            else:
                left = self.handle_expr(expr_node.items[0])
                op_token = expr_node.items[1] 
                right = self.handle_expr(expr_node.items[2])
                return self.build_binop(left, op_token, right)

        elif isinstance(expr_node, F23.Add_Operand):
            return self.handle_expr(expr_node.items)

        elif isinstance(expr_node, tuple): # THese are mostly used for the assignement task used inside the intrinsic arg spec list

            if len(expr_node) == 1:
                return self.handle_expr(expr_node[0])

            elif len(expr_node) == 3:
                left, op_token, right = expr_node
                # print(right)
                left_ast = self.handle_expr(left)
                right_ast = self.handle_expr(right)
                return self.build_binop(left_ast, op_token, right_ast)
            else:
                raise NotImplementedError(f'Not implemented for tuple with a size not equal to 1 or 3')

        elif isinstance(expr_node,F23.Int_Literal_Constant):
            return ast.Constant(value = int(expr_node.string))

        elif isinstance(expr_node, F23.Level_2_Unary_Expr):
            op_token, operand_node = expr_node.children
            operand_ast = self.handle_expr(operand_node)  
        
            op_map = {
                '-': ast.USub(),
                '+': ast.UAdd(),
            }
        
            op = op_map.get(op_token)
            if not op:
                raise NotImplementedError(f"Unary operator {op_token} not supported.")
        
            return ast.UnaryOp(op=op, operand=operand_ast)

        elif isinstance(expr_node,F23.Level_4_Expr):
            return self.handle_level_4epr(expr_node)
        
        elif isinstance(expr_node,F23.Parenthesis):
            return self.handle_expr(expr_node.items[1]) # we directly send the element inside the paranthesis 
            
        elif isinstance(expr_node, F23.Name):
            return ast.Name(id=expr_node.string, ctx=ast.Load())

        elif isinstance(expr_node,F23.Mult_Operand):
            left_ast = self.handle_expr(expr_node.items[0])
            op_token = expr_node.items[1] 
            right_ast = self.handle_expr(expr_node.items[2])
            return self.build_binop(left_ast, op_token, right_ast)
            
        elif isinstance(expr_node,(F23.And_Operand,F23.Or_Operand)):
            return self.handle_OR_AND_Operand(expr_node)

        elif isinstance(expr_node, F23.Equiv_Operand):
            left = self.handle_expr(expr_node.items[0])
            op_token = expr_node.items[1] 
            op_str = op_token.strip().strip('.')  
            op_map = {
                'AND': ast.And(),
                'OR': ast.Or(),
            }            
            op = op_map.get(op_str.upper())
            right = self.handle_expr(expr_node.items[2])
            values = [left,right]
            return ast.BoolOp(op=op,values = values)
            
        elif isinstance(expr_node,F23.Logical_Literal_Constant):
            bool_val,_ = expr_node.children
            return ast.Constant(value = False if bool_val.strip().strip('.')   == "FALSE" else True)

        elif isinstance(expr_node, (F23.Char_Literal_Constant,F23.Int_Literal_Constant)):
            expr_node = expr_node.string.strip(" ' ")
            return ast.Constant(value = expr_node)

        elif isinstance(expr_node,F23.Actual_Arg_Spec): # THese sometimes corresponds to eleemnts from the 
            # of instrinsic methods inner variables values 
            if len(expr_node.children) > 1:
                name, dim = expr_node.children
                if name.string.lower() == "dim" and isinstance(dim,F23.Int_Literal_Constant):# THis case is valid only eleements
                    # that use the axis argument but need to handle in which we might not need this but something else such as where etc... 
                    # thus requires a verification in amont of before translating this
                    if isinstance(dim, F23.Int_Literal_Constant):
                        value = int(dim.children[0]) - 1
                        return ast.keyword(arg='axis',value = ast.Constant(value = value ))
                    else:
                        raise NotImplementedError(f'The axis value for DIM is not implemeneted for :{type(dim)}')
                else:
                    if "=" in expr_node.tostr() and len(expr_node.children) == 2:
                        rhs_ast = self.handle_expr(expr_node.children[1])
                        stmt = ast.keyword(
                            arg = expr_node.children[0].string,
                            value = rhs_ast
                        )
                        return stmt
                    else:
                        raise NotImplementedError(f"not implemented handle_expr: actual_arg_spec for the expression_node:{expr_node}")
                    
        elif isinstance(expr_node,(F23.Write_Stmt,F23.Print_Stmt)):
            if not any(walk(walk(expr_node,F23.Io_Control_Spec),F23.Int_Literal_Constant)):
                stmt = self.handle_print_stmt(expr_node)
                return stmt
        
        elif isinstance(expr_node, F23.Call_Stmt):
            return self.handle_call_stmt(expr_node)
        
        elif isinstance(expr_node,F23.Subscript_Triplet):
            shape = []
            limits = expr_node.tostr().split(':')
            lb = limits[0]
            if len(limits) > 1:
                ub = limits[1]
                if lb:
                    lb = lb + '-1'
                lb = self.simplify_limits(lb)
                ub = self.simplify_limits(ub)

                shape.append((f"{lb}:{ub}",expr_node))
            elif len(limits)==1:
                shape.append((f"{lb}",expr_node))
            args = []
            for sh,node in shape:
                if ':' in sh and isinstance(node, F23.Subscript_Triplet):
                    # It's a slice
                    if sh == ':':
                        # Simple ':' slice
                        slice_node = ast.Slice()
                    else:
                        # Possibly lb:ub
                        lb_ub = sh.split(':')
                        lb = lb_ub[0].strip() or None
                        ub = lb_ub[1].strip() if len(lb_ub) > 1 else None
                        # print(ast.dump(self.handle_expr(node.children[1])))
                        slice_node = ast.Slice(
                            lower=self.handle_expr(node.children[0]) if lb else None,
                            upper=self.handle_expr(node.children[1]) if ub else None
                        )
                    args.append(slice_node)
                else:
                    # it's a direct index
                    if isinstance(node,ast.AST):
                        expr_node = node
                    else:
                        expr_node = self.handle_expr(node)
                    args.append(expr_node)
            
            return args 
        
        elif isinstance(expr_node,F23.Assignment_Stmt):
            lhs = self.handle_expr(expr_node.children[0])
            rhs = self.handle_expr(expr_node.children[2])
            return ast.Assign(
                targets = lhs,
                value=rhs
            )

        else:
            raise NotImplementedError(f"Unsupported node type: {type(expr_node)}, for node:{expr_node}")
        
    def apply_mask_to_rhs(self, node):
        """
        Recursively walk the RHS and apply `[mask]` to any variable or subscript
        if it's listed in `self.extractor.all_array_info`.

        Example transformations:
            array     -> array[mask]
            array[:]  -> array[:][mask]
        """
        
        # Handle variable names like: array
        if isinstance(node, ast.Name):
            for elements in self.extractor.all_array_info.values():
                if node.id in elements.keys():
                    return ast.Subscript(
                        value=ast.Name(id=node.id, ctx=ast.Load()),
                        slice=ast.Name(id="mask", ctx=ast.Load()),
                        ctx=ast.Load()
                    )

        # Handle subscript access like: array[:]
        elif isinstance(node, ast.Subscript):
            base = node.value
            if isinstance(base, ast.Name):
                for elements in self.extractor.all_array_info.values():
                    if base.id in elements.keys():
                        # Recursively apply masking to subscript slice
                        node.slice = self.apply_mask_to_rhs(node.slice)

                        # Case 1: It's a full slice like [:]
                        if isinstance(node.slice, ast.Slice):
                            return ast.Subscript(
                                value=node,
                                slice=ast.Name(id="mask", ctx=ast.Load()),
                                ctx=ast.Load()
                            )

                        # Case 2: It's a tuple like [:, 1]
                        elif isinstance(node.slice, ast.Tuple):
                            # Only apply mask if any element in the tuple is a slice
                            if any(isinstance(elt, ast.Slice) for elt in node.slice.elts):
                                return ast.Subscript(
                                    value=node,
                                    slice=ast.Name(id="mask", ctx=ast.Load()),
                                    ctx=ast.Load()
                                )

                        else:
                            return node

        # Optionally handle attribute access like obj.attr (if you want to mask those too)
        elif isinstance(node, ast.Attribute):
            # Check if this attribute is a known array (only if it's in tracking attributes in all_array_info)
            attr_str = node.attr  # For example: "obj.attr"
            for elements in self.extractor.all_array_info.values():
                if attr_str in elements.keys():
                    return ast.Subscript(
                        value=node,
                        slice=ast.Name(id="mask", ctx=ast.Load()),
                        ctx=ast.Load()
                    )

        # Recursively walk all child nodes
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                new_values = []
                for item in value:
                    if isinstance(item, ast.AST):
                        new_values.append(self.apply_mask_to_rhs(item))
                    else:
                        new_values.append(item)
                setattr(node, field, new_values)
            elif isinstance(value, ast.AST):
                setattr(node, field, self.apply_mask_to_rhs(value))

        return node