import numpy as np
from fparser.two.utils import walk
from fparser.two import Fortran2003 as F23
from fparser.two import Fortran2008 as F28
from processor import Processor
import re

class F2NP:
    def __init__(self):
        self.result = []
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
                r'\bMIN\b': 'min',
                r'\bMAX\b': 'max',
                r'\bMAXVAL\b': 'np.max',
                r'\bMINVAL\b': 'np.min',
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


    def recursive(self, block):
        if hasattr(block, "content"):
            idx = 0
            while idx < len(block.content):
                child = block.content[idx]
                if isinstance(child, F23.Subroutine_Stmt):
                    child = self.handle_subroutine_stmt(child)
                    self.npcode += f"{self.indentation_level * '    '}{child}\n"
                    self.indentation_level += 1
                elif isinstance(child, F23.Type_Declaration_Stmt):
                    child = self.handle_type_declaration_stmt(child)
                    if child is not None:
                        self.npcode += f"{self.indentation_level * '    '}{child}\n"
                    else:
                        del block.content[idx]
                        continue
                elif isinstance(child, F23.Nonlabel_Do_Stmt):
                    child = self.handle_do_stmt(child)
                    self.npcode += f"{self.indentation_level * '    '}{child}\n"
                    self.indentation_level += 1
                elif isinstance(child, F23.If_Then_Stmt):
                    if walk(child, F23.Part_Ref):
                        child = self.handle_assignment(child)
                    else:
                        child = child.tostr()
                    condition = self.handle_if_condition(child)
                    self.npcode += f"{self.indentation_level * '    '}{condition}\n"
                    self.indentation_level += 1
                elif isinstance(child, (F23.Else_If_Stmt, F23.Else_Stmt)):
                    self.indentation_level -= 1
                    if isinstance(child, F23.Else_If_Stmt) and walk(child, F23.Part_Ref):
                        child = self.handle_assignment(child)
                    else:
                        child = child.tostr()
                    condition = self.handle_if_condition(child)
                    self.npcode += f"{self.indentation_level * '    '}{condition}\n"
                    self.indentation_level += 1
                elif isinstance(child, (F23.End_If_Stmt, F23.End_Do_Stmt, F23.End_Subroutine_Stmt)):
                    self.indentation_level -= 1
                    del block.content[idx]
                    continue
                    #condition = self.handle_end_stmt(child)
                    #self.npcode += f"{self.indentation_level * '    '}{condition}\n"
                elif isinstance(child, F23.Print_Stmt):
                    pychild = self.handle_print_stmt(child)
                    self.npcode += f"{self.indentation_level * '    '}{pychild}\n"
                elif isinstance(child, F23.Assignment_Stmt):
                    pychild = self.handle_assignment(child)
                    self.npcode += f"{self.indentation_level * '    '}{pychild}\n"
                else:
                    self.recursive(child)
                idx += 1

    def handle_subroutine_stmt(self, stmt):
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
        return f"def {subroutine_name}({arg_list}):"

    def handle_call_stmt(self, stmt):
        for child in stmt.children:
            if child is None:
                continue
            if isinstance(child, F23.Name):
                subroutine_name = child.tostr()
            elif isinstance(child, F23.Actual_Arg_Spec_List):
                arg_list = child.tostr()
        return f"{subroutine_name}({arg_list})"

    def handle_type_declaration_stmt(self, stmt):
        var_part = []
        for child in stmt.children:
            if child is None:
                continue
            if isinstance(child, F23.Intrinsic_Type_Spec):
                if child.children[0]=='REAL':
                    dtype = 'float32'
                elif child.children[0]=='INTEGER':
                    dtype = 'int'
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


    def simplify_limits(self, expression):
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

    def handle_end_stmt(self, stmt):
        assert isinstance(stmt, (F23.End_Do_Stmt, F23.End_If_Stmt, F23.End_Function_Stmt, F23.End_Subroutine_Stmt)), (
                f"Unexpected statement type: {type(stmt).__name__}. Expected one of: "
                f"End_Do_Stmt, End_If_Stmt, End_Function_Stmt, End_Subroutine_Stmt.")
        return ''

    def handle_specification(self, stmt):
        print(f"# Specification: {stmt}")
        self.result.append(f"# Specification: {stmt}")

    def handle_where(self, stmt):
        condition = stmt.items[0].string
        print(f"np.where({condition})")
        self.result.append(f"np.where({condition})")

    def handle_do_stmt(self, stmt):
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

    def handle_if_condition(self, condition):
        for fortran_op, python_op in self.replacements.items():
            condition = re.sub(fortran_op, python_op, condition, flags=re.IGNORECASE)
        return condition

    def handle_print_stmt(self, stmt):
        assert isinstance(stmt, F23.Print_Stmt), (
                f"Unexpected statement type: {type(stmt).__name__}. Expected one of: "
                f"Print_Stmt")
        output_item_list = ''
        for child in stmt.children:
            if isinstance(child, F23.Output_Item_List):
                output_item_list = child.tostr()
        return f"print({output_item_list})"

    def handle_intrinsic_function_reference(self, stmt_str, intrinsic_function_reference):
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

    def handle_real_literal_constant(self, stmt_str, real_literal_constant):
        for item in real_literal_constant:
            pre = item.children[1]
            stmt_str = stmt_str.replace(f"_{pre}", '')
        return stmt_str

    def handle_part_ref(self, stmt_str, part_ref):
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

    def handle_assignment(self, stmt):
        stmt_str = stmt.tostr()
        part_ref = walk(stmt, F23.Part_Ref)
        intrinsic_function_reference = walk(stmt, F23.Intrinsic_Function_Reference)
        real_literal_constant = walk(stmt, F23.Real_Literal_Constant) 

        if intrinsic_function_reference:
            stmt_str = self.handle_intrinsic_function_reference(stmt_str, intrinsic_function_reference)

        if real_literal_constant:
            stmt_str = self.handle_real_literal_constant(stmt_str, real_literal_constant)

        if part_ref:
            stmt_str = self.handle_part_ref(stmt_str, part_ref)

        return stmt_str
