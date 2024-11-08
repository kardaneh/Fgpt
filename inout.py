from collections import defaultdict
from fparser.two.utils import walk
from fparser.two import Fortran2003 as F23

def traverse_execution_part(subroutine_tree, dummy_arg_list):
    usage = defaultdict(lambda: {'intent': None, 'first_use_parent': None}) 
    def traverse_block(block):
        if hasattr(block, "content"):
            for child in block.content:
                if isinstance(child, F23.Assignment_Stmt):
                    lhs_var = None
                    if isinstance(child.items[0], F23.Name):
                        lhs_var = child.items[0].tostr()
                    elif isinstance(child.items[0], F23.Part_Ref):
                        lhs_var = child.items[0].children[0].tostr()
                    else:
                        raise ValueError(f"Error: Unexpected type for lhs_var.")

                    rhs_expr = child.items[-1]

                    if lhs_var in dummy_arg_list:
                        if usage[lhs_var]['intent'] is None:
                            usage[lhs_var]['intent'] = 'OUT'
                        elif usage[lhs_var]['intent'] == 'IN':
                            usage[lhs_var]['intent'] = 'INOUT'
                        if usage[lhs_var]['first_use_parent'] is None:
                            usage[lhs_var]['first_use_parent'] = child

                    for name in walk(rhs_expr, F23.Name):
                        if isinstance(name.parent, F23.Section_Subscript_List):
                            continue
                        if name.tostr() in dummy_arg_list:
                            var_name = name.tostr().lower()
                            if usage[var_name]['intent'] is None:
                                usage[var_name]['intent'] = 'IN'
                            elif usage[var_name]['intent'] == 'OUT' and usage[var_name]['first_use_parent'] == child:
                                usage[var_name]['intent'] = 'INOUT'
                            if usage[var_name]['first_use_parent'] is None:
                                usage[var_name]['first_use_parent'] = child

                elif isinstance(child, (F23.If_Then_Stmt, F23.Else_If_Stmt)):
                    for name in walk(child, F23.Name):
                        if isinstance(name.parent, F23.Section_Subscript_List):
                            continue
                        if name.tostr() in dummy_arg_list:
                            var_name = name.tostr().lower()
                            if usage[var_name]['intent'] is None:
                                usage[var_name]['intent'] = 'IN'
                            if usage[var_name]['first_use_parent'] is None:
                                usage[var_name]['first_use_parent'] = child

                else:
                    traverse_block(child)

    execution_part = walk(subroutine_tree, F23.Execution_Part)[0]
    traverse_block(execution_part)
    intent_dict = {var: props['intent'] for var, props in usage.items()}
    return intent_dict

'''usage = traverse_execution_part(subroutine_tree, dummy_arg_list)

for var, intent in usage.items():
    print(f"INTENT({intent})::{var}")
'''
