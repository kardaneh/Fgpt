import os
import ast
import logging
import operator
from typing import Generator,Dict,List
from collections import defaultdict, deque

# We will use the nodeTransformer https://docs.python.org/3/library/ast.html#ast.NodeTransformer to create a general purpose
# class in charge of visiting nodes and replacing them with a global class instance, based on these examples : 
# https://elshad-karimov.medium.com/unpacking-pythons-ast-module-writing-a-python-code-transformer-eef528b0ed15 
# https://jason-manuel.com/2021/08/16/working-with-pythons-abstract-syntax-trees/ and the previous function type approach to identify and 
# replace. 
class ReplaceGlobals(ast.NodeTransformer):
    def __init__(self, cls_info):
        self.cls_info = cls_info

    def get_attr_node(self, name):
        for _ , instances in self.cls_info.items():
            for inst_key, value in instances.items():
                cls_attr = value.get("attributes", [])
                other_object_instances = value.get("instances",[])
                
                if name in cls_attr:
                    return ast.Attribute(
                        value=ast.Name(id=inst_key, ctx=ast.Load()),
                        attr=name,
                        ctx=ast.Load()
                    )
                elif other_object_instances:# This is to handle the cases on which we have other classes intialized inside the class itself
                    # thus requires to intialized with the attributes of the class 
                    for key in list(other_object_instances.keys()):
                        
                        other_object_attributes = other_object_instances[key].get('attributes')
                        if name in other_object_attributes:
                            return ast.Attribute(
                                value=other_object_instances[key]['class_name'],
                                attr=name,
                                ctx=ast.Load()
                            )
                    
        return None

    def visit_Name(self, node):
        replacement = self.get_attr_node(node.id)
        if replacement:
            return replacement
        return node
    
    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        # Only handle self.xxx form
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            name = node.attr

            # Check if it's a local attribute
            for _, instances in self.cls_info.items():
                for _, value in instances.items():
                    local_attrs = value.get("attributes", [])
                    other_instances = value.get("instances", [])

                    if name in local_attrs:
                        return node  # keep as self.name

                    for key in other_instances.keys():
                        other_object_attributes = other_instances[key].get("attributes", {})
                        if name in other_object_attributes:
                            # Build self.instance.name
                            return ast.Attribute(
                                value=other_instances[key]['class_name'],
                                attr=name,
                                ctx=ast.Load()
                            )

        return node

    def visit_List(self, node):
        node.elts = [self.visit(elt) for elt in node.elts]
        return node
    
    def visit_Assign(self, node):
        node = self.generic_visit(node)  # recursively visit all the children nodes
        for i, target in enumerate(node.targets):
            if isinstance(target, ast.Name):
                replacement = self.get_attr_node(target.id)
                if replacement:
                    node.targets[i] = replacement
            if isinstance(target,ast.Subscript):
                if isinstance(target.value, ast.Name):
                    replacement = self.get_attr_node(target.value.id)
                    if replacement:
                        node.targets[i].value = replacement
        return node

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            method_name = node.value.func.id

            for _ , instances in self.cls_info.items():
                for inst_key, value in instances.items():
                    methods = value.get('methods',[])
                    other_object_instances = value.get("instances",[])
                    if method_name in methods:
                        node.value.func = ast.Attribute(
                                            value=ast.Name(id=inst_key, ctx=ast.Load()),
                                            attr=method_name,
                                            ctx=ast.Load()
                                        )
                        return node
                    elif other_object_instances:# This is to handle the cases on which we have other classes intialized inside the class itself 
                        # and require to be attributed to the intialized class 
                        for key in list(other_object_instances.keys()):
                            
                            other_object_attributes = other_object_instances[key].get('methods')
                            if method_name in other_object_attributes:
                                node.value.func =  ast.Attribute(
                                    value=other_object_instances[key]['class_name'],
                                    attr=method_name,
                                    ctx=ast.Load()
                                )
                            return node 
        return node

    def visit_For(self,node):
        node = self.generic_visit(node)
        if isinstance(node,ast.For):
            if isinstance(node.iter, ast.Call):
                # Retrieve the args and see if one them is dependant on the global attribute
                for i, arg in enumerate(node.iter.args):
                    if isinstance(arg,ast.Name):
                        replacement = self.get_attr_node(arg.id)
                        if replacement:
                            node.iter.args[i] = replacement
        return node

    def visit_If(self, node):
        node = self.generic_visit(node) 

        if isinstance(node.test, ast.Name): # These primarily reference the cases of logical cases
            replacement = self.get_attr_node(node.test.id)
            if replacement:
                node.test = replacement

        # In the if statement there could be a simple comparasions test or either a complex bool operation(and/or) cases which needs to 
        # be handled.
        
        # Handle comparisons: a < i
        elif isinstance(node.test, ast.Compare):
            self._replace_compare(node.test)
    
        # Handle boolean operations: a < i and i > b
        elif isinstance(node.test, ast.BoolOp):
            for i, value in enumerate(node.test.values):
                if isinstance(value, ast.Compare):
                    self._replace_compare(value)
                elif isinstance(value, ast.Name):
                    replacement = self.get_attr_node(value.id)
                    if replacement:
                        node.test.values[i] = replacement
                elif isinstance(value, ast.Subscript):
                    if isinstance(value.value, ast.Name):
                        replacement = self.get_attr_node(value.value.id)
                        if replacement:
                            value.value = replacement
    
        return node

    def _replace_compare(self, compare_node):
        # THe compare node contains a left and right where on the right side in some case a just values or elements(arrays,binop) to be compared upon
        # Left side
        if isinstance(compare_node.left, ast.Name):
            replacement = self.get_attr_node(compare_node.left.id)
            if replacement:
                compare_node.left = replacement
    
        elif isinstance(compare_node.left, ast.Subscript):
            if isinstance(compare_node.left.value, ast.Name):
                replacement = self.get_attr_node(compare_node.left.value.id)
                if replacement:
                    compare_node.left.value = replacement
    
        # Comparators (right side)
        for i, comp in enumerate(compare_node.comparators):
            if isinstance(comp, ast.Name):
                replacement = self.get_attr_node(comp.id)
                if replacement:
                    compare_node.comparators[i] = replacement
    
            elif isinstance(comp, ast.Subscript):
                if isinstance(comp.value, ast.Name):
                    replacement = self.get_attr_node(comp.value.id)
                    if replacement:
                        comp.value = replacement
    
            elif isinstance(comp, ast.BinOp):
                compare_node.comparators[i] = self.visit_BinOp(comp)

    def visit_BinOp(self, node):
        self.generic_visit(node)  
        
        # Left side
        if isinstance(node.left, ast.Subscript):
            if hasattr(node.left, "value") and isinstance(node.left.value, ast.Name):
                replacement = self.get_attr_node(node.left.value.id)
                if replacement:
                        node.left.value = replacement
                        
        elif isinstance(node.left, ast.Name):
            replacement = self.get_attr_node(node.left.id)
            if replacement:
                node.left = replacement

        # Right side
        if isinstance(node.right, ast.Subscript):
            if hasattr(node.right, "value") and isinstance(node.right.value, ast.Name):
                replacement = self.get_attr_node(node.right.value.id)
                if replacement:
                        node.right.value = replacement
                        
        elif isinstance(node.right, ast.Name):
            replacement = self.get_attr_node(node.right.id)
            if replacement:
                node.left = replacement
                
        elif isinstance(node.right, ast.BinOp):
            self.visit_BinOp(node.right)

        return node

 # SemanticIndexAdjuster
class AdjustIndices(ast.NodeTransformer):
    def __init__(self,conv_vars,attributes:Dict,global_attributes:Dict,**kwargs):
        self.CONV_VARS = conv_vars # This corresponds to the conventional loop variables such as ji,jst,jl etc... 
        self.attribute_info = attributes
        self.global_attributes = global_attributes.get("attributes", {})
        self.instances_global_attributes = global_attributes.get("instances", {})
        self.adjusted_vars = set()

        self.exclude_index = kwargs.get("exclude_index")

    def visit_Subscript(self, node):
        self.generic_visit(node)
        
        arr_name = None
        if isinstance(node.value, ast.Name):
            arr_name = node.value.id
        elif isinstance(node.value, ast.Attribute):
            arr_name = node.value.attr

        # Adjust only if dim_str == 1, which reflects the default lower bound in FORTRAN.
        # Arrays that do not follow this convention should be verified before applying any transformation,
        # to avoid incorrect conversions. In general, loop variables from conventional FORTRAN loops
        # don't require changes, as they're already adapted for Python. However, if an array has a non-default
        # lower bound (not starting at 1), the corresponding loop variable must be corrected to
        # account for the offset between FORTRAN and Python indexing, thus recovering to the original Fortran index 
        if arr_name and arr_name in self.attribute_info:
            dims_info = self.attribute_info[arr_name]
            if isinstance(node.slice, ast.Tuple):
                new_elts = []
                for i, elt in enumerate(node.slice.elts):
                    dim_info_str = dims_info[i].get("dim_str")
                    # Case 1: dim_str is a digit
                    if dim_info_str.isdigit():
                        if dim_info_str == "1":
                            new_elts.append(self._adjust_index(elt))
                        else:
                            offset = 1 - int(dim_info_str)
                            new_elts.append(self._apply_offset_if_convvar(elt, offset))
                
                    # Case 2: dim_str is a variable (needs global_attributes lookup or if it's present inside one of the composition class)
                    else:
                        # resolve via global attributes or the perhaps either in one of the composition classes if such instance is present
                        resolved_str = ""
                        dimension = self.global_attributes.get(dim_info_str)

                        if dimension is not None:
                            resolved_str = str(dimension[0])
                        elif self.instances_global_attributes:
                            for key in list(self.instances_global_attributes.keys()):
                                instance_attributes = self.instances_global_attributes[key].get("attributes")
                                if instance_attributes:
                                    dimension = instance_attributes.get(dim_info_str)
                                    if dimension:
                                        resolved_str = str(dimension[0])

                        if resolved_str == "1":
                            new_elts.append(self._adjust_index(elt))
                        elif resolved_str:
                            offset = 1 - int(resolved_str)
                            new_elts.append(self._apply_offset_if_convvar(elt, offset))
                        else:
                            raise KeyError(f"Could not resolve dimension info for: {dim_info_str}")
                        
                node.slice.elts = new_elts
            else:
                # Single-dimension array
                dim_info_str = dims_info[0].get("dim_str")
                # print(ast.unparse(ast.fix_missing_locations(node)))
                if dim_info_str.isdigit():
                    if dim_info_str == "1":
                        node.slice = self._adjust_index(node.slice)
                    else:
                        offset = 1 - int(dim_info_str)
                        node.slice = self._apply_offset_if_convvar(node.slice, offset)
                
                # Case 2: dim_str is a variable (needs global_attributes lookup)
                else:
                    resolved_str = ""
                    dimension = self.global_attributes.get(dim_info_str)

                    if dimension is not None:
                        resolved_str = str(dimension[0])
                    elif self.instances_global_attributes:
                        for key in list(self.instances_global_attributes.keys()):
                            instance_attributes = self.instances_global_attributes[key].get("attributes")
                            if instance_attributes:
                                dimension = instance_attributes.get(dim_info_str)
                                if dimension:
                                    resolved_str = str(dimension[0])
                
                    if resolved_str == "1":
                        node.slice = self._adjust_index(node.slice)
                    elif resolved_str:
                        offset = 1 - int(resolved_str)
                        node.slice = self._apply_offset_if_convvar(node.slice, offset)
                    else:
                        raise KeyError(f"Could not resolve dimension info for: {dim_info_str}")
                    
        else:
            # fallback: adjust everything if we don’t know the array
            if isinstance(node.slice, ast.Tuple):
                node.slice.elts = [self._adjust_index(elt) for elt in node.slice.elts]
            else:
                node.slice = self._adjust_index(node.slice)
    
        return node

    def visit_Assign(self, node):
        # print(f'Before visit_Assign: {ast.unparse(ast.fix_missing_locations(node))}')
        
        node.value = self.visit(node.value)
        new_targets = [self.visit(t) for t in node.targets] 
    
        node.targets = new_targets
        # This is to handle targets that are conventional variables that are used as indices inside the arrays but who get
        # their values assigned and doesn't come from for loops ex :jsl = value, but also target values which has compare in them this is to ensure that the mask type arrays don't get substracted
        if isinstance(node.targets[0], (ast.Name,ast.Attribute)):
            name = node.targets[0].id if isinstance(node.targets[0],ast.Name) else node.targets[0].attr
            if name in self.CONV_VARS: # THIS IS to modify in the case of CONV_vars ARE in the left hand side 
                node.value = self._adjust_assignment_rhs(node.value)
            # Check if the right hand assigement is that of conv vars 
            if isinstance(node.value, (ast.Name,ast.Attribute)): # This is just to ensure that the in the case elements of conv_vars get's reassigned to another variable thus doesn't require modification 
                value_name = node.value.id if isinstance(node.value, ast.Name) else node.value.attr # RHS 
                if value_name in self.CONV_VARS:
                    self.adjusted_vars.add(name)
            elif isinstance(node.value, ast.Compare):
                self.adjusted_vars.add(name)

        return node

    def visit_Call(self,node):
        try:
            self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr in ['argmin','argmax']:
                # Check for the subscript node inside
                subscript_nodes = ast_walk(node,ast.Subscript)
                if subscript_nodes:
                    
                    subscript_node = next(iter(subscript_nodes))
                    arr_name = subscript_node.value.id if isinstance(subscript_node.value, ast.Name) else subscript_node.value.attr
                    
                    dim_info = self.attribute_info[arr_name]
                    if dim_info and len(dim_info) == 1:
                        if dim_info[0]['dim_str'] != "0": # If the lower bound is 0 this doesnt' require the creation of BinOp
    
                            return ast.BinOp( # lb + min_idx for python and lb + min_idx - 1 for fortran
                                left = node,
                                op = ast.Add(),
                                right = ast.Constant(value = int(dim_info[0]['dim_str']))
                            )
                    elif dim_info and len(dim_info) > 1:
                        # First we need to retrieve the dim info based on where the SLICE is
                        slices = getattr(subscript_node.slice, "elts", [])
                        slice_positions = [i for i, elem in enumerate(slices) if isinstance(elem, ast.Slice)]
                        if len(slice_positions) == 1: # MEans only one slice in the multi dimensional array
                            lb = dim_info[slice_positions[0]]['dim_str']
                            if lb != "0":
                                return ast.BinOp( 
                                    left = node,
                                    op = ast.Add(),
                                    right = ast.Constant(value = int(lb))
                                )
                        else: # THis is in the case we have more than one slice thus requires the axis keyword
                            # Here we need to check the axis value to ensure that we pick the correct slice
                            axis_value = None
                            if isinstance(node.keywords[0].value, ast.Constant):
                                axis_value = node.keywords[0].value.value + 1
                            
                            lb = dim_info[slice_positions[axis_value]]['dim_str']
                            if lb != "0":
                                return ast.BinOp( 
                                    left = node,
                                    op = ast.Add(),
                                    right = ast.Constant(value = int(lb))
                                )                        
            return node 
        except Exception as e:
            raise RuntimeError(f'RuntimeERROR in visit_Call of AdjustIndices: {ast.dump(node,indent=4)}') from e
    
    def _apply_offset_if_convvar(self, node, offset):
        # Fortran arrays can have arbitrary lower bounds (0..n, 1..n, -3..n) and since python arrays always start at 0. When converting Fortran code to Python,
        # we adjust variable references by applying an offset so that Python accesses
        # the same array elements and produces the same values as the original Fortran code, despite the difference in array lower bounds.
        if isinstance(node, ast.Name) and node.id in self.CONV_VARS:
            node = ast.BinOp(left = node, op = ast.Add(), right = ast.Constant(value=offset))
            
        elif isinstance(node, ast.BinOp) and isinstance(node.left, ast.Name) and node.left.id in self.CONV_VARS:
            left = node.left
            right = node.right
            op = node.op
                
            if isinstance(right, ast.Constant):
                original_value = right.value
                
                # i - N
                if isinstance(op, ast.Sub):
                    new_value = original_value - offset  # i - N - offset
                    if new_value == 0:
                        node = left  # i - N + N -> i
                    else:
                        node =  ast.BinOp(
                                    left=left,
                                    op=ast.Sub(),
                                    right=ast.Constant(value=new_value)
                                )
                
                # i + N
                elif isinstance(op, ast.Add):
                    new_value = original_value + offset  # i + N + offset
                    if new_value == 0:
                        node = left  # i + N - N -> i
                    else:
                        node = ast.BinOp(
                                    left=left,
                                    op=ast.Add(),
                                    right=ast.Constant(value=new_value)
                                )
                                    
        return node

    def _adjust_index(self, index_node):
        try:
            if isinstance(index_node, ast.Name):
                if index_node.id not in self.CONV_VARS and index_node.id not in self.adjusted_vars:
                    # Need a checker method that checks if one of the varaibles has either been affected is that of int type 
                    # thus requires us to modify it directly 
                    if (not self.exclude_index) or (index_node.id not in self.exclude_index):
                        return self._subtract_one(index_node)
    
            elif isinstance(index_node, ast.BinOp):
                # print(f' adjust_index, binop:{ast.unparse(ast.fix_missing_locations(index_node))}')
                return self._handle_binop(index_node)
    
            elif isinstance(index_node, ast.Subscript): 
                # print(f'_adjust_index:{ast.unparse(ast.fix_missing_locations(index_node))}')
                return self.visit_Subscript(index_node)
    
            elif isinstance(index_node,ast.Constant):
                return ast.Constant(value=index_node.value - 1)
    
            elif isinstance(index_node,ast.Attribute):
                if index_node.attr not in self.CONV_VARS and index_node.attr not in self.adjusted_vars:
                    if (not self.exclude_index) or (index_node.attr not in self.exclude_index):
                        return self._subtract_one(index_node)
                    # return self._subtract_one(index_node)
                    
            elif isinstance(index_node, ast.Call):
                return ast.BinOp(
                    left = index_node,
                    op = ast.Sub(),
                    right = ast.Constant(value=1)
                )
    
            return index_node
        except Exception as e:
            raise RuntimeError(f"_adjust_index failed for node={ast.dump(index_node,indent=4)}") from e

    # This is for right hand side assignement handling 
    def _adjust_assignment_rhs(self, rhs):
        try:
            # print(f'inside _adjust_assignment_rhs: {ast.unparse(ast.fix_missing_locations(rhs))}')
            if isinstance(rhs, ast.Name) and rhs.id not in self.CONV_VARS:
                return self._subtract_one(rhs)
                
            elif isinstance(rhs, ast.Subscript):
                return self._subtract_one(rhs)
    
            elif isinstance(rhs, ast.BinOp):
                if isinstance(rhs.op, ast.Sub):
                    return ast.BinOp(left=rhs.left, op=ast.Sub(), right=ast.Constant(value=rhs.right.value + 1))
    
                elif isinstance(rhs.op, ast.Add):
                    if rhs.right.value == 1:
                        return rhs.left
                    else:
                        return ast.BinOp(
                            left=rhs.left,
                            op=ast.Add(),
                            right=ast.Constant(value=rhs.right.value - 1)
                        )
    
            return rhs
        except Exception as e:
            raise RuntimeError(f'_adjust_assignment_rhs failed for node={ast.dump(rhs,indent=4)}') from e

    def _subtract_one(self, node):
        return ast.BinOp(
            left=node,
            op=ast.Sub(),
            right=ast.Constant(value=1)
        )

    def _handle_binop(self, node):
        # print(f'handle_binop:{ast.unparse(ast.fix_missing_locations(node))}')
        try:
            self.generic_visit(node)
            if (isinstance(node.left, ast.Name) and node.left.id not in self.CONV_VARS and node.left.id not in self.adjusted_vars and isinstance(node.right, ast.Constant)):
                if (not self.exclude_index) or (node.left.id not in self.exclude_index):
                    if isinstance(node.op, ast.Sub):
                        # i - 1 -> i - 2
                        return ast.BinOp(
                            left=node.left,
                            op=ast.Sub(),
                            right=ast.Constant(value=node.right.value + 1)
                        )
                    elif isinstance(node.op, ast.Add):
                        # i + 2 -> i + 1 or i + 1 -> i 
                        if node.right.value == 1:
                            return node.left
                        else:
                            return ast.BinOp(
                                left=node.left,
                                op=ast.Add(),
                                right=ast.Constant(value=node.right.value - 1)
                            )     
            return node
        except Exception as e:
            raise RuntimeError(f'_handle_binop failed for node={ast.dump(node,indent=4)}') from e


def get_instance_name(node_name):
    clean_name = '_'.join(filter(None, node_name.strip().split('_')))
    split_name = clean_name.split('_')

    if len(split_name) == 1:
        # Format: processor, isolator
        instance_name = split_name[0].lower()
    elif len(split_name) == 2:
        # Format: global_module
        instance_name = split_name[0].lower()[0] + split_name[1].lower()[0]
    else: # This means we have a name that perhaps have mutliple _ which is quiet not good to begin with
        # Use first letter of each part up to create an abbreviation 
        instance_name = ''.join(part.lower()[0] for part in split_name)
    
    return instance_name

def identify_replace_all(ast_list: list, cls_info: dict):
    try:
        transformer = ReplaceGlobals(cls_info)
        for i, node in enumerate(ast_list):
            ast_list[i] = transformer.visit(node)
    except Exception as e:
        logging.error(f'Error in identify_replace_all: {e}')
        raise 


def ast_walk(node, node_type: ast.AST) -> Generator:
    """
    Recursively walk an AST tree, yielding all nodes or nodes of a specific type.

    Parameters
    ----------
    node : ast.AST
        The root AST node to walk through.
    node_type : type or None
        The specific AST node type to filter by. If None, yields all nodes.

    Yields
    ------
    ast.AST or None
        AST nodes matching the specified type. Yields None if an exception occurs.

    Notes
    -----
    The function recursively traverses the AST, yielding nodes of the specified
    type. If `node_type` is None, all nodes are yielded.
    """
    try:
        if node_type is None or isinstance(node, node_type):
            yield node
        for child in ast.iter_child_nodes(node):
            yield from ast_walk(child, node_type)
    except Exception:
        logging.exception(f'Exception in ast_walk')
        yield None

def find_folder(root_dir, target_folder):
    for dirpath, dirnames, _ in os.walk(root_dir):
        if target_folder in dirnames:
            return os.path.join(dirpath, target_folder)
    return None

def safe_eval_expr(node):
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = safe_eval_expr(node.left)
        right = safe_eval_expr(node.right)
        
        # Supported operators
        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
        }

        op_type = type(node.op)
        if op_type in ops:
            return ops[op_type](left, right)
        else:
            raise NotImplementedError(f"Operator {op_type} not supported.")
    else:
        raise NotImplementedError(f"Unsupported AST node type: {type(node)}")

def update_dict(primary_dict, secondary_dict):
    """
    Automatically update the `instances` section of the primary_dict by scanning
    for composed classes (composition) and fetching their attributes and methods
    from the secondary_dict.
    """

    for _, classes in primary_dict.items():
        for _, class_content in classes.items():
            primary_attrs = class_content.get("attributes", {})
            instances = class_content.get("instances", {})

            # For each instance in the primary class, check if it's a composed class
            for instance_name, instance_data in instances.items():
                # Now we must find the class definition of this instance in the secondary_dict
                for _, secondary_classes in secondary_dict.items():
                    if instance_name in secondary_classes:
                        composed_class = secondary_classes[instance_name]
                        secondary_attrs = composed_class.get("attributes", {})
                        secondary_methods = composed_class.get("methods", {})

                        # Prepare instance sub-structure # composition type
                        instance_attrs = instance_data.setdefault("attributes", {})
                        instance_methods = instance_data.setdefault("methods", {})

                        # Add attributes only if not already present in class-level attributes
                        for attr_name, attr_val in secondary_attrs.items():
                            if attr_name not in primary_attrs:
                                instance_attrs.setdefault(attr_name, attr_val)

                        # Add methods only if not already present in class-level methods
                        for method_name, method_val in secondary_methods.items():
                            instance_methods.setdefault(method_name, method_val)

    return primary_dict


def collect_dependencies(node):
    """Collect all variable names this value depends on."""
    deps = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            deps.add(child.id)
        elif isinstance(child, ast.Attribute):
            deps.add(child.attr)
    return deps

def order_assignments(assign_nodes:List, diff:List) -> List:
    """
    Works similiarily to the search_dependant_variabels of Transformer class with the addition of creating a sorted values in which it's 
    ordered using the topological sort using Kahn's algorithm to ensure proper valid order. 
    """
    # Build dependency graph
    graph = defaultdict(set)
    indegree = defaultdict(int)
    
    for assign in assign_nodes:
        if not assign.targets:
            continue
        target = assign.targets[0]
        if isinstance(target, ast.Name):
            name = target.id
        elif isinstance(target, ast.Attribute):
            name = target.attr
        else:
            continue
        
        if diff and name not in diff:
            continue
        
        if diff:
            deps = collect_dependencies(assign.value) & set(diff)
        else:
            deps = collect_dependencies(assign.value)
        graph[name].update(deps) # Create the dependecies graph for attributes mostly between scalars with intialized values 

    # Compute indegrees
    for var, deps in graph.items():
        for _ in deps:
            indegree[var] += 1

    # THe graphs in our case is the dependant dependeee type elements https://www.interviewcake.com/concept/java/topological-sort, thus we first go from the dependedant 
    # then the dependee 
    # Kahn’s algorithm for topological sort code found here: https://www.geeksforgeeks.org/dsa/topological-sorting-indegree-based-solution/ 
    queue = deque([v for v in graph if indegree[v] == 0])
    result = []

    while queue:
        v = queue.popleft()
        result.append(v)
        for u in graph:
            if v in graph[u]:
                indegree[u] -= 1
                if indegree[u] == 0:
                    queue.append(u)

    # If cycles remain, just append the rest arbitrarily
    for v in graph:
        if v not in result:
            result.append(v)

    return result
