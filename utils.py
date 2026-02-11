import os
import ast
import logging
import operator
import jax
import jax.numpy as jnp
from jax import jit, lax
import equinox as eqx
from typing import Generator,Dict,List,Set,Optional,Callable,Any
from dataclasses import dataclass
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
                other_object_instances = value.get("instances",{})

                if name.isupper():
                    name = name.lower()

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

                    if name.isupper():
                        name = name.lower()

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

    def visit_Call(self,node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            method_name = node.func.id
            # First the method name and then check the args 
            for _ , instances in self.cls_info.items():
                for inst_key, value in instances.items():
                    methods = value.get('methods',[])
                    other_object_instances = value.get("instances",[])
                    if method_name in methods:
                        node.func = ast.Attribute(
                                            value=ast.Name(id=inst_key, ctx=ast.Load()),
                                            attr=method_name,
                                            ctx=ast.Load()
                                        )
                        new_args = []
                        for arg in node.args:
                            new_arg = self.visit(arg)
                            new_args.append(new_arg)
                        node.args = new_args

                        return node
                    
                    elif other_object_instances:# This is to handle the cases on which we have other classes intialized inside the class itself 
                        # and require to be attributed to the intialized class 
                        for key in list(other_object_instances.keys()):
                            
                            other_object_attributes = other_object_instances[key].get('methods')
                            if method_name in other_object_attributes:
                                node.func =  ast.Attribute(
                                    value=other_object_instances[key]['class_name'],
                                    attr=method_name,
                                    ctx=ast.Load()
                                )
                                new_args = []
                                for arg in node.args:
                                    new_arg = self.visit(arg)
                                    new_args.append(new_arg)
                                node.args = new_args
                            return node 
            # Among the args check if we have any attributes that might come the clss or the composed classes 

        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "logging"):
            for arg in node.args:
                if isinstance(arg, ast.JoinedStr):
                    new_values = []
                    for value in arg.values:
                        new_value = self.visit(value)  # This will now call visit_FormattedValue
                        new_values.append(new_value)
                    arg.values = new_values

            return node
        
        return self.generic_visit(node)

    def visit_Expr(self, node):
        # print(ast.dump(node,indent=4))
        node.value = self.visit(node.value)
        return node
    
    def visit_FormattedValue(self, node):
        node.value = self.visit(node.value)
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


# This class acts like a semantic index adjuster to ensure that the indices present inside array, elemetns sents as arguments and scalars are corrected 
# to ensure that the python 
# SemanticIndexAdjuster
class AdjustIndices(ast.NodeTransformer):
    def __init__(self,conv_vars,array_info:Dict,cls_attributes:Dict,**kwargs):
        self.CONV_VARS = conv_vars # This corresponds to the conventional loop variables such as ji,jst,jl etc... 
        self.array_info = {k.casefold(): v for k, v in array_info.items()}
        self.cls_attributes = cls_attributes.get("attributes", {}) # Attributes of the class (arrays, sclaras)
        self.instances_global_attributes = cls_attributes.get("instances", {}) # Attributes of probably other object class if present inside teh parent class 
        self.adjusted_vars = kwargs.get("adjusted_vars",set())
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
        if arr_name and arr_name in self.array_info:
            dims_info = self.array_info[arr_name.casefold()]
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
                        dimension = self.cls_attributes.get(dim_info_str)

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
                    dimension = self.cls_attributes.get(dim_info_str)

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
            # fallback: adjust everything if we don’t know the array: which mostly means that it's a functions
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
        # This is also true for variables that have be assigned loop variables makeing it as a indirect reference to it. This could be either that of Name, Attribute with name or Array
        if isinstance(node.targets[0], (ast.Name,ast.Attribute)):
            name = node.targets[0].id if isinstance(node.targets[0],ast.Name) else node.targets[0].attr
            if name in self.CONV_VARS: # THIS IS to modify in the case of CONV_vars ARE in the left hand side 
                node.value = self._adjust_assignment_rhs(node.value)
            # Check if the right hand assigement is that of conv vars 
            elif name in self.adjusted_vars: # This is just to ensure that the in the case elements of conv_vars get's reassigned to another variable thus doesn't require modification 
                # but their 
                node.value = self._adjust_index(node.value)
            
            elif isinstance(node.value, ast.Compare) or name == 'mask': # This is to retrieve all the mask type of elements 
                self.adjusted_vars.add(name)
        
        elif isinstance(node.targets[0], ast.Subscript):
            if isinstance(node.targets[0].value,(ast.Name, ast.Attribute)):
                name = node.targets[0].value.id if isinstance(node.targets[0].value,ast.Name) else node.targets[0].value.attr
                if name in self.adjusted_vars:
                    node.value = self._adjust_index(node.value)
        
        return node

    def visit_For(self, node):
        self.generic_visit(node)

        loop_vars = self._extract_loop_vars(node.target)

        used_vars = set()
        for child in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                used_vars.add(child.id)

        unused = [v for v in loop_vars if v not in used_vars]
        if unused:
            print(f" ⚠️ Unused loop variable(s): {unused}")

            for var in unused:
                self._rename_var_in_target(node.target, var, "_")

        if not isinstance(node.iter, ast.Call) or not hasattr(node.iter, 'args'):
            return node  

        new_args = []
        for arg in node.iter.args:
            new_arg = self._process_arg(arg, node)
            if new_arg:
                new_args.append(new_arg)

        node.iter.args = new_args
        return node

    def _extract_loop_vars(self, target):
        """Extract all variable names from the loop target (handles tuples)."""
        if isinstance(target, ast.Name):
            return [target.id]
        elif isinstance(target, (ast.Tuple, ast.List)): # In the case we have enumerate instead of range
            vars_ = []
            for elt in target.elts:
                vars_.extend(self._extract_loop_vars(elt))
            return vars_
        return []

    def _rename_var_in_target(self, target, old, new):
        """Rename a variable in the loop target."""
        if isinstance(target, ast.Name) and target.id == old:
            target.id = new
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._rename_var_in_target(elt, old, new)

    def _process_arg(self, arg, node):
        """Handles transformation logic for each argument in node.iter.args"""
        if isinstance(arg, ast.BinOp):
            left, right = arg.left, arg.right

            if isinstance(left, ast.Name) and left.id in self.adjusted_vars:
                return self._handle_adjusted_left(arg, node)

            elif isinstance(left, ast.Subscript) and isinstance(left.value, (ast.Name, ast.Attribute)):
                return self._handle_adjusted_left(arg, node)
            
            elif isinstance(left, ast.Name) and left.id not in self.adjusted_vars:
                return arg

            # Case 4: Right is an adjusted variable
            elif isinstance(right, ast.Name) and right.id in self.adjusted_vars:
                raise NotImplementedError("Not implemented yet for adjusted right-hand variable in visit_For.")

            else:
                return arg  
        return arg

    def _handle_adjusted_left(self, binop_node, parent_node):
        """Handles cases where left is an adjusted variable or subscript"""
        right = binop_node.right

        if isinstance(right, ast.Constant) and right.value == 1:
            return binop_node.left
        else:
            return ast.BinOp(
                left=parent_node,
                op=ast.Add(),
                right=ast.Constant(value=1)
            )

    def visit_If(self, node):
        self.generic_visit(node)
        if isinstance(node.test, ast.Compare):
            self._handle_compare(node.test)
        
        if ((not node.body or all(isinstance(n, ast.Pass) for n in node.body)) and not node.orelse):
            # Return None to delete the empty 'if' node entirely
            return None

        return node

    def _handle_compare(self,node):
        # In the comparator, what we are trying to mostly doing is to check if the variables or loop variables might need to be modified
        # FOr example if a loop variables is direclty being compared to Constant, Name, etc.. and since these loop variables are already modified to the python range
        # Which is also called INDIRECT REFERENCE

        if isinstance(node.left, ast.Name) and (node.left.id in self.CONV_VARS or node.left.id in self.adjusted_vars):
            for i in range(len(node.comparators)):
                # NEed to handle the case where the nodes compare themselves but perhaps with a certain index +, for example : loc == loc + 1
                node.comparators[i] = self._adjust_index(node.comparators[i])
        elif isinstance(node.left,ast.Subscript):
            if isinstance(node.left.value, ast.Name) and (node.left.value.id in self.CONV_VARS or node.left.value.id in self.adjusted_vars):
                for i in range(len(node.comparators)):
                    if isinstance(node.comparators[i], ast.Subscript) and node.comparators[i].value not in self.adjusted_vars:
                        continue
                    else:
                        node.comparators[i] = self._adjust_index(node.comparators[i])
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
                    
                    dim_info = self.array_info[arr_name.casefold()]
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
        if isinstance(node, ast.Name) and (node.id in self.CONV_VARS or node.id in self.adjusted_vars):
            node = ast.BinOp(left = node, op = ast.Add(), right = ast.Constant(value=offset))
            
        elif isinstance(node, ast.BinOp): # THis is to handle cases when the 
            left = node.left
            right = node.right
            op = node.op

            is_valid_left = False

            # Case 1: left is a Name
            if isinstance(left, ast.Name):
                if left.id in self.CONV_VARS or left.id in self.adjusted_vars:
                    is_valid_left = True

            # Case 2: left is a Subscript of a Name (A[i]) and A is in adjusted_vars
            elif isinstance(left, ast.Subscript):
                if isinstance(left.value, ast.Name) and left.value.id in self.adjusted_vars:
                    is_valid_left = True

            if is_valid_left and isinstance(right, ast.Constant):
                original_value = right.value

                if isinstance(op, ast.Sub):
                    new_value = original_value - offset
                    if new_value == 0:
                        node = left
                    else:
                        node = ast.BinOp(
                            left=left,
                            op=ast.Sub(),
                            right=ast.Constant(value=new_value)
                        )

                elif isinstance(op, ast.Add):
                    new_value = original_value + offset
                    if new_value == 0:
                        node = left
                    else:
                        node = ast.BinOp(
                            left=left,
                            op=ast.Add(),
                            right=ast.Constant(value=new_value)
                        )

        elif isinstance(node, ast.Call):
            if not self._check_internal_call_element(node):
                node = ast.BinOp(left = node, op = ast.Add(), right = ast.Constant(value=offset))

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
                if index_node.value == 0:
                    return index_node
                else:
                    return ast.Constant(value=index_node.value - 1)
    
            elif isinstance(index_node,ast.Attribute):
                if index_node.attr not in self.CONV_VARS and index_node.attr not in self.adjusted_vars:
                    if (not self.exclude_index) or (index_node.attr not in self.exclude_index):
                        return self._subtract_one(index_node)
                    # return self._subtract_one(index_node)
                    
            elif isinstance(index_node, ast.Call):
                # Need to check if the usually int() internal elemnt is not that of the adjusted vars or that of the excluded_index 
                if self._check_internal_call_element(index_node):
                    return ast.BinOp(
                        left = index_node,
                        op = ast.Sub(),
                        right = ast.Constant(value=1)
                    )
                else:
                    return index_node

            elif isinstance(index_node,ast.Slice):
                if index_node.lower is None and index_node.upper is None:
                    return index_node

                # Recursively adjust lower and upper if they exist
                new_lower = self._adjust_index(index_node.lower) if index_node.lower else None
                new_upper = index_node.upper  # self._adjust_index(index_node.upper) if index_node.upper else None
                new_step = self._adjust_index(index_node.step) if index_node.step else None

                return ast.Slice(lower=new_lower, upper=new_upper, step=new_step)

            return index_node
        except Exception as e:
            raise RuntimeError(f"_adjust_index failed for node={ast.dump(index_node,indent=4)}") from e

    def _check_internal_call_element(self, node):
        """
        Checks whether the node is a call to int(some_var), and that
        some_var is NOT in adjusted_vars or excluded_index.

        Returns True if it's safe to apply transformation.
        """
        # Check if the node is a call, and function called is `int`
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'int':
            # Ensure there is exactly one argument
            if len(node.args) == 1:
                arg = node.args[0]

                if isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name):
                    var_name = arg.value.id
                    if var_name not in self.adjusted_vars and ((not self.exclude_index) or (var_name not in self.exclude_index)):
                        return True
                else:
                    raise ValueError(f'arg is not that of Subscript but of : {type(node)}')
            else:
                raise ValueError(f'Number of args inside the int() > 1')
        return False

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
            # print(ast.dump(node,indent=4))
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
    """
    Search for a target folder within a directory tree.

    This function recursively walks through a directory starting at `root_dir`
    and returns the full path to the first occurrence of `target_folder`.

    Parameters
    ----------
    root_dir : str
        The root directory to start the search from.
    target_folder : str
        The name of the folder to find.

    Returns
    -------
    str or None
        The absolute path to the target folder if found; otherwise, ``None``.
    
    """
    for dirpath, dirnames, _ in os.walk(root_dir):
        if target_folder in dirnames:
            return os.path.join(dirpath, target_folder)
    return None

def find_used_globals(node, common_attributes):
    """
    Find all global variables used within an AST node.

    This function recursively traverses an AST node and collects the names
    of all variables that match entries in `common_attributes`, indicating
    their usage as global variables or shared attributes.

    Parameters
    ----------
    node : ast.AST
        The root AST node to analyze.
    common_attributes : iterable of str
        A collection of variable names considered as global or shared attributes.

    Returns
    -------
    set of str
        A set of global variable names found within the given AST node.
    """
    used_globals = set()

    def visit(n):
        if isinstance(n, ast.Name):
            if n.id in common_attributes:
                used_globals.add(n.id)
        # Recursively visit all child nodes
        for child in ast.iter_child_nodes(n):
            visit(child)

    visit(node)
    return used_globals

def attach_instance(node, instance_name='self'):
    """
    Recursively attach an instance reference to all variable names in an AST node.

    This function traverses an abstract syntax tree (AST) and converts any 
    variable reference (``ast.Name``) into an attribute of a given instance name 
    (e.g., converting ``x`` into ``self.x``). It handles nested AST nodes and 
    lists of nodes recursively.

    Parameters
    ----------
    node : ast.AST
        The root AST node to process.
    instance_name : str, optional
        The name of the instance to attach (default is 'self').

    Returns
    -------
    ast.AST
        A modified AST node where all variable names are converted into 
        instance attributes (e.g., ``x`` → ``self.x``).
    """

    if isinstance(node, ast.Name):
        return ast.Attribute(
            value=ast.Name(id=instance_name, ctx=ast.Load()),
            attr=node.id,
            ctx=ast.Load()
        )
    
    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            new_list = []
            for item in value:
                if isinstance(item, ast.AST):
                    new_list.append(attach_instance(item, instance_name))
                else:
                    new_list.append(item)
            setattr(node, field, new_list)
        elif isinstance(value, ast.AST):
            setattr(node, field, attach_instance(value, instance_name))
    
    return node

def safe_eval_expr(node, attributes=None):
    """
    Safely evaluate a restricted AST expression with given variable bindings.

    This function evaluates a subset of Python expressions represented as 
    AST nodes. It supports constant values, binary arithmetic operations 
    (addition, subtraction, multiplication, division, etc.), and variable 
    lookups from a provided attribute dictionary. It is designed to avoid 
    executing arbitrary code, unlike Python's built-in `eval`.

    Parameters
    ----------
    node : ast.AST
        The AST node representing the expression to evaluate.
    attributes : dict of {str: tuple}, optional
        A mapping of variable names to tuples containing their values.
        For example, ``{'x': (5,), 'y': (10,)}``. Defaults to an empty dictionary.

    Returns
    -------
    any
        The evaluated result of the expression.

    Raises
    ------
    NameError
        If a variable is referenced that does not exist in `attributes`.
    NotImplementedError
        If the expression contains unsupported AST node types or operators.
    
    """
    if attributes is None:
        attributes = {}

    if isinstance(node, ast.Constant):
        return node.value

    elif isinstance(node, ast.BinOp):
        left = safe_eval_expr(node.left, attributes)
        right = safe_eval_expr(node.right, attributes)

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

    elif isinstance(node, ast.Name):
        var_name = node.id
        if var_name in attributes:
            return attributes[var_name][0]  # Get the actual value
        else:
            raise NameError(f"Variable '{var_name}' not found in attributes.")
    elif isinstance(node,ast.Attribute):
        if isinstance(node.attr, str):
            return attributes[node.attr][0]
    else:
        raise NotImplementedError(f"Unsupported AST node type: {type(node)}")


def update_methods(module_dict:Dict, function_defs:List):
    """
    Update the 'methods' entry of a module dictionary with new function definitions.

    This function adds or updates function definitions in the `'methods'` key of a given `module_dict`. Each entry in `function_defs` is expected to be
    an `ast.FunctionDef` node representing a function to be included in the module's method dictionary.

    Parameters
    ----------
    module_dict : dict
        A dictionary representing a module structure. It should contain a
        `'methods'` key mapping to a dictionary of existing function definitions.
    function_defs : list of ast.FunctionDef
        A list of function definition nodes to add or update in the module's
        `'methods'` dictionary.

    Returns
    -------
    dict
        The updated module dictionary with new or modified function definitions
        under the `'methods'` key.
    """

    # Loop over the top-level module(s)
    for _, module_content in module_dict.items():
        # Search for the inner dict that contains 'methods'
        for _, instance_val in module_content.items():
            if isinstance(instance_val, dict) and 'methods' in instance_val:
                methods_dict = instance_val['methods']
                for func_def in function_defs:
                    if isinstance(func_def, ast.FunctionDef):
                        methods_dict[func_def.name] = func_def
                # Ensure the dict is updated
                instance_val['methods'] = methods_dict
                break 

def collect_dependencies(node):
    """
    Collect all variable names that an AST node depends on.

    This function traverses an abstract syntax tree (AST) node and extracts all variable names referenced within it. Both plain variable names
    (`ast.Name`) and object attributes (`ast.Attribute`) are included in the resulting dependency set.

    Parameters
    ----------
    node : ast.AST
        The AST node to analyze for variable dependencies.

    Returns
    -------
    set of str
        A set containing the names of all variables and attributes used within
        the given AST node.
    """
    deps = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            deps.add(child.id)
        elif isinstance(child, ast.Attribute):
            deps.add(child.attr)
    return deps

def order_assignments(assign_nodes:List, diff:List) -> List:
    """
    Order assignment nodes based on variable dependencies using topological sorting.

    This function analyzes a list of assignment nodes to determine their dependency relationships, similar to the `search_dependent_variables` method in the
    `Transformer` class. It then orders the assignments using Kahn's algorithm for topological sorting to ensure that each variable is assigned only after its dependencies have been resolved.

    Parameters
    ----------
    assign_nodes : list
        A list of AST assignment nodes or equivalent representations of variable assignments.
    diff : list
        A list of dependency relationships or variable differences used to
        establish ordering constraints among assignments.

    Returns
    -------
    list
        A list of assignment nodes sorted in a dependency-respecting order
        according to topological sorting.

    Notes
    -----
    - Kahn's algorithm is used to guarantee a valid ordering where dependencies precede dependents.
    - This function is useful when generating or transforming code that relies on the correct order of variable initialization.

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

def search_convar_dependencies(conv_vars: List[str], node: ast.AST) -> Set[str]:
    """
    Identify variables that depend on given conventional variables within an AST node.

    This function traverses an abstract syntax tree (AST) and finds assignments where the right-hand side (RHS) expressions depend on any of the specified conventional variables (`conv_vars`). It returns the names of variables that are affected 
    or derived from those conventional variables.

    Parameters
    ----------
    conv_vars : list of str
        A list of conventional variable names to search for in expressions.
    node : ast.AST
        The root AST node to analyze for variable dependencies.

    Returns
    -------
    set of str
        A set of variable names that are directly affected by or derived from the given conventional variables.

    """
    adjusted_vars = set()

    for child in ast.walk(node):
        if isinstance(child, ast.Assign):
            value = child.value
            if _find_conv_vars_in_expr(value, conv_vars):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        adjusted_vars.add(target.id)
                    elif isinstance(target, ast.Attribute):
                        adjusted_vars.add(target.attr)
                    elif isinstance(target, ast.Subscript):
                        # Handles cases like x[i] = ...
                        if isinstance(target.value, ast.Name):
                            adjusted_vars.add(target.value.id)

    return adjusted_vars

def _find_conv_vars_in_expr(child: ast.AST, conv_vars: List[str]) -> bool:
    """
    Recursively check whether any of the given conventional variables appear in an expression node.

    This function traverses an AST expression node to determine if it contains any variables listed in `conv_vars`. It is typically used to detect whether 
    an assignment or operation depends on specific conventional variables.

    Parameters
    ----------
    child : ast.AST
        The AST node (usually an expression) to inspect.
    conv_vars : list of str
        A list of conventional variable names to search for within the expression.

    Returns
    -------
    bool
        True if any variable from `conv_vars` is found within the expression, 
        otherwise False.
    """
    if isinstance(child, ast.Name) and child.id in conv_vars:
        return True
    elif isinstance(child, ast.Attribute):
        if isinstance(child.value, ast.Name) and child.value.id in conv_vars:
            return True
        if child.attr in conv_vars:
            return True
    elif isinstance(child, ast.BinOp):
        name = None
        if isinstance(child.left, (ast.Name, ast.Attribute)):
            name = child.left.id if isinstance(child.left, ast.Name) else child.left.attr
        elif isinstance(child.right, (ast.Name, ast.Attribute)):
            name = child.right.id if isinstance(child.right, ast.Name) else child.right.attr

        if name in conv_vars:
            return True
    elif isinstance(child, ast.Compare):
        # Check all parts of the comparison
        if isinstance(child.left, ast.Name) and child.left.id in conv_vars:
            return True
        for comparator in child.comparators:
            if isinstance(comparator, ast.Name) and comparator.id in conv_vars:
                return True

    return False