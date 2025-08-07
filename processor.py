import logging
import re, os
from fparser.two.utils import walk
from fparser.two import Fortran2003 as F23
from fparser.two import Fortran2008 as F28
from fparser.common.readfortran import FortranFileReader, FortranStringReader
from fparser.two.parser import ParserFactory
from collections import deque
from line_length import FortLineLength 

class Processor:
    """
    A class for parsing Fortran files, strings, and statements using fparser, 
    creating Abstract Syntax Trees (ASTs), generating Fortran code, and performing 
    various mappings.

    Attributes:
    -----------
    parser : object
        An instance of the Fortran 2008 parser created by fparser.
    """
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.parser = ParserFactory().create(std="f2008")
        self.line_length = FortLineLength()
        current_dir = os.getcwd()
        self.benchmark_dir = os.path.join(current_dir, 'benchmark')
        os.makedirs(self.benchmark_dir, exist_ok=True)

    def parse_fortran_file(self, file_path):
        try:
            reader = FortranFileReader(file_path, ignore_comments=False)
            parse_tree = self.parser(reader)
            logging.info(f"Successfully parsed file: {file_path}")
            return parse_tree
        except Exception as e:
            logging.error(f"Failed to parse file: {file_path}, Error: {e}")
            raise

    def parse_fortran_string(self, string):
        try:
            reader = FortranStringReader(string, ignore_comments=False)
            parse_tree = self.parser(reader)
            logging.info(f"Successfully parsed string!")
            return parse_tree#.children[0]
        except Exception as e:
            logging.error(f"Failed to parse string, Error: {e}")
            raise

    def parse_fortran_statement(self, stmt_str):
        code = f"""
          program foo
          {stmt_str}
          end
        """
        try:
            reader = FortranStringReader(code, ignore_comments=True)
            parse_tree = self.parser(reader)
            for node in parse_tree.children:
                for child in node.content:
                    if not isinstance(child, F23.Program_Stmt) and not isinstance(child, F23.End_Program_Stmt):
                        logging.info(f"Successfully parsed statement: {stmt_str}")
                        return child#.children[0]
            logging.warning(f"No valid statements found in: {stmt_str}")
            return None
        except Exception as e:
            logging.error(f"Failed to parse statement: {stmt_str}, Error: {e}")
            raise

    def parse_fortran_comment(self, stmt_str):
        code = f"""
    program foo
    {stmt_str}
    end
    """
        try:
            reader = FortranStringReader(code, ignore_comments=False)
            parse_tree = self.parser(reader)
            for node in parse_tree.children:
                if not hasattr(node, 'content'):
                    continue
                for child in node.content:
                    if not isinstance(child, F23.Program_Stmt) and not isinstance(child, F23.End_Program_Stmt):
                        return child.children[0].children[0]
        except Exception as e:
            raise RuntimeError(f"An error occurred while parsing Fortran code: {e}")

    def initiate_empty_routine(self, subroutine_name):
        try:
            code_template = f"""
            subroutine read_dummy()
                open(unit=1363, file='{self.benchmark_dir}/{subroutine_name}/dummy.bin', form='unformatted', status='old')
                write(*,*) '--- inside the read dummy routine for {subroutine_name} ---'
            end subroutine read_dummy
            """
            reader = FortranStringReader(code_template, ignore_comments=True)
            parse_tree = self.parser(reader)
            return parse_tree.children[0]
        except Exception as e:
            logging.error(f"Failed to generate Fortran code for routine {subroutine_name}, Error: {e}")
            raise

    def separate_entity_declarations(self, declaration_stmt):
        try:
            left_part = []
            right_part = []
            for child in declaration_stmt.children:
                if child==None:
                    continue
                if not isinstance(child, F23.Entity_Decl_List):
                    left_part.append(child.tostr())
                if isinstance(child, F23.Entity_Decl_List):
                    for child_child in walk(child, F23.Entity_Decl):
                        right_part.append(child_child.string)

            left_part_merged = ', '.join([name for name in left_part])
            new_decl = []
            for variable in right_part:
                decl = f"{left_part_merged} :: {variable}"
                new_decl.append(F23.Type_Declaration_Stmt(decl))
            return new_decl
        except Exception as e:
            print(f"An error occurred during separation of variables: {e}")
            return None

    def add_entity_to_declaration(self, declaration_stmt, var_modif):
        try:
            left_part = []
            right_part = []
            for child in declaration_stmt.children:
                if child==None:
                    continue
                if not isinstance(child, F23.Entity_Decl_List):
                    left_part.append(child.tostr())
                if isinstance(child, F23.Entity_Decl_List):
                    for child_child in walk(child, F23.Entity_Decl):
                        right_part.append(child_child.string)
                        if child_child.string in var_modif:
                            right_part.append(child_child.string+'_cpu')

            left_part_merged = ', '.join([name for name in left_part])
            right_part_merged = ', '.join([name for name in right_part])
            decl = f"{left_part_merged} :: {right_part_merged}"
            new_decl = F23.Type_Declaration_Stmt(decl)
            return new_decl
        except Exception as e:
            print(f"An error occurred during adding an entity to a declaration: {e}")
            return None

    def separate_entity_allocation(self, allocate_stmt):
        try:
            lst = []
            opt = None

            for child in allocate_stmt.children:
                if child is None:
                    continue
                if isinstance(child, F23.Allocation_List) or isinstance(child, F28.Allocation_List):
                    for grandchild in child.children:
                        lst.append(grandchild.tostr())
                elif isinstance(child, F23.Alloc_Opt_List) or isinstance(child, F28.Alloc_Opt_List):
                    for grandchild in child.children:
                        opt = grandchild.tostr()

            new_allocation = []
            for variable in lst:
                if opt is not None:                                                                                                              
                    allocation = f"allocate({variable}, {opt})"                                      
                else:                                                                                  
                    allocation = f"allocate({variable})"   

                new_allocation.append(F23.Allocate_Stmt(allocation))

            logging.info("Successfully generated allocation statements")
            return new_allocation

        except Exception as e:
            logging.error(f"Failed to generate allocation statements, Error: {e}")
            raise

    def add_entity_to_allocation(self, allocate_stmt, var_modif):
        try:
            new_allocation = []
            for child in allocate_stmt.children:
                if child is None:
                    continue
                if isinstance(child, F23.Allocation_List) or isinstance(child, F28.Allocation_List):
                    for grandchild in child.children:
                        allocate_name = grandchild.children[0].tostr()
                        code = f"if(.not. allocated({allocate_name}))then\n{allocate_stmt.tostr()}\nend if"
                        new_allocation.append(self.parse_fortran_statement(code))
                        if allocate_name in var_modif:
                            allocate_name_add = allocate_name + '_cpu'
                            allocation = re.sub(rf'\b{allocate_name}\b', allocate_name_add, allocate_stmt.tostr())
                            code = f"if(.not. allocated({allocate_name_add}))then\n{allocation}\nend if"
                            new_allocation.append(self.parse_fortran_statement(code))
            logging.info("Successfully generated allocation statements")
            return new_allocation
        except Exception as e:
            logging.error(f"Failed to generate allocation statements, Error: {e}")
            raise

    def map_declaration(self, implicit_dec, explicit_dec=None, dimensions=None):
        try:
            if walk(implicit_dec, F23.Intrinsic_Type_Spec):
                type_and_attributes = walk(implicit_dec, F23.Intrinsic_Type_Spec)[0].tostr()
            else:
                raise ValueError("variable type is not present!")
            
            if walk(implicit_dec, F23.Entity_Decl):
                array_name = walk(implicit_dec, F23.Entity_Decl)[0].tostr()
            else:
                raise ValueError("variable name is not present!")
            
            if dimensions is None:
                assert explicit_dec is not None, "explicit_dec must be provided when assumed_shape is not None."
                if walk(explicit_dec, F23.Explicit_Shape_Spec):
                    shape = []
                    for dim in walk(explicit_dec, F23.Explicit_Shape_Spec):
                        shape.append(dim.tostr())
                    dimensions = ', '.join([name for name in shape])
                else:
                    raise ValueError("array shape is not present!")
            
            if walk(implicit_dec, F23.Intent_Attr_Spec):
                intent_attr = walk(implicit_dec, F23.Intent_Attr_Spec)[0].tostr()
                new_decl = f"{type_and_attributes}, DIMENSION({dimensions}), {intent_attr} :: {array_name}"
            else:
                new_decl = f"{type_and_attributes}, DIMENSION({dimensions}) :: {array_name}"
            
            logging.info(f"Mapped declaration: {new_decl}")
            parsed_decl = F23.Type_Declaration_Stmt(new_decl)
            return parsed_decl
        except Exception as e:
            logging.error(f"Failed to map declaration, Error: {e}")
            raise

    def combine_allocate_declaration(self, variable_declarations):
        try:
            for item in variable_declarations:
                if isinstance(item, F23.Allocate_Stmt):
                    if walk(item, F23.Allocate_Shape_Spec):
                        shape = []
                        for dim in walk(item, F23.Allocate_Shape_Spec):
                            shape.append(dim.tostr())
                        dimensions = ', '.join([name for name in shape])
                    else:
                        raise ValueError("allocarion shape is not present!")
                elif isinstance(item, F23.Type_Declaration_Stmt):
                    if walk(item, F23.Intrinsic_Type_Spec):
                        type_and_attributes = walk(item, F23.Intrinsic_Type_Spec)[0].tostr()
                    else:
                        raise ValueError("variable type is not present!")
                    if walk(item, F23.Entity_Decl):
                        array_name = walk(item, F23.Entity_Decl)[0].tostr()
                    else:
                        raise ValueError("variable name is not present!")
                else:
                    raise ValueError('Unrecognized statement')
            
            combined_statement = f"{type_and_attributes}, DIMENSION({dimensions}) :: {array_name}"
            logging.info(f"Combined statement: {combined_statement}")
            parsed_decl = F23.Type_Declaration_Stmt(combined_statement)
            return parsed_decl
        except Exception as e:
            logging.error(f"Failed to combine allocate and declaration, Error: {e}")
            raise

    def remove_intent_and_save(self, type_declaration_stmts):
        exclude = ['SAVE', 'INTENT', 'PUBLIC']
        cleaned_statements = []
        try:
            for stmt in type_declaration_stmts:
                if not isinstance(stmt, F23.Type_Declaration_Stmt):
                    cleaned_statements.append(stmt)
                    continue
                left_part = []
                right_part = ""
                for child in stmt.children:
                    if isinstance(child, F23.Intrinsic_Type_Spec):
                        left_part.append(child.tostr())
                    elif isinstance(child, F28.Attr_Spec_List) or isinstance(child, F23.Attr_Spec_List):
                        for grandchild in child.children:
                            if not any(excl in grandchild.string for excl in exclude):
                                left_part.append(grandchild.tostr())
                    elif isinstance(child, F23.Entity_Decl_List):
                        right_part = child.tostr()

                left_part_merged = ', '.join(left_part)
                cleaned_stmt_str = f"{left_part_merged} :: {right_part}"
                cleaned_stmt = F23.Type_Declaration_Stmt(cleaned_stmt_str)
                cleaned_statements.append(cleaned_stmt)
            logging.info("Successfully removed INTENT and SAVE attributes from statements")
            return cleaned_statements
        except Exception as e:
            logging.error(f"Failed to remove INTENT and SAVE attributes, Error: {e}")
            raise

    def out_module_fortran(self, subroutine_name):
        code = f"""
        module module_global
        implicit none
        integer, parameter :: i_std = 4
        integer, parameter :: r_std = 8
        integer(kind = i_std), parameter :: nsnow=3
        integer(kind = i_std), parameter :: nslm=11
        integer(kind = i_std), parameter :: nvm = 15
        integer(kind = i_std), parameter :: nstm = 3
        integer(kind = i_std), parameter :: kjpindex = 4717
        integer                          :: ier
        integer(kind = i_std)            :: ic0, ic
        real(kind = r_std)               :: icr, start_time, stop_time
        contains
        subroutine declaration_initialization()
        open(unit=1363, file='{self.benchmark_dir}/{subroutine_name}/global.bin', form='unformatted', status='old')
        write(*,*)'--- add the declaration and initialization in module global ---'
        end subroutine declaration_initialization
        end module module_global
        """
        try:
            reader = FortranStringReader(code, ignore_comments=False)
            parse_tree = self.parser(reader)
            logging.info("Successfully parsed module code")
            return parse_tree
        except Exception as e:
            logging.error(f"Failed to parse module code, Error: {e}")
            raise

    def check_point(self, var, var_cpu, info):
        code = ""

        if 'DIMENSION' in info:
            if 'LOGICAL' not in info:
                code_template = f"""
                IF (ALL({var} .EQ. {var_cpu})) THEN
                    write(*,*) 'Test passed: All elements in {var}_gpu are equal to {var_cpu}.'
                ELSE
                    write(*,*) ''
                    write(*,*) 'Test failed: All elements in {var}_gpu do not match {var_cpu}.'
                    write(*,'(A, E25.16)') 'Maximum absolute error:',  maxval(abs({var} - {var_cpu}))
                    write(*,'(A, 2E25.16)') 'Min and Max of {var}_gpu:', minval({var}), maxval({var})
                    write(*,'(A, 2E25.16)') 'Min and Max of {var_cpu}:', minval({var_cpu}), maxval({var_cpu})
                    write(*,*) ''
                ENDIF
                """
            else:
                code_template = f"""
                IF (ALL({var} .EQV. {var_cpu})) THEN
                    write(*,*) 'LOGICAL EQV test passed: All elements in {var}_gpu are equal to {var_cpu}.'
                ELSE
                    write(*,*) ''
                    write(*,*) 'LOGICAL EQV test failed: Not all elements in {var}_gpu match {var_cpu}.'
                    write(*,*) ''
                ENDIF
                """
        else:
            if 'LOGICAL' not in info:
                code_template = f"""
                IF ({var} .EQ. {var_cpu}) THEN
                    write(*,*) 'Test passed: {var}_gpu is equal to {var_cpu}.'
                ELSE
                    write(*,*) ''
                    write(*,*) 'Test failed: {var}_gpu does not match {var_cpu}.'
                    write(*,'(A, E25.16)') 'Absolute error:', abs({var} - {var_cpu})
                    write(*,'(A, 2E25.16)') '{var}_gpu:', {var}
                    write(*,'(A, 2E25.16)') '{var_cpu}:', {var_cpu}
                    write(*,*) ''
                ENDIF
                """
            else:
                code_template = f"""
                IF ({var} .EQV. {var_cpu}) THEN
                    write(*,*) 'LOGICAL EQV test passed: {var}_gpu is equal to {var_cpu}.'
                ELSE
                    write(*,*) ''
                    write(*,*) 'LOGICAL EQV test failed: {var}_gpu does not match {var_cpu}.'
                    write(*,'(A, L1)') '{var}_gpu:', {var}
                    write(*,'(A, L1)') '{var_cpu}:', {var_cpu}
                    write(*,*) ''
                ENDIF
                """
        code = self.parse_fortran_statement(code_template)

        return code

    def out_main_fortran(self):
        code = """
        program main
        implicit none
        write(*,*)'--- inside the main program ---'
        contains
        end program main
        """
        try:
            reader = FortranStringReader(code, ignore_comments=False)
            parse_tree = self.parser(reader)
            logging.info("Successfully parsed main program code")
            return parse_tree
        except Exception as e:
            logging.error(f"Failed to parse main program code, Error: {e}")
            raise

    def write_fortran_code_to_file(self, code, file_path):
        try:
            code = self.line_length.process(code.tostr())
            with open(file_path, 'w') as f:
                f.write(str(code))
            logging.info(f"Successfully wrote code to file: {file_path}")
        except Exception as e:
            logging.error(f"Failed to write code to file: {file_path}, Error: {e}")
            raise

    def update_global_module(self, input_dict, file_path, subroutine_name, module_tree):
        try:
            self.out_module = self.out_module_fortran(subroutine_name)
            write_stmt_code = "\n".join(self.write_stmt)
            for call in walk(module_tree, F23.Call_Stmt):
                assert isinstance(call.children[0], F23.Name), f"Expected F23.Name, but got {type(call.children[0])}"
                assert isinstance(call.children[1], F23.Actual_Arg_Spec_List), \
                        f"Expected F23.Actual_Arg_Spec_List, but got {type(call.children[1])}"
                if call.children[0].tostr() == subroutine_name:
                    code_template = (
                            f"{call.tostr()}\n"
                            f"open(unit=1363, file='{self.benchmark_dir}/{subroutine_name}/global.bin', form='unformatted', status='replace')\n"
                            f"{write_stmt_code}\n"
                            "close(1363)"
                            )
                    call.parent.children[call.parent.children.index(call)] = self.parse_fortran_statement(code_template)
                    break

            for node in self.out_module.content:
                if isinstance(node, F23.Module):
                    for idx, subnode in enumerate(node.content):
                        if isinstance(subnode, F23.Specification_Part):
                            if self.add_to_usestm:
                                for stmt in self.add_to_usestm:
                                    subnode.content.insert(0, stmt)
                            ldx = len(subnode.content) - 1
                            for stmt in self.add_to_module:
                                subnode.content.insert(ldx + 1, stmt)
                                ldx += 1
                            if self.acc_declare_copyin:
                                subnode.content.insert(ldx + 1, self.acc_declare_copyin_cmd)
                                ldx += 1
                            if self.acc_declare_create:
                                subnode.content.insert(ldx + 1, self.acc_declare_create_cmd)
                                ldx += 1
                        elif isinstance(subnode, F23.Module_Subprogram_Part):
                            for jdx, subsubnode in enumerate(subnode.content):
                                if isinstance(subsubnode, F23.Subroutine_Subprogram):
                                    subroutine = walk(subsubnode, F23.Name)[0].string
                                    for kdx, subsubsubnode in enumerate(subsubnode.content):
                                        if isinstance(subsubsubnode, F23.Execution_Part):
                                            ldx = len(subsubsubnode.content) - 1
                                            if subroutine == 'declaration_initialization':
                                                for stmt in self.reads_in_decleration_routine:
                                                    subsubsubnode.content.insert(ldx + 1, stmt)
                                                    ldx += 1
                                                for stmt in self.add_to_routin:
                                                    subsubsubnode.content.insert(ldx + 1, stmt)
                                                    ldx += 1
                                            #if subroutine == 'initialization':
                                                for stmt in self.reads_in_read_routine:
                                                    subsubsubnode.content.insert(ldx + 1, stmt)
                                                    ldx += 1
                                                subsubsubnode.content.insert(ldx + 1,F23.Close_Stmt(f"close(1363)"))
                                                ldx += 1
                                                if self.acc_declare_create:
                                                    subsubsubnode.content.insert(ldx + 1, self.acc_update_device_cmd)
                                                    ldx += 1
            logging.info("Successfully updated the global module")
            self.write_fortran_code_to_file(self.out_module, file_path)
        except Exception as e:
            logging.error(f"Failed to update global module, Error: {e}")
            raise

    def update_main_program(self, custom_dec_inout, custom_subroutine_trees, call_stmts, \
            dummy_add_decl, \
            error_flag, \
            acc_data_copyin,\
            var_modif, file_path, \
            subroutine_name, dummy_args, module_tree, \
            childs_subroutine_tree=None):
        try:
            self.out_main = self.out_main_fortran()
            custom_module_name = walk(walk(self.out_module,F23.Module_Stmt), F23.Name)[0].string
            custom_subroutines_names = [name.string for name in walk(walk(self.out_module,F23.Subroutine_Stmt),F23.Name)]
            #specification_part = self.remove_intent_and_save(custom_dec_inout)
            initialization_part = self.initialization_statement(custom_dec_inout)
            if self.dummy_list:
                print('need to build an initialization for in/inout dummy args: ')
                specification_part_dummy = self.remove_intent_and_save(self.dummy_list)
                block_tree = self.generate_read_routine(specification_part_dummy, initialization_part, subroutine_name)

            custom_dec_inout.append(dummy_add_decl)
            if error_flag.keys():
                for key in error_flag.keys():
                    custom_dec_inout.append(error_flag[key]['error_flag_decl'])
            specification_part = self.remove_intent_and_save(custom_dec_inout)

            if initialization_part:
                self.write_stmt = []
                for call in walk(module_tree, F23.Call_Stmt):
                    assert isinstance(call.children[0], F23.Name), f"Expected F23.Name, but got {type(call.children[0])}"
                    assert isinstance(call.children[1], F23.Actual_Arg_Spec_List), \
                            f"Expected F23.Actual_Arg_Spec_List, but got {type(call.children[1])}"
                    if call.children[0].tostr() == subroutine_name:
                        arg_string = [string.strip() for string in call.children[1].tostr().split(',')]
                        for rstmt in initialization_part:
                            assert isinstance(rstmt.children[0].children[2], F23.Input_Item_List)
                            assert len(rstmt.children[0].children[2].children) == 1
                            arg = rstmt.children[0].children[2].tostr()
                            corresponding_element = arg_string[dummy_args.index(arg)]
                            self.write_stmt.append(F23.Write_Stmt(f"write(1363){corresponding_element}").tostr())
                        write_dummy_code = "\n".join(self.write_stmt)
                        code_template = (
                                f"{call.tostr()}\n"
                                f"open(unit=1363, file='{self.benchmark_dir}/{subroutine_name}/dummy.bin', form='unformatted', status='replace')\n"
                                f"{write_dummy_code}\n"
                                "close(1363)"
                                )
                        call.parent.children[call.parent.children.index(call)] = self.parse_fortran_statement(code_template)
                        break

            for node in self.out_main.content:
                if isinstance(node, F23.Main_Program):
                    for idx, subnode in enumerate(node.content):
                        if isinstance(subnode, F23.Specification_Part):
                            use = walk(subnode, F23.Use_Stmt)
                            kdx = len(use) - 1
                            use_stmt = 'use ' + custom_module_name
                            subnode.content.insert(kdx + 1, F23.Use_Stmt(use_stmt))
                            kdx = len(subnode.content) - 1
                            for stmt in specification_part:
                                is_modified = any(name.string in var_modif for name in walk(stmt, F23.Entity_Decl))
                                if is_modified:
                                    subnode.content.insert(kdx + 1, self.add_entity_to_declaration(stmt, var_modif))
                                else:
                                    subnode.content.insert(kdx + 1, stmt)
                                kdx += 1
                        elif isinstance(subnode, F23.Execution_Part):
                            kdx = len(subnode.content) - 1
                            #for name in custom_subroutines_names:
                            subroutine_call = "Call declaration_initialization"
                            subnode.content.insert(kdx + 1, F23.Call_Stmt(subroutine_call))
                            kdx += 1

                            subnode.content.insert(kdx + 1, self.create_call_stmt(block_tree))
                            kdx += 1

                            code_start = """
                            call SYSTEM_CLOCK(ic0, icr, ic)
                            start_time = ic0*1.0/icr
                            """

                            subnode.content.insert(kdx + 1, self.parse_fortran_statement(code_start))
                            kdx += 1

                            subnode.content.insert(kdx + 1, call_stmts[0])
                            kdx += 1

                            code_end = f"""
                            call SYSTEM_CLOCK(ic0, icr, ic)
                            stop_time = ic0*1.0/icr
                            WRITE(*,*) "Execution time : ",stop_time - start_time
                            open(unit=1363, file='{self.benchmark_dir}/{subroutine_name}/time.txt', status='unknown', position='append')
                            WRITE(1363,*) stop_time - start_time
                            close(1363)
                            """

                            subnode.content.insert(kdx + 1, self.parse_fortran_statement(code_end))
                            kdx += 1

                            write_statements = []
                            for modified_var in var_modif:
                                stmt = F23.Write_Stmt(f"write(1363) {modified_var}") 
                                write_statements.append(stmt.tostr()) 
                            write_stmt_code = "\n".join(write_statements)
                            code_template = (
                                f"open(unit=1363, file='{self.benchmark_dir}/{subroutine_name}/output.bin', form='unformatted', status='replace')\n"
                                f"{write_stmt_code}\n"
                                "close(1363)"
                            )
                            
                            subnode.content.insert(kdx + 1, self.parse_fortran_statement(code_template))
                            kdx += 1

                            for modified_var in var_modif:
                                stmt_str = modified_var + '_cpu' + '=' + modified_var
                                subnode.content.insert(kdx + 1, F23.Assignment_Stmt(stmt_str))
                                kdx += 1

                            #for name in custom_subroutines_names:
                            #if name == 'initialization':
                            subroutine_call = "Call declaration_initialization"
                            subnode.content.insert(kdx + 1, F23.Call_Stmt(subroutine_call))
                            kdx += 1
                            subnode.content.insert(kdx + 1, self.create_call_stmt(block_tree))
                            kdx += 1
                            reductions = ''
                            if error_flag.keys():
                                all_arrors = ', '.join([err for err in list(error_flag.keys())])
                                reductions =f'REDUCTION(+:{all_arrors})'
                                for key in error_flag.keys():
                                    subnode.content.insert(kdx + 1, error_flag[key]['error_flag_init'])
                                    kdx += 1
                            
                            if acc_data_copyin:
                                copyinlist = ', '.join([item for item in acc_data_copyin])
                                acc_copyin_cmd = Processor().parse_fortran_comment(f"!$ACC ENTER DATA COPYIN({copyinlist})")
                                acc_delete_cmd = Processor().parse_fortran_comment(f"!$ACC EXIT DATA DELETE({copyinlist})")
                                subnode.content.insert(kdx + 1, acc_copyin_cmd)
                                kdx += 1
                            acc_kernels_cmd, acc_end_kernels_cmd = Processor().parse_fortran_comment(f"!$ACC PARALLEL LOOP INDEPENDENT {reductions}"),\
                                    Processor().parse_fortran_comment(f"!$ACC END PARALLEL")

                            subnode.content.insert(kdx + 1, self.parse_fortran_statement(code_start))
                            kdx += 1

                            subnode.content.insert(kdx + 1, acc_kernels_cmd)
                            kdx += 1
                            subnode.content.insert(kdx + 1, call_stmts[1])
                            kdx += 1
                            subnode.content.insert(kdx + 1, acc_end_kernels_cmd)
                            kdx += 1

                            subnode.content.insert(kdx + 1, self.parse_fortran_statement(code_end))
                            kdx += 1

                            acc_update_self_str = ', '.join([modified_var for modified_var in var_modif])# \
                                    #if modified_var in self.acc_declare_create])
                            acc_update_self_cmd = self.parse_fortran_comment(f"!$ACC UPDATE SELF({acc_update_self_str})")
                            subnode.content.insert(kdx + 1, acc_update_self_cmd)
                            kdx += 1
                            if acc_data_copyin:
                                subnode.content.insert(kdx + 1, acc_delete_cmd)
                                kdx += 1

                            for modified_var in var_modif:
                                var_info = var_modif[modified_var]
                                if_construct = self.check_point(modified_var, modified_var+'_cpu', var_info)
                                subnode.content.insert(kdx + 1, if_construct)
                                kdx += 1

                            if error_flag.keys():
                                for key in error_flag.keys():
                                    subnode.content.insert(kdx + 1, error_flag[key]['write_calls'])
                                    kdx += 1
                            
                        elif isinstance(subnode, F23.Internal_Subprogram_Part):
                            if self.dummy_list:
                                node.content.insert(idx + 1, block_tree)
                            for custom_subroutine_tree in custom_subroutine_trees:
                                node.content.insert(idx + 1, custom_subroutine_tree)
                            if childs_subroutine_tree is not None:
                                for child_subroutine in childs_subroutine_tree:
                                    node.content.insert(idx + 1, child_subroutine)
            logging.info("Successfully updated the main program")
            self.write_fortran_code_to_file(self.out_main, file_path)
        except Exception as e:
            logging.error(f"Failed to update main program, Error: {e}")
            raise

    def create_call_stmt(self, subroutine_tree):
        try:
            subroutine_stmt = walk(subroutine_tree, F23.Subroutine_Stmt)[0]
            subroutine_stmt_string = subroutine_stmt.tostr()
            subroutine_name = subroutine_stmt_string.replace('SUBROUTINE', '').strip()
            subroutine_call = f"CALL {subroutine_name}"
            subroutine_call_obj = F23.Call_Stmt(subroutine_call)
            return subroutine_call_obj
        except Exception as e:
            print(f"An error occurred in creating a Call_Stmt object: {e}")
            raise

    def generate_read_routine(self, specification_part, initialization_part, subroutine_name):
        try:
            block = self.initiate_empty_routine(subroutine_name)
            if hasattr(block, "content"):
                idc = 0
                while idc < len(block.content):
                    child = block.content[idc]
                    if isinstance(child, F23.Subroutine_Stmt):
                        subroutine_stmt = "subroutine "
                        for grandchild in child.children:
                            if isinstance(grandchild, F23.Name):
                                subroutine_stmt += f"{grandchild.tostr()}"
                        dummy_arg_list = []
                        for stmt in specification_part:
                            entity_dec = walk(stmt, F23.Entity_Decl)
                            for entity in entity_dec:
                                dummy_arg_list.append(entity.tostr())
                        dummy_arg = ', '.join([name for name in dummy_arg_list])
                        subroutine_stmt += f"({dummy_arg})"
                        block.content[idc] = F23.Subroutine_Stmt(subroutine_stmt)
                        for stmt in specification_part:
                            block.content.insert(idc + 1, stmt)
                    elif isinstance(child, F23.Execution_Part):
                        kdx = len(child.content) - 1
                        for stmt in initialization_part:
                            child.content.insert(kdx + 1, stmt)
                            kdx += 1
                        child.content.insert(kdx + 1,F23.Close_Stmt(f"close(1363)"))
                        kdx += 1
                    idc += 1
                return block
        except Exception as e:
            logging.error(f"Failed to generate read routine, Error: {e}")
            raise

    def initialization_statement(self, items):
        read_list = []
        items_sep = []
        self.dummy_list = []
        for item in items:
            if len(walk(item, F23.Entity_Decl)) > 1:
                node_list = self.separate_entity_declarations(item)
                for node in node_list:
                    items_sep.append(node)
            else:
                items_sep.append(item)
        try:
            for item in items_sep:
                init = True
                intent = walk(item, F23.Intent_Spec)
                if F23.Intent_Spec('OUT') in intent:
                    init = False
                for child in item.children:
                    if isinstance(child, F23.Intrinsic_Type_Spec):
                        var_type = child.tostr()
                    if isinstance(child, F23.Entity_Decl_List):
                        var_name = child.tostr()
                if init:
                    self.dummy_list.append(item)
                    '''if 'REAL' in var_type:
                        stmt_str = f"Call random_number({var_name})"
                    elif 'INTEGER' in var_type:
                        stmt_str = f"{var_name} = 2"
                    elif 'LOGICAL' in var_type:
                        stmt_str = f"{var_name} = .TRUE."
                    else:
                        raise NotImplementedError(f"Variable type not implemented: {var_type}")
                    read_list.append(self.parse_fortran_statement(stmt_str))
                    '''
                    code_template = f"""
                    read(1363, iostat = ier){var_name}
                    if (ier /= 0) then
                    write(*,*) 'Error reading from file for {var_name}. ',' IOSTAT : ', ier
                    endif
                    """
                    read_list.append(self.parse_fortran_statement(code_template))
                    #read_list.append(F23.Read_Stmt(f"read(1363){var_name}"))
                    self.write_stmt.append(F23.Write_Stmt(f"write(1363){var_name}").tostr())
            #read_list.append(F23.Close_Stmt(f"close(1363)"))
            logging.info(f"processing initialization completed!")
            return read_list
        except Exception as e:
            logging.error(f"Error processing initialization: {e}")
            raise

    def add_declarations(self, input_dict, var_modif):
        try:
            self.add_to_module = []
            self.add_to_routin = []
            self.add_to_usestm = []
            self.acc_declare_create = []
            self.acc_declare_copyin = []
            self.reads_in_decleration_routine = []
            self.reads_in_read_routine = []
            self.write_stmt = []
            for key in sorted(input_dict):
                var_in_modif = False
                if key in var_modif:
                    var_in_modif = True
                for item in input_dict[key]:
                    is_dec_stmt = isinstance(item, F23.Type_Declaration_Stmt)
                    is_alo_stmt = isinstance(item, F23.Allocate_Stmt)
                    is_use_stmt = isinstance(item, F23.Use_Stmt)
                    if is_dec_stmt:
                        # for ACC
                        '''any_attr_spec = [case.string for case in walk(item, F23.Attr_Spec)]
                        is_array = walk(item, F23.Explicit_Shape_Spec)
                        if 'ALLOCATABLE' in any_attr_spec or is_array:
                            all_entity_names = walk(item, F23.Entity_Decl)
                            if 'PARAMETER' in any_attr_spec:
                                for entity_name in all_entity_names:
                                    self.acc_declare_copyin.append(entity_name.tostr())
                            else:
                                for entity_name in all_entity_names:
                                    self.acc_declare_create.append(entity_name.tostr())
                        '''
                        # 
                        if var_in_modif:
                            self.add_to_module.append(self.add_entity_to_declaration(item, var_modif))
                        else:
                            self.add_to_module.append(item)
                        all_entity_names = walk(item, F23.Entity_Decl)
                        initialized = walk(item, F23.Initialization)
                        attr_spec = walk(item, F23.Attr_Spec)
                        if not initialized:
                            for entity_name in all_entity_names:
                                self.acc_declare_create.append(entity_name.tostr())
                            if F23.Attr_Spec('ALLOCATABLE') not in attr_spec:
                                self.reads_in_decleration_routine.append(item)
                            else:
                                self.reads_in_read_routine.append(self.combine_allocate_declaration(input_dict[key]))
                        else:
                            for entity_name in all_entity_names:
                                for child in entity_name.children:
                                    if isinstance(child, F23.Name):
                                        self.acc_declare_copyin.append(child.tostr())
                    if is_alo_stmt:
                        #self.add_to_routin.append(item)
                        #if var_in_modif:
                        for allocation_stmt in self.add_entity_to_allocation(item, var_modif):
                            self.add_to_routin.append(allocation_stmt.children[0])
                        #else:
                        #    self.add_to_routin.append(item)
                    if is_use_stmt:
                        self.add_to_usestm.append(item)
            self.add_to_module = self.remove_intent_and_save(self.add_to_module)
            self.reads_in_decleration_routine = self.remove_intent_and_save(self.reads_in_decleration_routine)
            self.reads_in_decleration_routine = self.initialization_statement(self.reads_in_decleration_routine)
            self.reads_in_read_routine = self.initialization_statement(self.reads_in_read_routine)
            self.add_to_module = self.process_queue(self.add_to_module)
            if self.acc_declare_copyin:
                acc_declare_copyin_str = ', '.join([name for name in self.acc_declare_copyin])
                self.acc_declare_copyin_cmd = self.parse_fortran_comment(f"!$ACC DECLARE COPYIN({acc_declare_copyin_str})")
            if self.acc_declare_create:
                acc_declare_create_str = ', '.join([name for name in self.acc_declare_create])
                self.acc_declare_create_cmd = self.parse_fortran_comment(f"!$ACC DECLARE CREATE({acc_declare_create_str})")
                self.acc_update_device_cmd = self.parse_fortran_comment(f"!$ACC UPDATE DEVICE({acc_declare_create_str})")
            logging.info("Declarations and allocations processed successfully ")
        except Exception as e:
            logging.error(f"Failed to process declarations and allocations, Error: {e}")
            raise

    def process_queue(self, input_list):
        try:
            queue = deque(input_list)
            left_list = []
            middle_list = []
            right_list = []
            while queue:
                element = queue.popleft()
                attr_spec = [case.string for case in walk(element, F23.Attr_Spec)]
                array = walk(element, F23.Explicit_Shape_Spec)
                if 'PARAMETER' in attr_spec and not array:
                    initialization = walk(walk(element, F23.Initialization), F23.Name)
                    if initialization:
                        left_list.append(element)
                    else:
                        left_list.insert(0, element)
                elif 'ALLOCATABLE' in attr_spec or array:
                    right_list.append(element)
                else:
                    middle_list.append(element)
            combined_list = left_list + middle_list + right_list
            logging.info(f"Processing complete. PARAMETER elements sent to the left.")
            return list(combined_list)
        except Exception as e:
            logging.error(f"An error occurred while processing the queue: {e}")
            raise

    def compile_and_run(self, base_dir, modules_dir, mode="CPU"):
        target_module_dir_path = os.path.join(base_dir, modules_dir)
        for subdir in os.listdir(target_module_dir_path):
            subdir_path = os.path.join(target_module_dir_path, subdir)
            if os.path.isdir(subdir_path):
                print('\033[32m' + f"Compiling and running in {subdir_path}..." + '\033[0m')
                os.environ["SUBDIR_PATH"] = subdir_path
                os.environ["MODE"] = mode
                os.chdir(subdir_path)
                os.system("make clean -f {}".format(os.path.join(base_dir, "Makefile")))
                os.system("make -f {}".format(os.path.join(base_dir, "Makefile")))
                if os.path.exists(subdir):
                    print('\033[32m' + "Compilation process completed!" + '\033[0m')
                    dummy_bin = os.path.join(self.benchmark_dir, subdir, "dummy.bin")
                    global_bin = os.path.join(self.benchmark_dir, subdir, "global.bin")
                    if os.path.exists(dummy_bin) and os.path.exists(global_bin):
                        print('\033[32m' + "Benchmark files exist. Now running the unit tests ..." + '\033[0m')
                        os.system("./{}".format(subdir))
                        print('\033[32m' + f"Execution completed in {subdir_path}" + '\033[0m')
                    else:
                         if not os.path.exists(dummy_bin):
                             print('\033[31m' + f"Missing file: {dummy_bin}" + '\033[0m')
                         if not os.path.exists(global_bin):
                             print('\033[31m' + f"Missing file: {global_bin}" + '\033[0m')
                         print('\033[31m' + f"Benchmark files do not exist yet. Run the modified main code and then python executive.py " + '\033[0m')
                else:
                    print('\033[31m' + "Compilation failed or main_program not generated." + '\033[0m')
                    return 1
        os.chdir(base_dir)
        return 0

    def process_assign(self, side, assign):
        try:
            if isinstance(assign, F23.Part_Ref):
                array = walk(walk(assign, F23.Part_Ref), F23.Name)
                side.add(array[0].string)
            if isinstance(assign, F23.Name):
                scalar = walk(assign, F23.Name)
                side.add(scalar[0].string)
            logging.info("Successfully processed assignment")
        except Exception as e:
            logging.error(f"Failed to process assignment, Error: {e}")
            raise
