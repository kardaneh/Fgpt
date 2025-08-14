from fparser.two import Fortran2003 as F23
from collections import deque
from processor import Processor
from itertools import tee, zip_longest
from fparser.two.utils import walk

class Optimizer:
    def __init__(self, block):
        self.block_opt = block
        self.do_index = 0
        self.if_index = 0
        self.enddo_index = 0
        self.endif_index = 0
        self.current_do_info = []
        self.current_if_info = []
        self.current_if_cond = []
        self.current_doif_info = []
        self.assignment_stmt = {}
        self.nested_do_info = {}
        self.nested_if_info = {}
        self.nested_doif_info = {}
        self.argdo = ''
        self.argif = ''
        self.arg = 'assignment'

    def find_assignment_stmt(self):
        self._search_for_assignment_stmt(self.block_opt, self.arg)
        assert self.assignment_stmt.keys() == self.nested_do_info.keys()
        assert self.nested_if_info.keys() == self.nested_do_info.keys()
        assert self.current_do_info == []
        assert self.current_if_info == []
        assert self.current_if_cond == []
        assert self.current_doif_info == []


    def _search_for_assignment_stmt(self, exec_part, key):
        if isinstance(exec_part, F23.Assignment_Stmt):
            line_number = exec_part.item.span[0]
            key = key + ', ' + f"assignment: {line_number}"
            if key not in self.assignment_stmt:
                self.assignment_stmt[key] = []
            self.assignment_stmt[key].append(exec_part)
            self.nested_do_info[key] = [item for item in self.current_do_info]
            self.nested_if_info[key] = [item for item in self.current_if_cond]
            self.nested_doif_info[key] = [item for item in self.current_doif_info]
        elif isinstance(exec_part, F23.Block_Nonlabel_Do_Construct):
            for construct in exec_part.content:
                if isinstance(construct, F23.Nonlabel_Do_Stmt):
                    self.do_index += 1
                    line_number = construct.item.span[0]
                    line_parts = construct.tostr().split('=')
                    loop_index = line_parts[0].split()[-1]
                    start_end_stride_values = line_parts[1].split(',')
                    loop_start = start_end_stride_values[0].strip()
                    loop_end = start_end_stride_values[1].strip()
                    if len(start_end_stride_values)==2:
                        loop_stride = 1
                    elif len(start_end_stride_values)==3:
                        loop_stride = start_end_stride_values[2].strip()
                    else:
                        raise ValueError("Loop control error!")
                    self.current_do_info.append({'line_number':line_number, 'loop_index':loop_index, 'loop_start':loop_start, \
                            'loop_end':loop_end, 'loop_stride':loop_stride})
                    self.current_doif_info.append(construct)
                elif isinstance(construct, F23.End_Do_Stmt):
                    self.enddo_index += 1
                    self.current_do_info = self.current_do_info[:self.do_index - self.enddo_index]
                    self.current_doif_info = self.current_doif_info[:self.do_index - self.enddo_index]
                    if self.enddo_index == self.do_index:
                        self.do_index = 0
                        self.enddo_index = 0
                else:
                    self.argdo = ', '.join([f"DO {item['loop_index']}: {item['line_number']}" for item in self.current_do_info])
                    if self.endif_index == self.if_index:
                        self.arg = self.argdo
                    else:
                        self.arg = self.argif + ', ' + self.argdo
                    #self.arg = self.sort_key_value_string(self.arg)
                    self._search_for_assignment_stmt(construct, self.arg)
        elif isinstance(exec_part, F23.If_Construct):
            for construct in exec_part.content:
                if isinstance(construct, F23.If_Then_Stmt):
                    self.if_index += 1
                    self.current_if_info.append({'If_Then_Stmt': construct.item.span[0]})
                    self.current_if_cond.append(construct)
                    self.current_doif_info.append(construct)
                elif isinstance(construct, F23.Else_If_Stmt):
                    self.if_index += 1
                    self.endif_index += 1
                    self.current_if_info[self.if_index - self.endif_index -1] = {'Else_If_Stmt': construct.item.span[0]}
                    self.current_if_cond[self.if_index - self.endif_index -1] = construct
                    self.current_doif_info.append(construct)
                elif isinstance(construct, F23.Else_Stmt):
                    self.if_index += 1
                    self.endif_index += 1
                    self.current_if_info[self.if_index - self.endif_index -1] = {'Else_Stmt': construct.item.span[0]}
                    self.current_if_cond[self.if_index - self.endif_index -1] = construct
                    self.current_doif_info.append(construct)
                elif isinstance(construct, F23.End_If_Stmt):
                    self.endif_index += 1
                    self.current_if_info = self.current_if_info[:self.if_index - self.endif_index]
                    self.current_if_cond = self.current_if_cond[:self.if_index - self.endif_index]
                    self.current_doif_info = self.current_doif_info[:self.if_index - self.endif_index]
                    if self.endif_index == self.if_index:
                        self.if_index = 0
                        self.endif_index = 0
                else:
                    self.argif = ', '.join(f"{list(item.keys())[0]}: {list(item.values())[0]}" for item in self.current_if_info)
                    if self.enddo_index == self.do_index:
                        self.arg = self.argif
                    else:
                        self.arg = self.argdo + ', ' + self.argif
                    #self.arg = self.sort_key_value_string(self.arg)
                    self._search_for_assignment_stmt(construct, self.arg)
        elif hasattr(exec_part, "content"):
            self.arg = 'assignment'
            for construct in exec_part.content:
                self._search_for_assignment_stmt(construct, self.arg)

    def generate_execution_part(self):
        execution_part = []
        merged_dos = F23.Nonlabel_Do_Stmt('DO ji = 1, kjpindex')
        execution_part.append(merged_dos)
        keys, next_keys = tee(self.assignment_stmt.keys())
        next(next_keys, None)

        for key, next_key in zip_longest(keys, next_keys, fillvalue=None):
            doif_ends = deque()
            for item in self.nested_doif_info[key]:
                if isinstance(item, F23.Nonlabel_Do_Stmt):
                    if item.tostr() != merged_dos.tostr():
                        execution_part.append(item)
                        doif_ends.append(F23.End_Do_Stmt('ENDDO'))
                if isinstance(item, F23.If_Then_Stmt) or isinstance(item, F23.Else_If_Stmt) or isinstance(item, F23.Else_Stmt):
                    execution_part.append(item)
                    if next_key==None or (not walk(self.nested_doif_info[next_key], F23.Else_Stmt) and \
                            not walk(self.nested_doif_info[next_key], F23.Else_If_Stmt)):
                                doif_ends.append(F23.End_If_Stmt('ENDIF'))

            for item in self.assignment_stmt[key]:
                execution_part.append(item)

            while doif_ends:
                item = doif_ends.pop()
                execution_part.append(item)

        execution_part.append(F23.End_Do_Stmt('ENDDO'))

        for child in self.block_opt.children:
            if isinstance(child, F23.Execution_Part):
                child.children.clear()
                ldx = len(child.content) - 1
                for stmt in execution_part:
                    child.children.insert(ldx + 1, stmt)
                    ldx += 1
        self.block_opt = Processor().parse_fortran_string(self.block_opt.tofortran())
        return execution_part

    def sort_key_value_string(self, s):
        components = s.split(', ')
        kv_pairs = [(int(comp.split(': ')[1]), comp) for comp in components]
        sorted_kv_pairs = sorted(kv_pairs)
        sorted_components = [kv[1] for kv in sorted_kv_pairs]
        sorted_string = ', '.join(sorted_components)
        return sorted_string
