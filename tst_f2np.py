IF ((free_drain_coef(ji,jst) .GE. 0.5) .AND. (.NOT. ok_freeze_cwrr) ) THEN

IF (infilt_tot(ji) .LT. -min_sechiba .OR. infilt_tot(ji) .GT. flux_infilt(ji) + min_sechiba) THEN

ELSE IF ( (stempdiag(ji,jsl) .GE. (fr_center-fr_dT/2.)) .AND. (stempdiag(ji,jsl) .LT. (fr_center+fr_dT/2.)) ) THEN

    def handle_if_construct(self, stmt):
        assert isinstance(stmt, F23.If_Construct), (
                f"Unexpected statement type: {type(stmt).__name__}. Expected one of: "
                f"If_Construct")
        self.indentation_level = 0
        pycode = ""
        for child in stmt.children:
            if isinstance(child, F23.If_Then_Stmt):
                condition = self.handle_if_condition(child)
                pycode += f"{self.indentation_level * '    '}{condition}\n"
            elif isinstance(child, (F23.Else_If_Stmt, F23.Else_Stmt)):
                self.indentation_level -= 1
                condition = self.handle_if_condition(child)
                pycode += f"{self.indentation_level * '    '}{condition}\n"
            elif isinstance(child, F23.End_If_Stmt):
                self.indentation_level -= 1
                condition = self.handle_end_stmt(child)
                pycode += f"{self.indentation_level * '    '}{condition}\n"
            else:
                self.indentation_level += 1
                pycode += f"{self.indentation_level * '    '}{str(child).strip()}\n"
        return pycode

