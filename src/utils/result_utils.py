from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.data_utils import SimplexData

import numpy as np


class SimplexResult:
    def __init__(self, simplex_data: "SimplexData", status, phase):
        self.simplex_data = simplex_data
        self.status = status
        self.phase = phase

        if self.status == "optimal":
            self.final_vars, self.final_result = (
                self.process_result_from_simplex_table()
            )
            self.final_vars = self.final_vars.round(3)
            self.final_result = self.final_result.round(3)
        elif self.status == "optimal-not-single":
            self.first_vars = self.simplex_data.first_solution
            self.second_vars, self.final_result = (
                self.process_result_from_simplex_table()
            )
            self.final_vars = np.stack(
                (self.first_vars, self.second_vars), axis=-1
            ).round(3)
            self.final_result = self.final_result.round(3)
        elif self.status == "unbounded":
            self.final_vars = None
            self.final_result = (
                np.inf if self.simplex_data.condition == "max" else -np.inf
            )
        elif self.status == "infeasible":
            self.final_vars = None
            self.final_result = None

    def __repr__(self):
        return (
            f"SimplexResult({self.simplex_data.__dict__}, {self.status}, {self.phase})"
        )

    def process_result_from_simplex_table(self):
        result_var = np.zeros(self.simplex_data.number_var)
        basis_result_var = np.intersect1d(
            self.simplex_data.basis, np.arange(self.simplex_data.number_var)
        )
        row, col = np.where(self.simplex_data.simplex_table[:, basis_result_var] == 1)
        sort_order = np.argsort(col)
        row = row[sort_order]
        result_var[basis_result_var] = self.simplex_data.simplex_table[row, -2]
        result = (
            self.simplex_data.objective[: self.simplex_data.number_var] @ result_var
        )
        return result_var, result

    def get_result(self, return_value=False):
        if self.status == "optimal":
            result_str = self.build_str(self.final_vars, self.final_result)
            print(result_str)
            if return_value:
                return self.final_vars, self.final_result
        elif self.status == "optimal-not-single":
            result_str = self.build_str(
                self.second_vars, self.final_result, self.first_vars
            )
            print(result_str)
            if return_value:
                return self.result_var, self.final_result
        elif self.status == "unbounded":
            result_str = self.build_str()
            print(result_str)
            if return_value:
                return None, self.final_result
        elif self.status == "infeasible":
            result_str = self.build_str()
            print(result_str)
            if return_value:
                return None, None
        else:
            raise NotImplementedError("Not implemented yet")

    def build_str(self, vars=None, result=None, another_vars=None):
        if self.status == "optimal":
            optimal_str = (
                f"Optimal {self.simplex_data.condition} value: {np.round(result, 3)}"
            )
            var_str = ""
            for i, var in enumerate(vars):
                var_str += f"x{i+1} = {np.round(var, 3)}, "
            result_str = optimal_str + "\n" + var_str[:-2]
            return result_str
        elif self.status == "optimal-not-single":
            optimal_str = (
                f"Optimal {self.simplex_data.condition} value: {np.round(result, 3)}"
            )
            var_str = ""
            for i, var in enumerate(zip(another_vars, vars)):
                var_str += (
                    f"x{i+1} = {np.round(var[0], 3)}t + {np.round(var[1], 3)}*(1-t)\n"
                )
            var_str += "0<=t<=1"
            result_str = optimal_str + "\n" + var_str
            return result_str
        elif self.status == "unbounded":
            return f"Unbounded problem, no solution, {self.simplex_data.condition} value is {-np.inf if self.simplex_data.condition == 'min' else np.inf}"
        elif self.status == "infeasible":
            return "Infeasible problem, no solution"

    def __eq__(self, result: tuple[np.ndarray, float]) -> bool:
        if self.status == "optimal":
            return np.allclose(self.final_vars, result[0]) and np.isclose(
                self.final_result, result[1]
            )
        elif self.status == "optimal-not-single":
            return (
                np.allclose(self.final_vars, result[0])
                or np.allclose(self.final_vars[:, ::-1], result[0])
            ) and np.isclose(self.final_result, result[1])
        elif self.status == "unbounded":
            if self.simplex_data.condition == "max":
                return np.isposinf(result[1]) and result[0] is None
            elif self.simplex_data.condition == "min":
                return np.isneginf(result[1]) and result[0] is None
        elif self.status == "infeasible":
            return result[0] is None and result[1] is None
        else:
            raise NotImplementedError("Not implemented yet")
