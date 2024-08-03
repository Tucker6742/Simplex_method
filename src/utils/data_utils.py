import numpy as np
import polars as pl
from tabulate import tabulate
from dataclasses import dataclass
from pathlib import Path

np.seterr(divide="ignore", invalid="ignore")

from src.utils.result_utils import SimplexResult


@dataclass
class SimplexData:
    """
    A class to store the input of the simplex method.

    Behavior
    --------
    Read the input from the user and store it in a dataclass.

    Attributes
    ----------
    number_var : `int`
        The number of variables in the objective function and constrains.
    objective : `np.ndarray`
        The coefficients of the objective function
    condition : `str`
        The condition of the problem (max or min).
    number_constrain : `int`
        The number of constrains in the problem.
    constrain_ls : `np.ndarray`
        The coefficients of the constrains and the right side.
    sign : `np.ndarray`
        The sign of the constrains (1 for <=, 0 for =, -1 for >=).
    condition_sign : `int`
        The sign of the condition (1 for max, -1 for min).

    Class Methods
    -------
    get_input() -> `SimplexData`
        Read the input from the user and store it in a dataclass.

    Methods
    -------
    add_slack_var() -> `None`
        Add slack var to objective and constrain_ls
    """

    number_var: int
    objective: np.ndarray
    condition: str
    number_constrain: int
    constrain_ls: np.ndarray
    sign: np.ndarray
    condition_sign: np.ndarray

    @classmethod
    def get_input(cls):
        """
        Get the input from the user.

        Behavior
        --------
        Read the input from the user and store it in a data

        Returns
        -------
        - _ : `SimplexData`
            The input data stored in a dataclass.
        """

        number_var = int(
            input(
                f"Maximum number of variables to evaluate (both in constrains and objective function): "
            )
        )
        print(f"\nEnter the coefficients of the objective function.")
        objective = np.zeros(number_var)
        for i in range(number_var):
            objective[i] = float(eval(input(f"Enter coefficients of x{i+1}: ")))

        condition = input("Is this a max or a min problem? ")
        condition_sign = 1 if condition == condition else -1

        number_constrain = 1
        constrain_ls = np.zeros((1, number_var + 1))
        sign = np.zeros(1)

        while True:
            print(f"\nEnter the coefficients of the {number_constrain} constrain.")
            for i in range(number_var):
                constrain_ls[-1, i] = float(
                    eval(input(f"Enter coefficients of x{i+1}: "))
                )

            while True:
                equal = input(
                    "What is the equality of this constrain <=(leq), =(eq) or >=(geq)): "
                )
                match equal:
                    case "leq":
                        sign[-1] = 1
                        break
                    case "eq":
                        sign[-1] = 0
                        break
                    case "geq":
                        sign[-1] = -1
                        break
                    case _:
                        print("Value enter is not allowed try again.")

            constrain_ls[-1, -1] = float(
                eval(input(f"Enter coefficients of the right side: "))
            )
            more = input("is there any other constrain(y/n)?")
            if more == "n":
                break
            else:
                number_constrain += 1
                constrain_ls = np.append(
                    constrain_ls, np.zeros((1, number_var + 1)), axis=0
                )
                sign = np.append(sign, 0)

        return SimplexData(
            number_var,
            objective,
            condition,
            number_constrain,
            constrain_ls,
            sign,
            condition_sign,
        )

    @classmethod
    def read_from_txt(cls, file_path, condition, number):
        """
        Read the input from a text file.

        Behavior
        --------
        - Read the input from a text file and store it in a dataclass.
        - The format of the text file like an example below:
            1 -3 3
            3 -1 -2 <= 7
            -2 -4 4 <= 3
            1 0 2 <= 4
            -2 1 1 <=8
            [4 16 0] -44 min
            [5/4 0 11/8] 43/8 max

        Parameters
        ----------
        - file_path : `str`
            The path of the text file.
        - condition : `str`
            The condition of the problem (max or min).

        Returns
        -------
        - _ : `SimplexData`
            The input data stored in a dataclass.
        - Result :
        """
        with open(file_path, "r") as file:
            lines = []
            for all_lines in file:
                if all_lines == f"{number}.\n":
                    break
            for all_lines in file:
                lines.append(all_lines)
                if "max" in all_lines:
                    break

            number_constrain = len(lines) - 3
            number_var = len(lines[1].split()) - 2
            constrain_ls = np.zeros((number_constrain, number_var + 1))
            sign = np.zeros(number_constrain)
            for i, line in enumerate(lines[1:-2]):
                constrain_ls[i, :-1] = np.array(list(map(float, line.split()[:-2])))
                match line.split()[-2]:
                    case "<=":
                        sign[i] = 1
                    case ">=":
                        sign[i] = -1
                    case "=":
                        sign[i] = 0

                constrain_ls[i, -1] = float(line.split()[-1])

            objective = np.array(list(map(float, lines[0].split())))
            condition_sign = 1 if condition == condition else -1
            optimal_var = {"max": None, "min": None}
            optimal_val = {"max": None, "min": None}
            for line in lines[-2:]:
                var, component = line.split("]")
                var = var[1:].split()
                component = component.split()
                if len(var) != 0:
                    optimal_val[component[-1]] = eval(component[-2])
                    if "t" not in line:
                        optimal_var[component[-1]] = np.array(list(map(eval, var)))
                    else:
                        optimal_var[component[-1]] = np.empty(
                            number_var, dtype=np.dtypes.StringDType()
                        )
                        first_part = np.zeros(number_var)
                        second_part = np.zeros(number_var)
                        for i in range(number_var):
                            split_pos = var[i].index("t")
                            first_part[i] = np.round(eval(var[i][:split_pos]), 3)
                            second_part[i] = np.round(
                                eval(var[i][split_pos + 2 : -5]), 3
                            )
                        optimal_var[component[-1]] = np.stack(
                            (first_part, second_part), axis=-1
                        )
                elif "inf" in component[-2]:
                    optimal_val[component[-1]] = (
                        np.inf if component[-2] == "inf" else -np.inf
                    )
        return (
            SimplexData(
                number_var,
                objective,
                condition,
                number_constrain,
                constrain_ls,
                sign,
                condition_sign,
            ),
            optimal_var[condition],
            optimal_val[condition],
        )

    # ---------------------------------------- #
    def Gaussian_elimination(self, pivot_row, pivot_column):
        """
        Perform Gaussian elimination on the simplex table.

        Behavior
        --------
        - Perform Gaussian elimination on the simplex table.

        Parameters
        ----------
        pivot_row : `int`
            The row pivot of the simplex table.
        pivot_column : `int`
            The column pivot of the simplex table.

        Returns
        -------
        - _ : `None`
        """
        self.simplex_table[pivot_row] = (
            self.simplex_table[pivot_row] / self.simplex_table[pivot_row, pivot_column]
        )
        for i in range(self.simplex_table.shape[0]):
            if i == pivot_row:
                continue
            self.simplex_table[i] = (
                self.simplex_table[i]
                - self.simplex_table[i, pivot_column] * self.simplex_table[pivot_row]
            )

    # ---------------------------------------- #

    def convert_constrains(self):
        """
        Convert all constrain coefficient to non-negative values.

        Behavior
        --------
        - Find the constrains that have negative right side
        - If there is any, convert the whole row to positive.
        - Else do nothing.
        Returns
        -------
        - _ : `None`

        """
        fix_constrain = np.array(self.constrain_ls[:, -1] < 0)
        count_fix_constrain = np.count_nonzero(fix_constrain)
        if count_fix_constrain != 0:
            pos_fix_constrain = np.where(fix_constrain == True)
            self.constrain_ls[pos_fix_constrain, :] = (
                self.constrain_ls[pos_fix_constrain, :] * -1
            )
            self.sign[pos_fix_constrain] = self.sign[pos_fix_constrain] * -1

    def add_slack_var(self):
        """
        Add slack variables to the objective function and the constrain matrix.

        Behavior
        --------
        - Find the row that has equality sign.
        - Create a slack matrix for extra variables that match the sign of the coefficients.
        - Remove columns that have equality sign.
        - Add the slack matrix to the constrain matrix.
        - Padding the objective function with zeros to match the size of the constrain matrix.

        Returns
        -------
        - num_slack_var : `int`
            The number of slack variables added.

        """
        (equal_sign_row,) = np.where(self.sign == 0)
        slack_matrix = np.zeros((len(self.sign), len(self.sign)), dtype=np.float64)
        np.fill_diagonal(slack_matrix, self.sign)
        slack_matrix = np.delete(slack_matrix, (equal_sign_row), axis=1)

        # It has to be transpose, don't ask me why, it's how numpy works
        self.constrain_ls = np.insert(self.constrain_ls, -1, slack_matrix.T, axis=1)

        self.objective = np.pad(
            self.objective,
            (0, self.constrain_ls.shape[1] - self.objective.shape[0] - 1),
            "constant",
        )
        return slack_matrix.shape[1]

    def find_basis(self) -> None:
        """
        Find the original basis of the problem.

        Behavior
        --------
        - Find the original basis
            - Basis var is var that has a coefficient of 1 and the rest is 0 in the constrain matrix.
            - Take basis var from right to left priority
            - The number of basis var is equal to the number of constrains

        Create
        ------
        - self.basis : `np.ndarray`
            The original basis of the problem.

        Returns
        -------
        - _ : `None`
        """
        contain_one_number = np.count_nonzero(self.constrain_ls[:, :-1], axis=0) == 1

        contain_positive_number = np.all(self.constrain_ls[:, :-1] >= 0, axis=0)

        combine_filter = contain_one_number & contain_positive_number

        (possible_basis,) = np.where(combine_filter)

        self.basis = np.full((self.number_constrain,), -1)

        for col in possible_basis[::-1]:
            row = np.where(self.constrain_ls[:, col] == 1)[0][0]
            self.basis[row] = col
            if np.count_nonzero(self.basis == -1) == self.number_constrain:
                break

    def check_enough_basis(self):
        """
        Check if there are enough basis variable to solve the problem.

        Behavior
        --------
        - Check if the number of basis variable is less than the number of constrains.

        Returns
        -------
        - _ : `bool`
            Return True if there are enough basis variable, else False.
        """
        return np.count_nonzero(self.basis == -1) == 0

    def add_artificial_var(self):
        """
        Add artificial variables to the constrain that has geq (>=) sign.

        Behavior
        --------
        - For each row that doesn't have a basis variable, add an artificial variable.

        Returns
        -------
        - missing_row : `np.ndarray`
            The row that has an artificial variable.
        """
        usable_basis = self.basis[np.where(self.basis != -1)[0]]
        row_of_basis = np.where(self.constrain_ls[:, usable_basis] == 1)[0]
        missing_row = np.setdiff1d(np.arange(self.number_constrain), row_of_basis)
        artificial_matrix = np.zeros((self.number_constrain, len(missing_row)))
        for i, row in enumerate(missing_row):
            artificial_matrix[row, i] = 1

        self.constrain_ls = np.insert(
            self.constrain_ls, -1, artificial_matrix.T, axis=1
        )

        self.objective = np.pad(
            self.objective,
            (0, self.constrain_ls.shape[1] - self.objective.shape[0] - 1),
            "constant",
        )

        return (
            np.where(artificial_matrix)[0],
            np.where(artificial_matrix)[1] + self.number_var + self.num_slack_var,
        )

    def preprocess_problem(self):
        """
        Preprocess the problem before solving it.

        Behavior
        --------
        - Convert all constrain coefficient to non-negative values.
        - Add slack variables to the objective function and the constrain matrix.
        - Add artificial variables to the constrain that has geq (>=) sign.

        Create
        ------
        - self.artificial_var : `bool`
            Return True if there need to have artificial variables, else False.

        Returns
        -------
        - _ : `None`
        """
        self.convert_constrains()
        self.num_slack_var = self.add_slack_var()
        self.original_var_txt = [f"x{i+1}" for i in range(self.number_var)]
        self.slack_var_txt = [f"S{i+1}" for i in range(self.num_slack_var)]
        self.find_basis()
        self.header = self.original_var_txt + self.slack_var_txt
        if not self.check_enough_basis():
            self.artificial_var_row, self.artificial_var = self.add_artificial_var()
            self.artificial_var_txt = [
                f"R{i+1}" for i in range(len(self.artificial_var))
            ]
            self.header += self.artificial_var_txt

        self.header += ["Const", "Ratio"]

    # ---------------------------------------- #

    def create_simplex_table(self) -> None:
        """
        Create the simplex table.

        Behavior
        --------
        - Create a table that has the objective function and constrains. with extra space for the gaussian elimination.

        Create
        ------
        - self.simplex_table : `np.ndarray`
            The simplex table that has the objective function and constrains.

        Returns
        -------
        - _ : `None`
        """
        constrain_ls_shape = self.constrain_ls.shape
        self.simplex_table = np.full(
            (constrain_ls_shape[0] + 1, constrain_ls_shape[1] + 1),
            np.nan,
        )
        self.simplex_table[:-1, :-1] = self.constrain_ls

        if hasattr(self, "artificial_var"):
            artificial_var_row = self.artificial_var_row
            aux_matrix = -self.simplex_table[:-1, :-1]
            aux_matrix = aux_matrix[artificial_var_row]
            aux_matrix[:, -2 : -2 - len(self.artificial_var) : -1] = 0

            self.simplex_table[-1, :-1] = np.sum(aux_matrix, axis=0)
            self.basis[artificial_var_row] = self.artificial_var
        else:
            self.change_objective_phase_2()

    def find_column_pivot(self, condition="min") -> int:
        """
        Find the column pivot of the simplex table.

        Behavior
        --------
        - Find the column pivot by finding the the smallest negative if it's a min problems else find the largest positive

        Returns
        -------
        - chosen_columns : `int`
            The column pivot of the simplex table.
        """

        if condition == "min":
            possible_col = np.where(self.simplex_table[-1, :-2] < 0)[0]
            if len(possible_col) == 0:
                return -1
            min_val = np.min(self.simplex_table[-1, possible_col])
            chosen_column = np.where(self.simplex_table[-1, possible_col] == min_val)[
                0
            ][0]
        else:
            possible_col = np.where(self.simplex_table[-1, :-2] > 0)[0]
            if len(possible_col) == 0:
                return -1
            max_val = np.max(self.simplex_table[-1, possible_col])
            chosen_column = np.where(self.simplex_table[-1, possible_col] == max_val)[
                0
            ][0]
        return possible_col[chosen_column]

    def find_row_pivot(self, column_pivot) -> int:
        """
        Find the row pivot of the simplex table.

        Behavior
        --------
        - Find the row pivot by finding the smallest positive ratio.

        Parameters
        ----------
        column_pivot : `int`
            The column pivot of the simplex table.

        Returns
        -------
        - chosen_row : `int`
            The row pivot of the simplex table.
        """
        ratio = self.simplex_table[:-1, -2] / self.simplex_table[:-1, column_pivot]
        ratio[ratio == np.inf] = -np.inf
        zero_rows = np.where(ratio == 0)[0]
        positive_rows = np.where(ratio > 0)[0]
        if len(positive_rows) > 0:
            min_val = np.min(ratio[positive_rows])
            chosen_row = np.where(ratio == min_val)[0][-1]
        elif len(zero_rows) > 0:
            chosen_row = np.argmax(self.simplex_table[zero_rows, column_pivot])
            chosen_row = zero_rows[chosen_row]
        else:
            return -1
        return chosen_row

    def change_basis(self, pivot_row, pivot_column):
        """
        Change the basis of the problem.

        Behavior
        --------
        - Change the basis of the problem by changing the basis variable.

        Parameters
        ----------
        pivot_row : `int`
            The row pivot of the simplex table.
        pivot_column : `int`
            The column pivot of the simplex table.

        Returns
        -------
        - _ : `None`
        """
        self.basis[pivot_row] = pivot_column

    def drop_artificial_var(self):
        """
        Drop the artificial variable from the simplex table.

        Behavior
        --------
        - Drop the artificial variable from the simplex table.

        Returns
        -------
        - _ : `None`
        """
        self.simplex_table = np.delete(
            self.simplex_table, slice(-3, -3 - len(self.artificial_var), -1), axis=1
        )
        self.objective = np.delete(
            self.objective, slice(-1, -1 - len(self.artificial_var), -1)
        )
        del self.header[-3 : -3 - len(self.artificial_var) : -1]

    def change_objective_phase_2(self):
        """
        Change the objective function to phase 2.

        Behavior
        --------
        - Change the objective function to phase 2 by replacing basis var with non basis var.

        Returns
        -------
        - _ : `None`
        """

        basis_var_coeff_objective = self.objective[self.basis]

        aux_matrix = -self.simplex_table[:-1, :-1]
        aux_matrix[:, self.basis] = 0
        aux_matrix = aux_matrix * basis_var_coeff_objective[:, np.newaxis]
        formatted_objective = self.objective.copy()
        formatted_objective[self.basis] = 0
        self.simplex_table[-1, :-1] = np.sum(aux_matrix, axis=0) + np.append(
            formatted_objective, 0
        )

    def solve_phase_1(self):
        """
        Solve the problem using the simplex method.

        Behavior
        --------
        - Create the simplex table.
        - Find the original basis
        - Find the columns
        - Find the rows
        - Perform Gaussian elimination
        - Change the basis
        - Repeat thr find column step until there are no columns

        Returns
        -------
        - _ : `SimplexResult`
            The result of the simplex method.
        """
        self.create_simplex_table()
        self.display()
        while True:
            column_pivot = self.find_column_pivot()
            if column_pivot == -1:
                break
            row_pivot = self.find_row_pivot(column_pivot)
            if row_pivot == -1:
                return SimplexResult(self, "unbounded", phase=1)
            self.display(column_pivot, row_pivot)
            self.Gaussian_elimination(row_pivot, column_pivot)
            self.display()
            self.change_basis(row_pivot, column_pivot)
        if self.check_infeasible():
            return SimplexResult(self, "infeasible", phase=1)
        return SimplexResult(self, "optimal", phase=1)

    def check_infeasible(self):
        """
        Check if the problem is infeasible.

        Behavior
        --------
        - Check if the problem is infeasible.

        Returns
        -------
        - _ : `bool`
            Return True if the problem is infeasible, else False.
        """
        return (
            np.intersect1d(
                self.basis,
                np.arange(
                    self.number_var + self.num_slack_var,
                    self.simplex_table.shape[1] - 2,
                ),
            ).size
            > 0
        )

    def not_only_one_result(self):
        """
        Check if there are more than one result.

        Behavior
        --------
        - Check if there are more than one result.

        Returns
        -------
        - _ : `bool`
            Return True if there are more than one result, else False.
        """
        non_basis_var = np.setdiff1d(
            np.arange(self.simplex_table.shape[1] - 2), self.basis
        )
        return np.count_nonzero(self.simplex_table[-1, non_basis_var] == 0) > 0

    def solve_for_another_var(self):
        """
        Solve the problem for another variable.

        Behavior
        --------
        - Solve the problem for another variable.

        Returns
        -------
        - _ : `None`
        """
        self.first_solution = np.zeros(self.number_var)
        first_basis_var_pos = np.where(self.basis < self.number_var)[0]
        first_basis_var = self.basis[first_basis_var_pos]
        first_basis_val = self.simplex_table[first_basis_var_pos, -2]
        # sort_order = np.argsort(first_basis_var)
        # first_basis_val = first_basis_val[sort_order]
        self.first_solution[first_basis_var] = first_basis_val

        possible_col = np.setdiff1d(
            np.arange(self.simplex_table.shape[1] - 2), self.basis
        )
        next_col = possible_col[self.simplex_table[-1, possible_col] == 0][0]
        row_pivot = self.find_row_pivot(next_col)
        self.display(next_col, row_pivot)
        self.Gaussian_elimination(row_pivot, next_col)
        self.display()
        self.change_basis(row_pivot, next_col)

    def solve_phase_2(
        self, phase_1_result: None | SimplexResult = None
    ) -> SimplexResult:
        """
        Solve the problem using the simplex method.

        Behavior
        --------
        - Create the simplex table.
        - Find the original basis
        - Find the columns
        - Find the rows
        - Perform Gaussian elimination
        - Change the basis
        - Repeat thr find column step until there are no columns

        Returns
        -------
        - _ : `SimplexResult`
            The result of the simplex method.
        """
        if phase_1_result is not None:
            if phase_1_result.status != "optimal":
                return SimplexResult(self, "infeasible", phase=2)
            else:
                self.drop_artificial_var()
                self.change_objective_phase_2()
                self.display()
        else:
            self.create_simplex_table()
            self.display()
        while True:
            column_pivot = self.find_column_pivot(self.condition)
            if column_pivot == -1:
                break
            row_pivot = self.find_row_pivot(column_pivot)
            if row_pivot == -1:
                return SimplexResult(self, "unbounded", phase=2)
            self.display(column_pivot, row_pivot)
            self.Gaussian_elimination(row_pivot, column_pivot)
            self.display()
            self.change_basis(row_pivot, column_pivot)

        if self.not_only_one_result():
            self.solve_for_another_var()
            return SimplexResult(self, "optimal-not-single", phase=2)
        else:
            return SimplexResult(self, "optimal", phase=2)

    def solve(self) -> SimplexResult:
        """
        Solve the problem using the simplex method.

        Behavior
        --------
        - Preprocess the problem.
        - Solve the problem using phase 1.
        - Solve the problem using phase 2.

        Returns
        -------
        - _ : `SimplexResult`
            The result of the simplex method.
        """
        self.preprocess_problem()
        if hasattr(self, "artificial_var"):
            print("\nSolving the problem using phase 1\n")
            phase_1_result = self.solve_phase_1()
            if phase_1_result.status == "infeasible":
                return phase_1_result
            print("\nSolving the problem using phase 2\n")
            phase_2_result = self.solve_phase_2(phase_1_result)
        else:
            print("\nSolving the problem using phase 2\n")
            phase_2_result = self.solve_phase_2()
        return phase_2_result

    def display(self, pivot_col=None, pivot_row=None):
        """
        Display the current simplex table

        Behavior
        --------
        - Display the current simplex table

        Returns
        -------
        - _ : `None`
        """
        table = pl.DataFrame(self.simplex_table.round(3), schema=self.header)
        table = table.fill_nan("")
        if pivot_col is not None and pivot_row is not None:
            ratio_txt = [
                f"{np.round(const, 3)}/{np.round(var, 3)} = {np.round(const/var, 3)}"
                for const, var in zip(
                    self.simplex_table[:-1, -2], self.simplex_table[:-1, pivot_col]
                )
            ] + [""]
            table = table.with_columns(Ratio=pl.Series(ratio_txt))
            print(
                f"\nPivot point is at row {pivot_row+1} and column {pivot_col+1}, variable {self.header[pivot_col]}\n\n"
            )
            # print(table)
            print(
                tabulate(
                    table.transpose(),
                    headers=self.header,
                    tablefmt="rounded_grid",
                    # floatfmt=".2f",
                )
            )
        else:
            # print(table)
            print(
                tabulate(
                    table.transpose(),
                    headers=self.header,
                    tablefmt="rounded_grid",
                    # floatfmt=".2f",
                )
            )
