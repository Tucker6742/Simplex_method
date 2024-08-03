from pathlib import Path
import numpy as np
from src.utils.data_utils import SimplexData
import pytest

all_test_cases = []
ids = []
NUM_TEST_CASES = 18

for i in range(NUM_TEST_CASES):
    for condition in ["max", "min"]:
        test_input, optimal_var, optimal_val = SimplexData.read_from_txt(
            Path(f"test/test_case.txt"), condition, i + 1
        )
        all_test_cases.append((test_input, optimal_var, optimal_val))
        ids.append(f"test_case_{i+1}_{condition}")


@pytest.mark.parametrize(
    "test_input, optimal_var, optimal_val", all_test_cases, ids=ids
)
def test_solve(test_input: SimplexData, optimal_var, optimal_val):
    solution = test_input.solve()
    solution.get_result()
    # print(solution.final_vars, solution.final_result)
    # print(optimal_var, optimal_val)
    assert solution == (optimal_var, optimal_val)


if __name__ == "__main__":
    chosen_test_case = 5
    condition = "max"
    offset = 0 if condition == "max" else 1
    test_solve(
        all_test_cases[(chosen_test_case - 1) * 2 + offset][0],
        all_test_cases[(chosen_test_case - 1) * 2 + offset][1],
        all_test_cases[(chosen_test_case - 1) * 2 + offset][2],
    )
