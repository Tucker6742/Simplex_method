from src.utils.data_utils import SimplexData


if __name__ == "__main__":
    test_input = SimplexData.get_input()
    solution = test_input.solve()
    solution.get_result()
