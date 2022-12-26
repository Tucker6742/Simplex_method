import numpy as np
import pandas as pd
from tabulate import tabulate
import sys
np.set_printoptions(precision=3, suppress=True)


def Gauss_eliminate(table, pivot_row, pivot_column):
    for i in range(np.shape(table)[0]):
        if (i == pivot_row or table[i][pivot_column] == 0):
            continue
        else:
            coeff = -table[i][pivot_column]/table[pivot_row][pivot_column]
            table[i] = table[i]+coeff*table[pivot_row]


def normalize(table, pivot_row, pivot_column):
    table[pivot_row] = table[pivot_row]/table[pivot_row][pivot_column]


def update_and_print_out(display_table, constrain_ls, z, delta, basis_coeff, j):
    display_table.iloc[2:-2, 2:] = constrain_ls
    display_table.iloc[2:-2, 0] = basis_coeff
    display_table.iloc[2:-2, 1] = j
    display_table.iloc[-2, 2:] = z
    display_table.iloc[-1, 2:-1] = delta
    print(tabulate(display_table, showindex="never", tablefmt="fancy_grid"))

# Get objective function, constrains, signs,max/min, number of var,number of constrain


number_var = int(input(
    f"Maximum number of variables to evaluate (both in constrains and objective function): "))

print(f"Enter the coefficients of the objective function.")
objective = np.zeros(number_var)
for i in range(number_var):
    objective[i] = float(eval(input(f"Enter coefficients of x{i+1}: ")))

condition = input("Is this a max or a min problem? ")

number_constrain = 1
constrain_ls = np.zeros((1, number_var+1))
sign = np.zeros(1)
while True:
    print(f"Enter the coefficients of the {number_constrain} constrain.")
    for i in range(number_var):
        constrain_ls[-1,
                     i] = float(eval(input(f"Enter coefficients of x{i+1}: ")))

    while True:
        equal = input(
            "What is the equality of this constrain <=(leq), =(eq) or >=(geq)): ")
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

    constrain_ls[-1, -
                 1] = float(eval(input(f"Enter coefficients of the right side: ")))
    more = input("is there any other constrain(y/n)?")
    if (more == "n"):
        break
    else:
        number_constrain += 1
        constrain_ls = np.append(
            constrain_ls, np.zeros((1, number_var+1)), axis=0)
        sign = np.append(sign, 0)





# add slack var to objective and constrain_ls

zeros, = np.where(sign == 0)
slack_matrix = np.zeros((len(sign), len(sign)), dtype=np.float64)
np.fill_diagonal(slack_matrix, sign)
slack_matrix = np.delete(slack_matrix, (zeros), axis=1)
constrain_ls = np.insert(constrain_ls, -1, slack_matrix.T, axis=1)
objective = np.append(objective, np.zeros(np.shape(slack_matrix)[1]))

# convert constrains

fix_constrain = np.array(constrain_ls[:, -1] < 0)
count_fix_constrain = np.count_nonzero(fix_constrain)
if (count_fix_constrain != 0):
    pos_fix_constrain = np.where(fix_constrain == True)
    constrain_ls[pos_fix_constrain, :] = constrain_ls[pos_fix_constrain, :]*-1

# convert objective

if condition == "min":
    objective = objective*-1

# find basis

single_number = np.array([np.count_nonzero(constrain_ls[:, :-1], axis=0) == 1])
geq_0 = np.all(constrain_ls[:, :-1] >= 0, axis=0)
possible_basis = single_number & geq_0
count_possible_basis = np.count_nonzero(possible_basis)

if (count_possible_basis < number_constrain):

    # find row not OK
    list_row = np.arange(number_constrain)
    _, good_col = np.where(possible_basis == True)
    good_row, _ = np.where(constrain_ls[:, good_col] != 0)
    bad_row = [item for item in list_row if item not in good_row]

    # get sum of all error rows
    check_row = -np.sum(constrain_ls[bad_row, :], axis=0)
    constrain_ls = np.vstack((constrain_ls, check_row))

    # priority row to pivot
    priority = np.array([0]*3)
    # If last row has a negative number then repeat
    while (np.any(constrain_ls[-1, :-1] < 0)):
        # find most negative number there
        fixing_pivot_col = np.argmin(constrain_ls[-1, :-1])
        fixing_ratio = np.divide(
            constrain_ls[:-1, -1], constrain_ls[:-1, fixing_pivot_col])
        filter_fixing_ratio = np.extract(fixing_ratio > 0, fixing_ratio)
        filter_pivot_row = np.argmin(filter_fixing_ratio)
        possible_fixing_pivot_row = np.where(
            fixing_ratio == filter_fixing_ratio[filter_pivot_row])[0]
        for i in possible_fixing_pivot_row:
            if priority[i] == 0:
                fixing_pivot_row = i
                priority[i] = 1
                break
            else:
                continue
        normalize(constrain_ls, fixing_pivot_row, fixing_pivot_col)
        Gauss_eliminate(constrain_ls, fixing_pivot_row, fixing_pivot_col)

    if (not np.all(constrain_ls[-1, :] == 0)):
        print(f"There is 0 solution for this problems.")
        sys.exit()


    # export table
    constrain_ls = constrain_ls[:-1, :]

# Extract basis

single_number = np.array([np.count_nonzero(constrain_ls[:, :-1], axis=0) == 1])
geq_0 = np.all(constrain_ls[:, :-1] >= 0, axis=0)
possible_basis = single_number & geq_0
_, cols = np.where(possible_basis == True)
_, rows = np.where(constrain_ls[:, cols].T != 0)

# Normalize basis
count_basis = 0
for x, y in zip(rows, cols):
    normalize(constrain_ls, x, y)
    count_basis += 1
    if count_basis == number_constrain:
        break

# get basis coeff
value = [(r, c) for r, c in zip(rows, cols)]
basis_coeff = np.zeros(0)
count_coeff = 0
category = [('row', np.int64), ('col', np.int64)]
basis_point = np.array(value, dtype=category)
basis_point = np.sort(basis_point, order='row')
for i, j in basis_point:
    basis_coeff = np.append(basis_coeff, objective[j])
    count_coeff += 1
    if (count_coeff == number_constrain):
        break

j = np.array([i[1] for i in basis_point])
j = j.T

# make table

display_table = pd.DataFrame(columns=range(np.shape(constrain_ls)[
                             1]+2), index=range(np.shape(constrain_ls)[0]+4))

header = ["c_i"]
header.extend(list(map(str, objective)))
display_table.iloc[0, 1:-1] = header
name = ["c_j", "j", "result"]
name = name[:2]+["x"+str(i)
                 for i in range(np.shape(constrain_ls)[1]-1)] + name[2:]
display_table.iloc[1] = name
display_table.iloc[2:-2, 0] = basis_coeff
display_table.iloc[2:-2, 1] = j
display_table.iloc[2:-2, 2:] = constrain_ls
bottom = ["z", "delta"]
display_table.iloc[-2:, 1] = bottom
display_table = display_table.fillna("")

run_count = 1

# Run simplex method until end
while (True):
    print()
    print(f"{run_count} iteration")
    print()
    z = np.dot(constrain_ls.T, basis_coeff)
    delta = objective-z[:-1]

    copy_z = np.round(z, 3)
    copy_delta = np.round(delta, 3)
    copy_constrain_ls = np.round(constrain_ls, 3)
    update_and_print_out(display_table, copy_constrain_ls,
                         copy_z, copy_delta, basis_coeff, j)
    if (not np.any(delta > 0)):
        print()
        print(f"There is a solution for this problems")

        # Extract answer
        count_answer = np.array(
            [np.count_nonzero(constrain_ls[:, :-1], axis=0) == 1])
        answer = np.zeros(len(objective), dtype=np.float64)
        for i, bool in np.ndenumerate(count_answer):
            if bool:
                pos, = np.where(constrain_ls[:, i[1]] == 1)
                answer[i[1]] = constrain_ls[pos[0], -1]
        result = np.dot(
            answer, objective) if condition == "max" else -np.dot(answer, objective)
        answer = answer[:number_var]
        result = round(result, 12)
        print(f"The answer of this problems is {answer}")
        print(f"The {condition} value of this problem is {result}")
        break
    else:
        filter_col = np.extract(delta > 0, delta)
        filter_pivot_col = np.where(
            np.isinf(filter_col), -np.Inf, filter_col).argmax()
        index, = np.where(delta == filter_col[filter_pivot_col])
        pivot_col = index[0]
        ratio = np.divide(constrain_ls[:, -1], constrain_ls[:, pivot_col])
        filter_row = np.extract(ratio > 0, ratio)
        if (len(filter_row) == 0):
            print()
            print(f"Unable to pivot in column {pivot_col}\n")
            print(f"The problem is unbounded")
            sys.exit()

        filter_pivot_row = filter_row.argmin()
        index_r, = np.where(ratio == filter_row[filter_pivot_row])
        pivot_row = index_r[0]
        normalize(constrain_ls, pivot_row, pivot_col)
        Gauss_eliminate(constrain_ls, pivot_row, pivot_col)
        basis_coeff[pivot_row] = objective[pivot_col]

        copy_constrain_ls = np.round(constrain_ls, 3)
        copy_z = np.round(z, 3)
        copy_delta = np.round(delta, 3)
        j[pivot_row] = pivot_col
        print()
        print(f"Pivot point is at ({pivot_row}, {pivot_col})")
        run_count += 1
