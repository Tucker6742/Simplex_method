# Simplex table

## Get raw data

* Get the number of var in the problems
* Get the coefficient of the objective function
* Get the condition (max or min)
* Get the coefficient of each constrain
* Get the sign of each constrain
* Return the data

## Preprocessing

### Convert constrain

* Convert all right hand side of constrain to non negative number

### Add slack var

* For each equation
  * If <= add positive slack var
  * If >= add negative slack var
  * If = add no slack var

### Check basis var

* Calculate the number of var that are eligible to be basis (going from the end to the front)
* If there are enough basis var then skip the artificial step
* Else add artificial var

### Add artificial var

* For each row that doesn't have a basis, add a artificial var

## Solve

### If has artificial var (Phase 1)

* Solve a min problems to find another form of the problem that has basis without artificial var
* Get the result

### If has no artificial var (Phase 2)

* If going through phase 1, change the objective function so that there is no basis var in it.
* Solve the original problem
* Get the result

### How to solve

* Constrain_ls matrix and a row that is the objective function
* Find the original basis

  * Basis var is var that has a coefficient of 1 and the rest is 0
  * Take basis var from right to left priority
* Find the columns

  * In the last row
    * Find the smallest positive if it's a max problems
    * Else find the smallest negative if it's a min problems
* Find the rows

  * Divide the right hand side by the pivot column
  * Find the smallest positive number
  * If there isn't any positive number (0 or negative) then the problem is unbounded, if `min` then it's -inf and reverse for `max`
* Gauss-Jordan elimination

  * Divide the pivot row by the pivot element
  * Gauss-Jordan elimination all other rows
* Change the basis

  * In the row selected, change the basis to the pivot column

Repeat from the "Find the columns" until you can't find any columns

## Get the result

* In phase 1

  * If the basis in the final row has artificial var then the problem is infeasible
  * Else get the new simplex matrix by removing the artificial var columns
* In phase 2

  * If outside of the basis there is no zero in the last row then return the result by going through each row, find the basis var and set it to the right hand side of the constrain (this work even if the basis var consist of slack var)
  * Else there exist another solution
    * Select one of the other 0 col as basis var next
    * Do the solve step 1 more time if possible (there are positive numbers in the gaussian elimination)
    * Get another solution.
    * The final solution is the linear interpolation of the two solutions
    * `var_n = var_n_result_1 * t + var_n_result_2 * (1 - t)` `(0<=t<=1>)`

## Table example

| x1  | x2  | S1  | S2  | S3  | const | ratio     |
| --- | --- | --- | --- | --- | ----- | --------- |
| 1   | 1   | 1   | 0   | 0   | 55    | 55/1=55   |
| 2   | 3   | 0   | 1   | 0   | 120   | 120/2=60  |
| 12  | 30  | 0   | 0   | 1   | 960   | 960/12=80 |
| 3   | 4   | 0   | 0   | 0   | 0     |           |
