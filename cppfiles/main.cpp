#include<iostream>
#include<stdlib.h>



class MatrixOperations
{
    public:
        void matrix_mul(int **result_matrix, int **matrix_one, int **matrix_two, const int MATRIX_ONE_ROWS, const int MATRIX_TWO_ROWS, const int MATRIX_ONE_COLUMNS, const int MATRIX_TWO_COLUMNS){
            if (MATRIX_ONE_ROWS == MATRIX_TWO_COLUMNS){
                for (int iter_i = 0; iter_i < MATRIX_ONE_ROWS; iter_i ++)
                {
                    int stat_iter = 0;
                    for(int iter_j = 0; iter_j < MATRIX_ONE_COLUMNS; iter_j ++)
                    {
                        result_matrix[iter_i][stat_iter] = matrix_one[iter_i][stat_iter] + matrix_two[stat_iter][iter_i];
                        std::cout << "curent value for [result_matrix]:\t" << result_matrix[iter_i][stat_iter];
                    }
                    stat_iter ++;
                }
            }
            else{std::cout << "\nerror mismatch with shapes\n";}
        }
};

int main(){

    const int SIZE = 123;
    int matrix_for_test[SIZE];
    int *pmatrix_for_test = matrix_for_test;

    std::cout << "\n---------------\n" <<"array head adress by [matrix_for_test] item:\t" << matrix_for_test << "\n============\n" << "array head adress by [pmatrix_for_test] item:\t" << pmatrix_for_test << "\n---------------\n";
    for (int iter = 0; iter < SIZE; iter ++){matrix_for_test[iter] = rand() % 30;}
    for (int iter = 0; iter < SIZE; iter ++){std::cout << "============\n" << "adress:\t" << &pmatrix_for_test[iter] << "\n" << "iter:\t" << matrix_for_test[iter] << "\n============";}

    return 0;

}