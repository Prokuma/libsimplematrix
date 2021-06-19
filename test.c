#include "matrix.h"
#include <stdio.h>

int main() {
    Matrix a, b, ab, inv_ab, tmp;
    Vector vectors[3];
    int i;
    float a_array[9] = {3.0, 0.0, 0.0, -2.0, -2.0, 4.0, 0.0, -1.0, 3.0};
    float b_array[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    float v_array[3] = {1.0, 1.0, 1.0};

    initialize_matrix(&a, 3, 3);
    initialize_matrix(&b, 3, 3);
    initialize_matrix(&ab, 3, 3);
    initialize_matrix(&inv_ab, 3, 3);
    initialize_matrix(&tmp, 3, 3);
    for(i = 0; i < 3; i++){
        initialize_vector(&vectors[i], 3);
    }
    printf("Initialized\n");

    copy_matrix_from_array(&a, a_array);
    copy_matrix_from_array(&b, b_array);
    copy_vector_from_array(&vectors[0], v_array);
    copy_vector_from_array(&vectors[1], v_array);
    copy_vector_from_array(&vectors[2], v_array);

    matrix_dot(&ab, &a, &b);
    printf("|ab|: %f\n", matrix_determinant(&ab));
    matrix_inverse(&inv_ab, &ab);
    
    for(i = 0; i < 3; i++){
        printf("Vector_%d\n", i);
        print_vector(&vectors[i]);
    }
    printf("AB\n");
    print_matrix(&ab);
    printf("AB^-1\n");
    print_matrix(&inv_ab);

    free_matrix(&a);
    free_matrix(&b);
    free_matrix(&ab);
    free_matrix(&inv_ab);
    free_matrix(&tmp);
    for(i = 0; i < 3; i++){
        free_vector(&vectors[i]);
    }
    return 0;
}
