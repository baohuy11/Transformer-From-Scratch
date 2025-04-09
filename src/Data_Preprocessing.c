#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../include/Data_Preprocessing.h"

#define MAXTRIX_SIZE 2
#define EMBEDDING_SIZE 512

void scale_matrix(float matrix[EMBEDDING_SIZE][MAXTRIX_SIZE]){
    float min_val = matrix[0][0];
    float max_val = matrix[0][0];

    for(int i = 0; i < EMBEDDING_SIZE; i++){
        for(int j = 0; j < MAXTRIX_SIZE; j++){
            if(matrix[i][j] < min_val) min_val = matrix[i][j];
            if(matrix[i][j] > max_val) max_val = matrix[i][j];
        }
    }

    if(min_val == max_val){
        fprintf(stderr, "All values in the matrix are the same. Scaling is not possible.\n");
        return;
    }

    for(int i = 0; i < MATRIX_SIZE; i++){
        for(int j = 0; j < MAXTRIX_SIZE; j++){
            matrix[i][j] = 2 * (matrix[i][j] - min_val) / (max_val - min_val) - 1;
            if(matrix[i][j] == 0) matrix[i][j] = 0.01; // Avoid zero values
        }
    }
}

void min_max_normalize_and_scale(float* data, size_t size, float new_min, float new_max){
    if(data == NULL || size == 0) return;

    float old_min = FLT_MAX;
    float old_max = -FLT_MAX;

    for(size_t i = 0; i < size; i++){
        if(data[i] < old_min) old_min = data[i];
        if(data[i] > old_max) old_max = data[i];
    }
    if(old_max == old_min){
        for(size_t i = 0; i < size; i++) data[i] = new_min; 
        return;
    }
    /* Normalize to [0, 1] */
    for(size_t i = 0; i < size; i++){
        data[i] = (data[i] - old_min) / (old_max - old_min);
    }
    /* Scale to the new range [new_min, new_max] */
    for(size_t i = 0; i < size; i++){
        data[i] = new_min + (data[i] * (new_max - new_min));
        if(data[i] == 0) data[i] = 0.01;
    }
}

size_t get_meaningful_length(const float* data, size_t size){
    size_t start = 0;
    size_t end = size;

    /* Find the start of meaningfull data */
    while(start < size && data[start] == 0){
        start++;
    }

    if(start == size) return 0; // All values are zero

    /* Find the end of meaningful data */
    end = size;
    while(end > start && data[end - 1] == 0){
        end--;
    }

    return end - start;
}

void Add_Positional_Encoding(float embedding_matrix[][2], int max_sentence_length){
    for(int i = 0; i < max_sentence_length; i++){
        if(!(embedding_matrix[i][0] == 0.0f && embedding_matrix[i][1] == 0.0f)){
            float sin_val = sin(i / 2.0f);
            float cos_val = cos(i / 2.0f);

            embedding_matrix[i][0] += sin_val;
            embedding_matrix[i][1] += cos_val;
        }
    }
}