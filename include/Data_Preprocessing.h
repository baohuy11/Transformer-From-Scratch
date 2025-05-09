#ifndef DATA_PREPROCESSING_H
#define DATA_PREPROCESSING_H

#include <stddef.h>
#include <float.h> 

#define MATRIX_SIZE 2
#define EMBEDDING_SIZE 512

void min_max_normalize_and_scale(float* data, size_t size, float new_min, float new_max);

size_t get_meaningful_length(const float* data, size_t size);

void Add_Positional_Encoding(float embedding_matrix[][MATRIX_SIZE], int max_sentence_length);

void scale_matrix(float matrix[EMBEDDING_SIZE][MATRIX_SIZE]);

#endif /* DATA_PREPROCESSING_H */