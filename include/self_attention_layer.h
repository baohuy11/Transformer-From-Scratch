#ifndef TRANSFORMER_ATTENTION_H
#define TRANSFORMER_ATTENTION_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define VOCAB_SIZE 1000
#define EMBEDDING_DIM 512
#define MAX_SEQ_LENGTH 128

// FUNCTION PROTOTYPES

// Compute the dot product of two vectors.
float dot_product(float *a, float *b, int dim);

// Compute the softmax of an input array.
void softmax(float *input, float *output, int length);

// Initialize the token embedding matrix.
void initialize_embedding(float embedding[VOCAB_SIZE][EMBEDDING_DIM]);

// Generate the positional encoding matrix.
void generate_positional_encoding(float positional_encoding[VOCAB_SIZE][EMBEDDING_DIM]);

// Lookup embedding for a given token index.
void lookup_embedding(float embedding[VOCAB_SIZE][EMBEDDING_DIM], int token_index, float *output);

// Concatenate token embedding with positional encoding.
void concatenate_embeddings(float token_embedding[EMBEDDING_DIM], float positional_embedding[EMBEDDING_DIM], float *output);

// Multiply matrix A by matrix B and store the result in matrix C.
void matrix_multiply(float A[MAX_SEQ_LENGTH][EMBEDDING_DIM], float B[EMBEDDING_DIM][EMBEDDING_DIM], float C[MAX_SEQ_LENGTH][EMBEDDING_DIM], int rows_A, int cols_A, int cols_B);

// Initialize a weight matrix with random values.
void initialize_weight_matrix(float weight[EMBEDDING_DIM][EMBEDDING_DIM]);

// Compute self-attention using trainable weight matrices for queries (Q), keys (K), and values (V).
void self_attention(float input[MAX_SEQ_LENGTH][EMBEDDING_DIM], float output[MAX_SEQ_LENGTH][EMBEDDING_DIM], int seq_length);

// A feed forward layer composed of two linear transformations with a ReLU activation in between.
void feed_forward(float input[MAX_SEQ_LENGTH][EMBEDDING_DIM], float output[MAX_SEQ_LENGTH][EMBEDDING_DIM], int seq_length);

// Apply layer normalization over the input.
void layer_normalization(float input[MAX_SEQ_LENGTH][EMBEDDING_DIM], float output[MAX_SEQ_LENGTH][EMBEDDING_DIM], int seq_length);

#endif // TRANSFORMER_ATTENTION_H