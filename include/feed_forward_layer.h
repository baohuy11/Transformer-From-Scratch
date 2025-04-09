#ifndef READ_WEIGHTS_H
#define READ_WEIGHTS_H

#include <stdio.h>

#define NUM_WEIGHTS 512

void read_weights(const char* path, double* semi_final_wegihts, int num_weights);

#endif // READ_WEIGHTS_H