# Transformer Model Implementation in C

This repository contains a C implementation of a Transformer model, focusing on the core components of the Transformer architecture including self-attention mechanisms, positional encoding, and feed-forward networks.

## Overview

The implementation includes the following key components:

- **Data Processing**
  - Text loading and cleaning
  - Tokenization
  - Word embedding generation

- **Transformer Architecture**
  - Self-attention mechanism
  - Positional encoding
  - Feed-forward networks
  - Backpropagation and weight updates

- **Model Training**
  - Loss calculation
  - Gradient clipping
  - Weight optimization

## Project Structure

```
Transformer-From-Scratch/
├── include/                 # Header files
│   ├── transformer_block.h  # Transformer block implementation
│   ├── self_attention_layer.h
│   ├── feed_forward_layer.h
│   ├── tokenizer.h
│   ├── utils.h
│   ├── backprop.h
│   ├── activation_functions.h
│   ├── Data_Preprocessing.h
│   └── Data_Loading_Cleaning.h
├── src/                    # Source files
│   ├── transformer_block.c
│   ├── self_attention_layer.c
│   ├── feed_forward_layer.c
│   ├── tokenizer.c
│   ├── utils.c
│   ├── backprop.c
│   ├── activation_functions.c
│   ├── Data_Preprocessing.c
│   └── Data_Loading_Cleaning.c
├── examples/              # Example code
│   └── main.c            # Main training loop
└── test_data.txt         # Sample training data
```

## Key Features

- **Self-Attention Mechanism**: Implements scaled dot-product attention
- **Positional Encoding**: Adds positional information to embeddings
- **Feed-Forward Networks**: Implements non-linear transformations
- **Backpropagation**: Includes gradient computation and weight updates
- **Activation Functions**: Implements various activation functions including LeakyReLU and Swish

## Requirements

- C compiler (GCC recommended)
- OpenMP (for parallel processing)
- Standard C libraries
- Math library (-lm)

## Building and Running

1. **Compile the project**:
   ```bash
   gcc -o transformer src/*.c examples/main.c -lm -fopenmp
   ```

2. **Run the model**:
   ```bash
   ./transformer
   ```

## Configuration

The model can be configured by modifying the following parameters in the code:

- `MAX_SENTENCE_LENGTH`: Maximum length of input sequences (default: 512)
- `MATRIX_SIZE`: Size of attention matrices (default: 2)
- `EMBEDDING_DIM`: Dimension of word embeddings (default: 2)
- `LEARNING_RATE`: Learning rate for optimization (default: 0.01)

## Training Data

The model expects input data in the format of `test_data.txt`, which should contain text data for training. The data will be automatically tokenized and processed by the model.

## Implementation Details

### Self-Attention Mechanism
The self-attention mechanism computes attention scores between all positions in the input sequence, allowing the model to capture long-range dependencies.

### Positional Encoding
Positional information is added to the embeddings using sine and cosine functions of different frequencies.

### Feed-Forward Network
The feed-forward network consists of two linear transformations with a non-linear activation function in between.

### Backpropagation
The implementation includes gradient computation and weight updates using the mean squared error loss function.

## Notes

- This implementation is a simplified version of the original Transformer model
- The model is designed for educational purposes and may need modifications for production use
- The current implementation uses fixed-size matrices and may need adjustments for different input sizes

## License

This project is open-source and available for educational and research purposes.