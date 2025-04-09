#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* Define hash table size */
#define TABLE_SIZE 5000

/**
 * @brief Structure representing a word token in the hash table.
 *
 * This structure stores a dynamically allocated word string,
 * a unique token ID, and a pointer for chaining nodes
 * (to handle hash collisions).
 */
typedef struct word_token_node {
    char *word;                         /**< Dynamically allocated string for the word */
    unsigned int token_id;              /**< Unique token ID assigned to this word */
    struct word_token_node *next;       /**< Pointer to the next node in case of hash collisions */
} word_token_node;

/* Global variables */

/** Global hash table to store unique word tokens. */
extern word_token_node* hashTable[TABLE_SIZE];

/** Global token counter used to generate unique token IDs. */
extern int global_token;

/* Hash table functions */

/**
 * @brief Computes a hash value for a given word using the DJB2 algorithm.
 *
 * @param word The input word string.
 * @return An unsigned int hash value.
 */
unsigned int hash(const char* word);

/**
 * @brief Inserts a word into the global hash table.
 *
 * If the word is not already present, this function stores the word
 * along with a unique token ID.
 *
 * @param word The word to insert.
 */
void insertWord(const char* word);

/**
 * @brief Checks if a word is already present in the global hash table.
 *
 * @param word The word to check.
 * @return 1 if the word is found, 0 otherwise.
 */
int isWordPresent(const char* word);

/**
 * @brief Extracts unique words from an array of sentences.
 *
 * This function processes the given sentences and inserts each unique
 * word into the global hash table.
 *
 * @param sentences A NULL-terminated array of sentence strings.
 */
void extractUniqueWords(char** sentences);

/**
 * @brief Retrieves the token ID for a given word.
 *
 * @param word The word to look up.
 * @return The token ID if present; otherwise, 0.
 */
unsigned int getTokenId(const char* word);

/**
 * @brief Prints all unique words stored in the hash table along with their token IDs.
 */
void Print_Tokens_And_Ids(void);

/* Word Embedding functions */

/**
 * @brief Generates a random float value.
 *
 * This function may be used for initializing word embeddings.
 *
 * @return A random float.
 */
float generate_random(void);

/**
 * @brief Generates word embeddings for a given token ID.
 *
 * Allocates a 2D float array representing the embedding.
 *
 * @param token_id The token ID for which to generate embeddings.
 * @return A pointer to a dynamically allocated 2D float array.
 */
float** Word_Embedding_Generation(int token_id);

/**
 * @brief Retrieves the word embedding for a given token ID.
 *
 * @param token_id The token ID.
 * @return A pointer to a float array representing the embedding.
 */
float* getEmbedding(unsigned int token_id);

/**
 * @brief Retrieves the word embedding for a given token ID into a fixed-size array.
 *
 * The result is stored in the provided expected_embedding array.
 *
 * @param token_id The token ID.
 * @param expected_embedding An array (of size 2) in which the embedding will be stored.
 */
void getEmbeddingByTokenId(unsigned int token_id, double expected_embedding[2]);

#endif /* TOKENIZER_H */
