#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#include "../include/tokenizer.h"

# define TABLE_SIZE 100000  // SIZE OF THE HASH TABLE

typedef struct word_token_node {
    char *word;                       /**< Dynamically allocated word string */
    unsigned int token_id;            /**< Unique token ID */
    struct word_token_node *next;     /**< Pointer to the next node */
    float embedding[2];               /**< 2-element word embedding vector */
} word_token_node;

word_token_node* hashTable[TABLE_SIZE] = {NULL};

int global_token = 1;

/**
 * @brief Computes a hash value for the given word using the DJB2 algorithm.
 *
 * Converts characters to lowercase to achieve case insensitivity.
 *
 * @param word The word to hash.
 * @return The hash value modulo TABLE_SIZE.
 */
unsigned int hash(const char *word){
    unsigned long hashValue = 5381;
    int c;
    while(c = *word++){
        hashValue = ((hashValue << 5) + hashValue) + tolower(c); // hash * 33 + c
    }
    return hashValue % TABLE_SIZE;
}

/**
 * @brief Inserts a word into the global hash table.
 *
 * Allocates memory for a new node, duplicates the word, assigns a unique token ID,
 * and inserts the node at the beginning of the list for the appropriate table index.
 *
 * @param word The word to insert.
 */
void insertWord(const char* word){
    unsigned int index = hash(word);
    word_token_node* newNode = (word_token_node*)malloc(sizeof(word_token_node));
    if(newNode == NULL){
        fprintf(stderr, "Memory allocation failed for newNode.\n");
        return;
    }
    newNode->word = strdup(word);
    if(newNode->word == NULL){
        fprintf(stderr, "Memory allocation failed for newNode.\n");
        free(newNode);
        return;
    }
    newNode->token_id = global_token++;
    newNode->next = hashTable[index];
    newNode->embedding[0] = 0.0f;
    newNode->embedding[1] = 0.0f;
    hashTable[index] = newNode;
}

/**
 * @brief Checks if a word is present in the hash table.
 *
 * @param word The word to search for.
 * @return 1 if the word is found, 0 otherwise.
 */
int isWordPresent(const char* word){
    unsigned int index = hash(word);
    word_token_node* current = hashTable[index];
    while(current){
        if(strcmp(current->word, word) == 0){
            return 1; // Word found
        }
        current = current->next;
    }
    return 0; // Word not found
}

/**
 * @brief Prints all unique words with their token IDs and embeddings.
 */
void Print_Tokens_And_Ids(void){
    printf("Token / Token ID / Embedding:\n");
    printf("Table Size: %d \n", TABLE_SIZE);
    for(int i = 0; i < TABLE_SIZE; i++){
        word_token_node* current = hashTable[i];
        while(current){
            printf("%s : %d : { %.4f, %.4f }\n", current->word, current->token_id, current->embedding[0], current->embedding[1]);
            current = current->next;
        }
    }
}

/**
 * @brief Retrieves the token ID for the given word.
 *
 * @param word The word to look up.
 * @return The token ID if found, or 0 if not found.
 */
unsigned int getTokenId(const char* word){
    unsigned int index = hash(word);
    word_token_node* current = hashTable[index];
    while(current){
        if(strcmp(current->word, word) == 0){
            return current->token_id; // Return the token ID
        }
        current = current->next;
    }
    return 0; // Word not found
}

/**
 * @brief Generates a random float between -50.0 and 50.0.
 *
 * @return A random float in the specified range.
 */
float generate_random(void){
    float random_frac = (float)rand() / (float)RAND_MAX; // Generate a random float between 0 and 1

    float min = -50.0f;
    float max = 50.0f;

    return min + random_frac * (max - min); // Scale to the desired range
}

/**
 * @brief Generates a word embedding for the given token ID.
 *
 * The embedding is computed using sine and cosine functions and random scaling
 * parameters. Memory is allocated for a 2 x 1 array, which the caller must free.
 *
 * @param token_id The token ID.
 * @return A pointer to a dynamically allocated 2D float array, or NULL on failure.
 */
float** Word_Embedding_Generation(int token_id){
    float** embedding = (float**)malloc(2 * sizeof(float*));
    if(embedding == NULL){
        fprintf(stderr, "Memory allocation failed for embedding array.\n");
        return NULL;
    }
    embedding[0] = (float*)malloc(sizeof(float));
    if(embedding[0] == NULL){
        fprintf(stderr, "Memory allocation failed for embedding[0].\n");
        free(embedding);
        return NULL;
    }
    embedding[1] = (float*)malloc(sizeof(float));
    if(embedding[1] == NULL){
        fprintf(stderr, "Memory allocation failed for embedding[1].\n");
        free(embedding[0]);
        free(embedding);
        return NULL;
    }

    float scale1 = generate_random();
    float scale2 = generate_random();

    embedding[0][0] = scale1 * sin((double)token_id);
    embedding[1][0] = scale2 * cos((double)token_id);

    return embedding;
}

/**
 * @brief Extracts unique words from an array of sentences and generates embeddings.
 *
 * For each sentence, the function tokenizes the sentence based on specified delimiters,
 * then checks if each token is unique. If a token is not present, it is inserted into
 * the hash table and an embedding is generated.
 *
 * @param sentences A NULL-terminated array of sentences.
 */
void extractUniqueWords(char** sentences){
    const char* delimiters = " ,.;!?-";
    for(int i = 0; sentences[i] != NULL; i++){
        char* sentence = strdup(sentences[i]);
        if(sentence == NULL){
            fprintf(stderr, "Memory allocation failed for sentence copy.\n");
            continue;
        }
        char* token = strtok(sentence, delimiters);
        while(token != NULL){
            if(!isWordPresent(token)){
                insertWord(token);
                unsigned int token_id = getTokenId(token);
                float** embedding = Word_Embedding_Generation(token_id);
                if(embedding == NULL){
                    fprintf(stderr, "Embedding generation failed for token \"%s\".\n", token);
                    continue;
                }
                unsigned int index = hash(token);
                word_token_node* current = hashTable[index];
                while(current){
                    if(strcmp(current->word, token) == 0){
                        current->embedding[0] = embedding[0][0];
                        current->embedding[1] = embedding[1][0];
                        break;
                    }
                    current = current->next;
                }
                free(embedding[0]);
                free(embedding[1]);
                free(embedding);

            }
            token = strtok(NULL, delimiters);
        }
        free(sentence);
    }
}

/**
 * @brief Retrieves the word embedding for a given token ID and stores it into an expected output array.
 *
 * If the token is found in the hash table, its embedding (a 2-element array)
 * is assigned to the provided expected_embedding array.
 *
 * @param token_id The token ID to look for.
 * @param expected_embedding Output array of size 2 to store the embedding values.
 */
void getEmbeddingByTokenId(unsigned int token_id, double expected_embedding[2]){
    for(int i = 0; i < TABLE_SIZE; i++){
        word_token_node* current = hashTable[i];
        while(current){
            if(current->token_id == token_id){
                expected_embedding[0] = current->embedding[0];
                expected_embedding[1] = current->embedding[1];
                return;
            }
            current = current->next;
        }
        return; // Token ID not found
    }
}