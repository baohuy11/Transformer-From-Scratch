#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#include "../include/tokenizer.h"

# define TABLE_SIZE 100000  // SIZE OF THE HASH TABLE

// STRUCTURE FOR THE HASH TABLE NODE
typedef struct word_token_node {
    char *word;                       // Dynamically allocated word string
    unsigned int token_id;            // Unique token ID 
    struct word_token_node *next;     // Pointer to the next node
    float embedding[2];               // 2-element word embedding vector
} word_token_node;

// GLOBAL VARIABLES
word_token_node* hashTable[TABLE_SIZE] = {NULL};
int global_token = 1;


// IMPROVED HASH FUNCTION USING DJB2 ALGORITHM
unsigned int hash(const char *word){
    unsigned long hashValue = 5381;
    int c;

    while(c = *word++){
        hashValue = ((hashValue << 5) + hashValue) + tolower(c); // hash * 33 + c
    }

    return hashValue % TABLE_SIZE;
}

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

    newNode->token_id = global_token;
    global_token++;
    newNode->next = hashTable[index];
    hashTable[index] = newNode;
}

// CHECK IF THE WORD IS PRESENT IN THE HASH TABLE
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

// PRINT THE TOKENS AND THEIR IDS
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

// GET THE TOKEN ID FOR THE GIVEN WORD
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

// GENERATE A RANDOM FLOAT BETWEEN -50.0 AND 50.0
float generate_random(void){
    float random_frac = (float)rand() / (float)RAND_MAX; // Generate a random float between 0 and 1

    float min = -50.0f;
    float max = 50.0f;

    return min + random_frac * (max - min); // Scale to the desired range
}

// GENERATE A WORD EMBEDDING FOR THE GIVEN TOKEN ID
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

// EXTRACT UNIQUE WORDS FROM AN ARRAY OF SENTENCES AND GENERATE EMBEDDINGS
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

// GET THE EMBEDDING FOR THE GIVEN TOKEN ID
void getEmbeddingByTokenId(unsigned int token_id, double expected_embedding[2]){
    // Initialize with default values
    expected_embedding[0] = 0.0;
    expected_embedding[1] = 0.0;

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
    }
    // Token ID not found - embeddings remain at default values
}