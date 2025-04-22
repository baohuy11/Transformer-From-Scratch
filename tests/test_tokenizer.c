#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/tokenizer.h"

void test_hash_function() {
    printf("Testing hash function...\n");
    const char* test_words[] = {"hello", "world", "transformer", "test", "hello"};
    int num_words = sizeof(test_words) / sizeof(test_words[0]);
    
    for (int i = 0; i < num_words; i++) {
        unsigned int hash_value = hash(test_words[i]);
        printf("Word: '%s' -> Hash: %u\n", test_words[i], hash_value);
    }
    printf("\n");
}

void test_word_insertion() {
    printf("Testing word insertion...\n");
    const char* test_words[] = {"hello", "world", "transformer", "test", "hello"};
    int num_words = sizeof(test_words) / sizeof(test_words[0]);
    
    for (int i = 0; i < num_words; i++) {
        if (!isWordPresent(test_words[i])) {
            insertWord(test_words[i]);
            printf("Inserted: '%s' with token ID: %u\n", test_words[i], getTokenId(test_words[i]));
        } else {
            printf("Word '%s' already exists with token ID: %u\n", test_words[i], getTokenId(test_words[i]));
        }
    }
    printf("\n");
}

void test_embedding_generation() {
    printf("Testing embedding generation...\n");
    const char* test_words[] = {"hello", "world", "transformer"};
    int num_words = sizeof(test_words) / sizeof(test_words[0]);
    
    for (int i = 0; i < num_words; i++) {
        unsigned int token_id = getTokenId(test_words[i]);
        if (token_id != 0) {
            float** embedding = Word_Embedding_Generation(token_id);
            if (embedding != NULL) {
                printf("Word: '%s' (Token ID: %u) -> Embedding: [%.4f, %.4f]\n",
                       test_words[i], token_id, embedding[0][0], embedding[1][0]);
                free(embedding[0]);
                free(embedding[1]);
                free(embedding);
            }
        }
    }
    printf("\n");
}

void test_sentence_processing() {
    printf("Testing sentence processing...\n");
    char* test_sentences[] = {
        "Hello world! This is a test.",
        "The transformer model is amazing.",
        "Natural language processing is fun.",
        NULL
    };
    
    extractUniqueWords(test_sentences);
    printf("Unique words extracted from sentences:\n");
    Print_Tokens_And_Ids();
    printf("\n");
}

int main() {
    printf("Starting tokenizer tests...\n\n");
    
    // Initialize random seed for embedding generation
    srand(42);
    
    test_hash_function();
    test_word_insertion();
    test_embedding_generation();
    test_sentence_processing();
    
    printf("All tokenizer tests completed.\n");
    return 0;
} 