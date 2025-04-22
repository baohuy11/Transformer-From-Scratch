#include <stdio.h>
#include <stdlib.h>
#include "include/Data_Loading_Cleaning.h"

int main() {
    // Test reading file
    char* content = readFileToString("test_data.txt");
    if (content == NULL) {
        printf("Error reading file\n");
        return 1;
    }
    printf("File content read successfully\n");

    // Test sentence splitting
    char** sentences = SplitSentences(content);
    if (sentences == NULL) {
        printf("Error splitting sentences\n");
        free(content);
        return 1;
    }
    printf("Sentences split successfully\n");

    // Print first few sentences
    printf("\nFirst few sentences:\n");
    for (int i = 0; sentences[i] != NULL && i < 3; i++) {
        printf("Sentence %d: %s\n", i + 1, sentences[i]);
    }

    // Test text cleaning
    char* cleaned = Cleaned_Text(content);
    if (cleaned == NULL) {
        printf("Error cleaning text\n");
        free(content);
        for (int i = 0; sentences[i] != NULL; i++) {
            free(sentences[i]);
        }
        free(sentences);
        return 1;
    }
    printf("\nCleaned text:\n%s\n", cleaned);

    // Clean up memory
    free(content);
    free(cleaned);
    for (int i = 0; sentences[i] != NULL; i++) {
        free(sentences[i]);
    }
    free(sentences);

    return 0;
} 