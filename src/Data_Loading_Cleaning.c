#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "../include/Data_Loading_Cleaning.h"

// READ THE ENTIRE CONTENTS OF A FILE INTO A DYNAMICALLY ALLOCATED STRING
char* readFileToString(const char *filename){
    // Open the file in read mode
    FILE *file = fopen(filename, "r");
    if(!file){
        perror("Error opening file");
        return NULL;
    }

    // Get the file size by seeking to the end
    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    rewind(file);  // Reset file pointer to beginning

    // Allocate memory for the file content plus null terminator
    char *fileContent = (char *)malloc(fileSize + 1);
    if (fileContent == NULL) {
        fclose(file);
        return NULL;
    }

    // Read the entire file into memory
    size_t bytesRead = fread(fileContent, 1, fileSize, file);
    if (bytesRead != fileSize) {
        free(fileContent);
        fclose(file);
        return NULL;
    }

    fileContent[fileSize] = '\0'; // Null-terminate the string
    fclose(file);

    return fileContent;
}

// SPLIT THE TEXT INTO SENTENCES BASED ON PUNCTUATION MARKS
char** SplitSentences(char *raw_text){
    const char *delimiters = ".!?";  // Sentence-ending punctuation marks

    // Allocate memory for up to 100 sentences
    char **sentence = malloc(100 * sizeof(char*));
    if (sentence == NULL) {
        return NULL;
    }

    // Split the text into sentences using strtok
    char *token = strtok(raw_text, delimiters);
    int i = 0;

    while(token != NULL && i < 99){  // Limit to 99 sentences to leave room for NULL terminator
        // Allocate memory for the current sentence
        sentence[i] = malloc((strlen(token) + 1) * sizeof(char));
        if (sentence[i] == NULL) {
            // Clean up previously allocated memory
            for (int j = 0; j < i; j++) {
                free(sentence[j]);
            }
            free(sentence);
            return NULL;
        }
        
        strcpy(sentence[i], token);
        i++;
        token = strtok(NULL, delimiters);
    }
    
    sentence[i] = NULL; // Null-terminate the array of sentences
    return sentence;
}

// CLEAN THE TEXT BY CONVERTING ALL CHARACTERS TO LOWERCASE AND REMOVING ALL NON-ALPHANUMERIC CHARACTERS
char* Cleaned_Text(char *raw_text){
    int len = strlen(raw_text);
    char *cleaned_text = malloc((len + 1) * sizeof(char));
    if(cleaned_text == NULL){
        fprintf(stderr, "Memory allocation failed for cleaned_text.\n");
        return NULL;
    }

    int j = 0;  // Index for cleaned_text
    int last_was_space = 0;  // Flag to track consecutive spaces

    for(int i = 0; i < len; i++){
        char c = tolower(raw_text[i]);  // Convert to lowercase
        
        if(isalnum(c)){  // Keep alphanumeric characters
            cleaned_text[j++] = c;
            last_was_space = 0;
        }
        else if(c == ' ' && !last_was_space){  // Keep single spaces
            cleaned_text[j++] = ' ';
            last_was_space = 1;
        }
        // Skip all other characters
    }
    
    cleaned_text[j] = '\0'; // Null-terminate the cleaned text
    return cleaned_text;
}