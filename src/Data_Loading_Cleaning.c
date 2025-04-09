#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "../include/Data_Loading_Cleaning.h"

char* readFileToString(const char *filename){
    FILE *file = fopen(filename, "r");

    if(!file){
        perror("Error opening file");
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    rewind(file);

    char *fileContent = (char *)malloc(fileSize + 1);

    fread(fileContent, 1, fileSize, file);

    fileContent[fileSize] = '\0'; // Null-terminate the string

    fclose(file);

    return fileContent;
}

char** SplitSentences(char *raw_text){
    const char *delimiters = ".!?";

    char **sentence = malloc(100 * sizeof(char*));
    char *token = strtok(raw_text, delimiters);

    int i = 0;
    while(token != NULL){
        sentence[i] = malloc((strlen(token) + 1) * sizeof(char));
        strcpy(sentence[i], token);
        i++;
        token = strtok(NULL, delimiters);
    }
    sentence[i] = NULL; // Null-terminate the array
    return sentence;
}

char* Cleaned_Text(char *raw_text){
    int len = strlen(raw_text);
    char *cleaned_text = malloc((len + 1) * sizeof(char));
    if(cleaned_text == NULL){
        fprintf(stderr, "Memory allocation failed for cleaned_text.\n");
        return NULL;
    }
    int j = 0;
    int last_was_space = 0;
    for(int i = 0; i < len; i++){
        char c = tolower(raw_text[i]);
        if(isalnum(c)){
            cleaned_text[j++] = c;
            last_was_space = 0;
        }else if(c == ' ' && !last_was_space){
            cleaned_text[j++] = ' ';
            last_was_space = 1;
        }
    }
    cleaned_text[j] = '\0'; // Null-terminate the cleaned text
    return cleaned_text;
}