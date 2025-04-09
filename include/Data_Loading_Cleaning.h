#ifndef DATA_LOADING_CLEANING_H
#define DATA_LOADING_CLEANING_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
/* File I/O functions */

/**
 * @brief Reads the entire content of a file into a dynamically allocated string.
 *
 * @param filename The name of the file to read.
 * @return A pointer to a C-string containing the file's content.
 *         The caller is responsible for freeing the returned string.
 *         Returns NULL on error.
 */
char* readFileToString(const char *filename);

/* Text processing functions */

/**
 * @brief Splits raw text into an array of sentences.
 *
 * This function tokenizes the input text into sentences. The returned
 * array is NULL-terminated (i.e. the final pointer in the array will be NULL).
 *
 * @param raw_text The input text to be split.
 * @return An array of C-strings (each sentence), or NULL on error.
 */
char** SplitSentences(char *raw_text);

/**
 * @brief Cleans the raw text by removing unwanted characters.
 *
 * This function performs basic cleaning on the input text such as
 * removing extra whitespace or control characters, and returns a new
 * dynamically allocated string.
 *
 * @param raw_text The original text.
 * @return A pointer to a cleaned version of the input text.
 */
char* Cleaned_Text(char *raw_text);

#endif /* DATA_LOADING_CLEANING_H */