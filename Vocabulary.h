#ifndef VOCABULARY_H
#define VOCABULARY_H

#include <unordered_map>
#include <vector>
#include <string>

std::unordered_map<std::string, int> createVocabulary(const std::vector<std::string> &tokens);
std::unordered_map<int, std::string> createReverseVocabulary(const std::unordered_map<std::string, int> &vocab);

#endif // VOCABULARY_H