#include "Vocabulary.h"

std::unordered_map<std::string, int> createVocabulary(const std::vector<std::string> &tokens) {
    std::unordered_map<std::string, int> vocab;
    int index = 0;
    for (const auto &token : tokens) {
        if (vocab.find(token) == vocab.end()) {
            vocab[token] = index++;
        }
    }
    return vocab;
}

std::unordered_map<int, std::string> createReverseVocabulary(const std::unordered_map<std::string, int> &vocab) {
    std::unordered_map<int, std::string> reverseVocab;
    for (const auto &pair : vocab) {
        reverseVocab[pair.second] = pair.first;
    }
    return reverseVocab;
}