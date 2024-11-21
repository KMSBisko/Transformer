#ifndef TEXT_PREDICTION_H
#define TEXT_PREDICTION_H

#include "TransformerEncoder.h"
#include <unordered_map>
#include <vector>
#include <string>

std::vector<std::vector<int>> createSequences(const std::vector<std::string> &tokens, int seqLength, const std::unordered_map<std::string, int> &vocab);
std::string generateText(const TransformerEncoder &model, const std::vector<int> &seed, int length, const std::unordered_map<int, std::string> &reverseVocab);

#endif // TEXT_PREDICTION_H