#include "Tokenizer.h"
#include <sstream>

std::vector<std::string> tokenize(const std::string &text) {
    std::istringstream stream(text);
    std::vector<std::string> tokens;
    std::string token;
    while (stream >> token) {
        tokens.push_back(token);
    }
    return tokens;
}