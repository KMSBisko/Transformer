#include <bits/stdc++.h>
#include "PositionalEncoding.h"
#include "TransformerEncoder.h"
#include "Matrix.h"
#include "TransformerBlock.h"
#include "FeedForward.h"
#include "MultiHeadAttention.h"
#include "ScaledDotProductAttention.h"
#include "TextPrediction.h"
#include "Vocabulary.h"
#include "Tokenizer.h"
#include "Vector.h"
using namespace std; 

std::vector<std::vector<int>> createSequences(const std::vector<std::string> &tokens, int seqLength, const std::unordered_map<std::string, int> &vocab) {
    std::vector<std::vector<int>> sequences;
    for (size_t i = 0; i <= tokens.size() - seqLength; ++i) {
        std::vector<int> seq;
        for (size_t j = 0; j < seqLength; ++j) {
            auto it = vocab.find(tokens[i + j]);
            if (it != vocab.end()) {
                seq.push_back(it->second);
            } else {
                // Handle unknown token
                seq.push_back(vocab.at("<UNK>")); // Assuming <UNK> is in the vocabulary for unknown tokens
            }
        }
        sequences.push_back(seq);
    }
    return sequences;
}

std::vector<double> softmax(const std::vector<double>& logits) {
    std::vector<double> exp_values(logits.size());
    double max_logit = *max_element(logits.begin(), logits.end());
    double sum_exp = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        exp_values[i] = exp(logits[i] - max_logit);
        sum_exp += exp_values[i];
    }
    for (size_t i = 0; i < exp_values.size(); ++i) {
        exp_values[i] /= sum_exp;
    }
    return exp_values;
}

int sampleFromDistribution(const std::vector<double>& distribution) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(distribution.begin(), distribution.end());
    return dist(gen);
}

int findNearestToken(double value, const std::unordered_map<int, std::string>& reverseVocab, double threshold) {
    int nearestToken = -1;
    double minDistance = std::numeric_limits<double>::max();
    for (const auto& pair : reverseVocab) {
        double distance = std::abs(value - pair.first);
        if (distance < minDistance) {
            minDistance = distance;
            nearestToken = pair.first;
        }
    }
    if (minDistance > threshold) {
        return -1; // Indicate that the nearest token is too far
    }
    return nearestToken;
}

int main() {
    // Redirect output to a file
    freopen("output.txt", "w", stdout);

    // Example text data
    std::string text = "The quick brown fox jumps over the lazy dog";
    auto tokens = tokenize(text);
    auto vocab = createVocabulary(tokens);
    auto reverseVocab = createReverseVocabulary(vocab);
    int seqLength = 5;
    auto sequences = createSequences(tokens, seqLength, vocab);

    // Initialize the transformer model
    int modelDim = 64, numHeads = 12, ffDim = 256, numLayers = 6;
    TransformerEncoder encoder(numLayers, modelDim, numHeads, ffDim);

    // Example input for the transformer
    Matrix input(seqLength, modelDim, 1.0); // Example input
    Matrix posEncoding = PositionalEncoding::compute(seqLength, modelDim);
    input = input + posEncoding;

    // Forward pass through the transformer
    Matrix output = encoder.forward(input);

    // Print the transformer encoder output
    std::cout << "Transformer Encoder Output:\n";
    for (int i = 0; i < output.rows(); ++i) {
        for (int j = 0; j < output.cols(); ++j)
            std::cout << output(i, j) << " ";
        std::cout << "\n";
    }

    // Generate text using the trained model
    std::vector<int> seed;
    for (const auto &token : tokens) {
        auto it = vocab.find(token);
        if (it != vocab.end()) {
            seed.push_back(it->second);
        } else {
            // Handle unknown token
            seed.push_back(vocab.at("<UNK>")); // Assuming <UNK> is in the vocabulary for unknown tokens
        }
    }
    int length = 10;

    std::vector<int> inputvector = seed;
    std::string result;
    double threshold = 2;

    for (int i = 0; i < length; ++i) {
        Matrix inputMatrix(inputvector.size(), 1);
        for (size_t j = 0; j < inputvector.size(); ++j) {
            inputMatrix(j, 0) = inputvector[j];
        }
        Matrix outputMatrix = encoder.forward(inputMatrix);

        // Apply softmax to the last row of the output matrix
        std::vector<double> logits(outputMatrix.cols());
        for (int j = 0; j < outputMatrix.cols(); ++j) {
            logits[j] = outputMatrix(outputMatrix.rows() - 1, j);
        }
        std::vector<double> probabilities = softmax(logits);

        // Sample the next token from the probability distribution
        int nextToken = sampleFromDistribution(probabilities);
        inputvector.push_back(nextToken);

        // Find the nearest token in reverseVocab
        int nearestToken = findNearestToken(nextToken, reverseVocab, threshold);
        if (nearestToken != -1) {
            result += reverseVocab.at(nearestToken) + " ";
        } else {
            // Handle unknown token
            result += "<UNK> ";
        }
    }

    std::string generatedText = result;
    std::cout << text << " " << generatedText << std::endl;
}