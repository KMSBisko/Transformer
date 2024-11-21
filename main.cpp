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
            seq.push_back(vocab.at(tokens[i + j]));
        }
        sequences.push_back(seq);
    }
    return sequences;
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
    int modelDim = 32, numHeads = 6, ffDim = 128, numLayers = 3;
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
        seed.push_back(vocab[token]);
    }
    int length = 10;

    // std::string generatedText = generateText(encoder, seed, 10, reverseVocab);
    std::vector<int> inputvector = seed;
    std::string result;
    for (int i = 0; i < length; ++i) {
        Matrix inputMatrix(inputvector.size(), 1);
        for (size_t j = 0; j < inputvector.size(); ++j) {
            inputMatrix(j, 0) = inputvector[j];
        }
        Matrix outputMatrix = encoder.forward(inputMatrix);
        int nextToken = static_cast<int>(outputMatrix(outputMatrix.rows() - 1, 0)); // Simplified for example
        inputvector.push_back(nextToken);
        result += reverseVocab.at(nextToken) + " ";
    }

    std::string generatedText = result;
    std::cout << "Generated Text: " << generatedText << std::endl;
}