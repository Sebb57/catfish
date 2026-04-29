#include <torch/script.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

torch::Tensor fen_to_tensor(const std::string& fen)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor board = torch::zeros({1, 13, 8, 8}, options);
    std::string placement = fen.substr(0, fen.find(' '));
    int rank = 0;
    int file = 0;

    for (char c : placement) {
        if (c == '/') {
            rank++;
            file = 0;
        } else if (isdigit(c)) {
            file += (c - '0');
        } else {
            int channel = -1;
            switch (c) {
                case 'p': channel = 1; break; 
                case 'r': channel = 2; break;
                case 'n': channel = 3; break;
                case 'b': channel = 4; break; 
                case 'q': channel = 5; break; 
                case 'k': channel = 6; break;

                case 'P': channel = 7; break; 
                case 'R': channel = 8; break;
                case 'N': channel = 9; break;
                case 'B': channel = 10; break; 
                case 'Q': channel = 11; break; 
                case 'K': channel = 12; break;

            }
            if (channel != -1 && rank < 8 && file < 8) {
                board[0][channel][rank][file] = 1.0f;
            }
            file++;
        }
    }
    board[0][12].fill_(1.0f); 
    return board;
}

float evaluate_board(torch::jit::script::Module& model, const std::string& fen)
{
    torch::Tensor board = fen_to_tensor(fen); 
    std::vector<std::string> tokens;
    std::string s;
    std::stringstream ss(fen);

    while (ss >> s) 
        tokens.push_back(s);
    if (tokens.size() < 6)
        throw std::runtime_error("Invalid FEN string provided.");

    long active_val = (tokens[1] == "w") ? 1 : 0;
    torch::Tensor active_player = torch::tensor({active_val}, torch::kLong);

    float hm_val = std::stof(tokens[5]);
    torch::Tensor halfmove = torch::tensor({hm_val}, torch::kFloat32);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(board);
    inputs.push_back(active_player);
    inputs.push_back(halfmove);

    at::Tensor output = model.forward(inputs).toTensor();
    return output.item<float>() * 100.0f;
}

int main() {
    const std::string model_path = "../model/johnBordello.pt";
    torch::jit::script::Module module;

    try {
        module = torch::jit::load(model_path);
        module.eval();
        std::cout << "Model loaded successfully." << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return -1;
    }

    std::string fen = "rnbqkb1r/pppn1ppp/4p3/3pP3/3P4/8/PPPN1PPP/R1BQKBNR w KQkq - 1 5";
    
    try {
        float score = evaluate_board(module, fen);
        std::cout << "FEN: " << fen << "\n";
        std::cout << "Evaluation score: " << score << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Inference error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
