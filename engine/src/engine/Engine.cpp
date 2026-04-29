#include "Engine.hpp"
#include "Eval.hpp"

Engine::Engine(const std::string& modelPath, std::string& fen)
{
    try {
        this->_fen = fen;
        this->_model = torch::jit::load(modelPath);
        this->_model.eval();
        std::cout << "Model loaded successfully." << std::endl;
    }
    catch (const c10::Error& e) {
        throw EngineException("Failed to load model.");
    }
}

float Engine::evaluate_board()
{
    try {
        float score = Eval::evaluate_board(this->_model, this->_fen);
        std::cout << "FEN: " << this->_fen << std::endl;
        std::cout << "Evaluation score: " << score << std::endl;
        return score;
    } catch (const EvalException& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
}
