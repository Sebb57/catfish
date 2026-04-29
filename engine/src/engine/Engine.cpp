#include "Engine.hpp"
#include "Eval.hpp"
#include "constants.hpp"
#include <cfloat>

Engine::Engine(const std::string& modelPath, std::string& fen, bool enginePlayer)
{
    try {
        this->_enginePlayer = enginePlayer;
        this->_fen = fen;
        this->_model = torch::jit::load(modelPath);
        this->_model.eval();
    }
    catch (const c10::Error& e) {
        throw EngineException("Failed to load model.");
    }
}

float Engine::evaluateBoard()
{
    try {
        float score = Eval::evaluate_board(this->_model, this->_fen);
        std::cout << "FEN: " << this->_fen << std::endl;
        std::cout << "Evaluation score: " << score << std::endl;
        return score;
    } catch (const EvalException& e) {
        std::cerr << e.what() << std::endl;
        return ERROR;
    }
}

void Engine::predictMove()
{
    Move bestMove;
    double bestScore = -DBL_MAX;
    
    for (Move move : this->_moves) { //optimize later
        // make_move(move)
        double score = -negamax(depth - 1, -inf, +inf);
        // undo_move(move)
        //
        if (score > bestScore) {
            bestScore = score;
            bestMove = move;
        }
    }
    std::cout << "Best move: " << bestMove << std::endl;
}
