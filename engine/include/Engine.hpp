#ifndef ENGINE_HPP_
    #define ENGINE_HPP_

#include <torch/script.h>

class Engine {
    torch::jit::script::Module _model;
    std::string _fen;
    bool _enginePlayer;

    public:
        Engine(const std::string& modelPath, std::string& fen, bool enginePlayer);
        ~Engine() = default;

        float evaluate_board();
};

class EngineException : public std::exception {
    protected:
        std::string _msg;
    public:
        EngineException(std::string msg) : _msg(msg) {}

        virtual const char* what() const noexcept { return this->_msg.c_str(); }
};

#endif /* ENGINE_HPP_ */

