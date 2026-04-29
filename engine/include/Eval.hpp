#ifndef EVAL_HPP_
    #define EVAL_HPP_

#include <torch/script.h>

class Eval {
    static torch::Tensor fen_to_tensor(const std::string& fen);
    public:
        ~Eval() = default;
        static float evaluate_board(torch::jit::script::Module& model, const std::string& fen);

};

class EvalException : public std::exception {
    protected:
        std::string _msg;
    public:
        EvalException(std::string msg) : _msg(msg) {}

        virtual const char* what() const noexcept { return this->_msg.c_str(); }
};

#endif /* EVAL_HPP_ */

