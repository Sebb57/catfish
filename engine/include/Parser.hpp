#ifndef PARSER_HPP_
    #define PARSER_HPP_

#include <iostream>

class Parser {
    std::string _fen;
    std::string _modelPath;
    bool _enginePlayer;

    int _retVal;
    public:
        Parser() = default;
        ~Parser() = default;

        bool parse(int const ac, char const * const av[]);
        std::string& getFen() noexcept { return this->_fen; }
        std::string& getModelPath() noexcept { return this->_modelPath; }
        bool getEnginePlayer() noexcept { return this->_enginePlayer; }
        int getRetVal() noexcept { return this->_retVal; }
};

class ParserException : public std::exception {
    protected:
        std::string _msg;
    public:
        ParserException(std::string msg) : _msg(msg) {}

        virtual const char* what() const noexcept { return this->_msg.c_str(); }
};

#endif /* PARSER_HPP_ */

