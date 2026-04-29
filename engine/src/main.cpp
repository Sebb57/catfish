#include <torch/script.h>
#include <iostream>
#include "Engine.hpp"
#include "Parser.hpp"
#include "constants.hpp"

int main(int const ac, char const * const av[])
{
    try {
        Parser parser;
        if (!parser.parse(ac, av))
            return parser.getRetVal();

        Engine engine(parser.getModelPath(), parser.getFen(), parser.getEnginePlayer());
        engine.predictMove();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return ERROR;
    }
}
