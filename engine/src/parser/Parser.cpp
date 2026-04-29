#include "Parser.hpp"
#include "constants.hpp"
#include <libconfig.h++>

bool Parser::parse(int const ac, char const * const av[])
{
    if (ac != 2 || !av || !av[1]) {
        this->_retVal = ERROR;
        return false;
    }

    libconfig::Config cfg;

    try {
        cfg.readFile(av[1]);
        const libconfig::Setting& s = cfg.lookup("settings");
        this->_fen = s["fen"].c_str();
        this->_enginePlayer = std::string(s["enginePlayer"]) == "W";
        this->_modelPath = s["modelPath"].c_str();
    } catch (...) {
        throw ParserException("Failed to read config file");
    }
    this->_retVal = SUCCESS;
    return true;
}
