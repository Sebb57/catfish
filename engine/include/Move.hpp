#ifndef MOVE_HPP_
    #define MOVE_HPP_

struct Move {
    int from;
    int to;
    int promotion;
    bool isCapture;
    bool isCastle;
    bool isEnPassant;
};

#endif /* MOVE_HPP_ */
