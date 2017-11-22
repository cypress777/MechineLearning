//
// Created by cypress on 17/11/2017.
//

#ifndef ML_MAIN_HPP
#define ML_MAIN_HPP

#include <vector>
using namespace std;

enum Colour {
    green,
    black,
    white,
};

enum  Sterm {
    straight = 0,
    curl,
    wave,
};

enum Sound {
    dull = 0,
    heavy,
    crispy,
};

enum Texture {
    clear = 0,
    unclear,
    blur,
};

enum Belly {
    flat = 0,
    slight,
    dent,

};

enum Feel {
    hard = 0,
    sticky,
};

struct MelonInfo {
    Colour clr;
    Sterm stm;
    Sound snd;
    Texture txtr;
    Belly blly;
    Feel feel;
    float density;
    float sugar;
    bool quality;
};


#endif //ML_MAIN_HPP
