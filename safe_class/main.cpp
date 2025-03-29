#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <type_traits>

#define RSI_ARITH_BEHAVIOUR_MUL rsi::rsi_arith_behaviour::STANDARD_BUT_NOT_SAME_SIZE
#define RSI_ARITH_ACCUM_BEHAVIOUR_MUL rsi::rsi_arith_accum_behaviour::NONE
#define RSI_ALLOW_CONVERSION 0

#include "MySafeInt.hpp"

using namespace rsi;

int main(){
    for(int i = 0; i < 20; i+=2){
        std::cout<<i<<std::endl;
    }
}