#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <type_traits>

#include "MySafeInt.hpp"

int main(){
    auto a = MySafeInt(static_cast<int32_t>(23));
    auto b = MySafeInt(static_cast<int32_t>(43));
    a = 2;
    // std::array<int, 8> arr = {1,2,3,4,5,6,7,8};
    // arr[a.value()];
    a * 3;
}