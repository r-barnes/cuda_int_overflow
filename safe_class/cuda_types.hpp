#pragma once

#include <cstdint>

struct { uint32_t x=0,y=0,z=0; } threadIdx;
struct { uint32_t x=128,y=1,z=1; } threadDim;
struct { uint32_t x=0,y=0,z=0; } blockIdx;
struct { uint32_t x=100,y=1,z=1; } blockDim;
struct { uint32_t x=0,y=0,z=0; } gridIdx;
struct { uint32_t x=1,y=1,z=1; } gridDim;