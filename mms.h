#pragma once
#include "grid.h"
#include <array>

namespace mms {
std::array<double, 5> source(double r, double time);
void fill_ghosts(Grid& grid, double time);
void run_study();
}
