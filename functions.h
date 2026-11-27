#pragma once

#include "include.h"
#include "grid.h"
#include <cstdint>

// Updated by the solver only after a complete accepted step (also on restart).
extern double g_time;
extern uint64_t g_step;

// Handlers only request a stop. The main function performs saving.
void install_signal_handlers();

// Simulation setup.
void load_parameters();
void create_grid();
// Physical cells first, then ghost cells; both operate on the supplied grid.
void set_initial_conditions(Grid& grid);
State photospheric_boundary_state(const State& interior);
void apply_boundary_conditions(Grid& grid);
void load_state(const string& filename);

// Simulation and shutdown.
void calculate(double step_factor = 1.0);
double dissipation_stiffness(const Grid& grid);
void output_results();
// Call at output times; time is dimensionless. Files are written in the working directory.
void save_results(const Grid& grid, double time, uint64_t step);
void save_initial_results(const Grid& grid, double time, uint64_t step);
void save_final_results(const Grid& grid, double time, uint64_t step);
void save_state(const string& filename);
