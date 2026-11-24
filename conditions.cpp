#include "functions.h"

// Initialize physical cells. Ghosts are set separately by the boundary conditions.
void set_initial_conditions(Grid& grid)
{
    if (grid.size() == 0 || grid.face(grid.begin()).r < 1.0)
        throw logic_error("Initial profiles require a grid with r >= 1.");
    const int first = grid.begin();
    const State initial = State::from_primitive(RHO_INITIAL, U_INITIAL, T_INITIAL, OMEGA_0, W_MINUS_INITIAL);
    for (int i = first; i < grid.end(); ++i)
        grid.set_state(i, initial);
}

State photospheric_boundary_state(const State& interior)
{
    // Fixed photospheric density, temperature and outgoing wave energy.
    // Velocity and incoming wave energy follow the adjacent physical cell.
    return State::from_primitive(1.0, interior.to_primitive().u, 1.0, OMEGA_0, interior.w_minus);
}

void apply_boundary_conditions(Grid& grid)
{
    if (grid.size() == 0)
        throw logic_error("Create the grid before applying boundary conditions.");
    const int ghosts = grid.begin();
    if (ghosts == 0)
        return;
    const State left = grid.cell(grid.begin()).state;
    const State right = grid.cell(grid.end() - 1).state;
    if (!left.valid() || !right.valid())
        throw logic_error("Set physical states before applying boundary conditions.");
    const State photosphere = photospheric_boundary_state(left);
    // Apply the photospheric data and the extrapolated outer state at every RK stage.
    // With the default two layers, a separate OpenMP team would cost more than copying.
    for (int j = 0; j < ghosts; ++j) {
        grid.set_state(j, photosphere);
        grid.set_state(grid.end() + j, right);
    }
}
