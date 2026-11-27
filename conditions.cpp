#include "functions.h"

// Initialize physical cells. Ghosts are set separately by the boundary conditions.
void set_initial_conditions(Grid& grid)
{
    if (grid.size() == 0 || grid.face(grid.begin()).r < 1.0)
        throw logic_error("Initial profiles require a grid with r >= 1.");
    for (int i = grid.begin(); i < grid.end(); ++i) {
        const Cell& cell = grid.cell(i);
        const double height = cell.r - 1.0;
        const double u = U_PH_INITIAL - U_INITIAL_RISE * expm1(-height / U_INITIAL_SCALE);
        const double rho = max(U_PH_INITIAL / (cell.area * u)
                               + (1.0 - U_PH_INITIAL / u) * exp(-height / RHO_INITIAL_SCALE),
                               RHO_INITIAL_FLOOR);
        const double temperature = T_PH_INITIAL - T_INITIAL_RISE * expm1(-height / T_INITIAL_SCALE)
                                   * pow(cell.r, T_INITIAL_RADIAL_EXPONENT);
        const double w_plus = OMEGA_0 * pow(cell.area, W_PLUS_INITIAL_AREA_EXPONENT);
        const double w_minus = W_MINUS_INITIAL_RATIO * w_plus * (-expm1(-height / W_MINUS_INITIAL_SCALE));
        grid.set_state(i, State::from_primitive(rho, u, temperature, w_plus, w_minus));
    }
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
