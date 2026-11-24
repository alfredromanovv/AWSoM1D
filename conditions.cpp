#include "functions.h"

// Initial profiles are defined only for r >= 1. Ghosts are set separately by BCs.
void set_initial_conditions(Grid& grid)
{
    if (grid.size() == 0 || grid.face(grid.begin()).r < 1.0)
        throw logic_error("Initial profiles require a grid with r >= 1.");
    const int first = grid.begin();
    const int count = grid.end() - first;
    vector<State> states(count);
    int invalid = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(|:invalid)
#endif
    for (int j = 0; j < count; ++j) {
        const Cell& c = grid.cell(first + j);
        const double x = c.r - 1.0;
        const double u = U_BASE + U_AMPLITUDE * (-expm1(-x / U_SCALE));
        const double rho = max(U_BASE / (c.area * u) +
            (1.0 - U_BASE / u) * exp(-x / RHO_SCALE), INITIAL_RHO_MIN);
        const double wp = OMEGA_0 * pow(c.area, W_AREA_POWER);
        const double wm = W_MINUS_FRACTION * wp * (-expm1(-x / W_MINUS_SCALE));
        try {
            states[j] = Primitive{rho, u, THETA * rho * T_INITIAL, wp, wm}.to_conservative();
        } catch (const invalid_argument&) {
            // An exception must not escape an OpenMP worker.
            invalid = 1;
        }
    }
    if (invalid)
        throw runtime_error("Initial profiles produced invalid states.");
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < count; ++j)
        grid.set_state(first + j, states[j]);
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
