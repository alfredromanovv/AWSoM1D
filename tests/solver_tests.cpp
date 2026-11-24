#include "../functions.h"
#include <cassert>

// A uniform gas at rest with equal wave populations has no reflection.
// The initial momentum RHS includes gravity and the difference between the
// area secant in the flux and the analytic derivative in the source.
int main()
{
    g_grid.create(10, 2, 1, 2, 1);
    const State initial = Primitive{1, 0, 1, 0.01, 0.01}.to_conservative();
    g_grid.fill(initial);
    apply_boundary_conditions(g_grid); // Match ghost cells restored by load_state.
    const double start = END_TIME - 1e-7;
    const double dt = END_TIME - start;
    g_time = start;
    save_state("solver_test.restart");
    calculate();
    assert(g_step == 1 && g_time == END_TIME);
    const int i = g_grid.begin() + 4;
    const Cell& c = g_grid.cell(i);
    const double area_secant = (g_grid.face(i + 1).area - g_grid.face(i).area) / c.width;
    const double pressure = initial.pressure() + 0.5 * (initial.w_plus + initial.w_minus);
    const double momentum_rhs = pressure * (c.area_gradient - area_secant) / c.area
        - initial.rho / (c.r * c.r);
    const double loss = D / sqrt(c.area * initial.rho) * 0.001;
    assert(abs((c.state.momentum - initial.momentum) / dt - momentum_rhs) < 1e-6);
    assert(abs((c.state.w_plus - initial.w_plus) / dt + loss) < 1e-6);
    assert(abs((c.state.w_minus - initial.w_minus) / dt + loss) < 1e-6);
    assert(abs((c.state.energy - initial.energy) / dt) < 1e-6);
    assert(abs((c.state.rho - initial.rho) / dt) < 1e-6);
    const Grid expected = g_grid;
    load_state("solver_test.restart");
    assert(g_step == 0 && g_time == start);
    calculate();
    for (int j = 0; j < g_grid.size(); ++j) {
        const auto& a = g_grid.cell(j).state;
        const auto& b = expected.cell(j).state;
        assert(a.rho == b.rho && a.momentum == b.momentum && a.energy == b.energy);
        assert(a.w_plus == b.w_plus && a.w_minus == b.w_minus);
    }
    cout << "Solver and restart checks passed.\n";
}
