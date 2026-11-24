#include "../functions.h"
#include <cassert>
#include <limits>

static bool close(double a, double b)
{
    return abs(a - b) <= 1e-10 * max(1.0, max(abs(a), abs(b)));
}

template<class F> static void rejects(F action)
{
    bool rejected = false;
    try { action(); } catch (const exception&) { rejected = true; }
    assert(rejected);
}

static void check_zero(const State& state)
{
    assert(state.rho == 0 && state.momentum == 0 && state.energy == 0);
    assert(state.w_plus == 0 && state.w_minus == 0);
}

int main()
{
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(1);
#endif
    Grid grid;
    rejects([&] { set_initial_conditions(grid); });
    grid.create();
    for (int i = 0; i < grid.size(); ++i)
        check_zero(grid.cell(i).state);
    rejects([&] { apply_boundary_conditions(grid); });
    set_initial_conditions(grid);
    // Initial conditions do not implicitly apply BCs.
    for (int i = 0; i < grid.begin(); ++i) {
        check_zero(grid.cell(i).state);
        check_zero(grid.cell(grid.end() + i).state);
    }
    const State first_before_bc = grid.cell(grid.begin()).state;
    apply_boundary_conditions(grid);
    assert(grid.cell(grid.begin()).state.energy == first_before_bc.energy);
    assert(grid.size() == NX + 2 * NG);
    assert(grid.face(grid.begin()).r == R_MIN);
    assert(grid.face(grid.end()).r == R_MAX);
    double volume = 0.0;
    for (int i = grid.begin(); i < grid.end(); ++i) {
        const Cell& c = grid.cell(i);
        assert(c.state.valid());
        volume += c.volume;
        if (i + 1 < grid.end())
            assert(close(grid.cell(i + 1).width / c.width, GRID_RATIO));
        const double x = c.r - 1.0;
        assert(close(c.state.velocity(), U_BASE + U_AMPLITUDE * (1 - exp(-x / U_SCALE))));
        assert(close(c.state.temperature(), 1 + T_AMPLITUDE * (1 - exp(-x / T_SCALE)) * pow(c.r, T_POWER)));
    }
    const double radial_volume = (pow(R_MAX, 3) - pow(R_MIN, 3)) / 3;
    if constexpr (SUPER_RADIAL_EXPANSION) {
        assert(grid.face(grid.begin()).area == 1.0);
        assert(volume >= radial_volume / (R_MIN * R_MIN));
    } else {
        assert(close(volume, radial_volume));
    }
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    Grid parallel;
    parallel.create();
    set_initial_conditions(parallel);
    apply_boundary_conditions(parallel);
    for (int i = 0; i < grid.size(); ++i) {
        const State& a = grid.cell(i).state;
        const State& b = parallel.cell(i).state;
        assert(grid.cell(i).r == parallel.cell(i).r);
        assert(a.rho == b.rho && a.momentum == b.momentum && a.energy == b.energy);
        assert(a.w_plus == b.w_plus && a.w_minus == b.w_minus);
    }
    for (int i = 0; i < grid.begin(); ++i) {
        const State& boundary = grid.cell(i).state;
        const State& interior = grid.cell(grid.begin()).state;
        assert(boundary.rho == 1.0);
        assert(close(boundary.temperature(), 1.0));
        assert(boundary.w_plus == OMEGA_0);
        assert(close(boundary.velocity(), interior.velocity()));
        assert(boundary.w_minus == interior.w_minus);
    }
    assert(grid.cell(grid.size() - 1).state.energy == grid.cell(grid.end() - 1).state.energy);
    const State state = State::from_primitive(2, -0.5, 3, 0.1, 0.2);
    assert(close(state.velocity(), -0.5) && close(state.temperature(), 3));
    const Primitive primitive = state.to_primitive();
    assert(close(primitive.rho, 2) && close(primitive.u, -0.5));
    assert(close(primitive.p, THETA * 2 * 3));
    assert(close(primitive.w_plus, 0.1) && close(primitive.w_minus, 0.2));
    const State roundtrip = primitive.to_conservative();
    assert(close(roundtrip.rho, state.rho) && close(roundtrip.momentum, state.momentum));
    assert(close(roundtrip.energy, state.energy));
    assert(close(roundtrip.w_plus, state.w_plus) && close(roundtrip.w_minus, state.w_minus));
    rejects([] { State{}.to_primitive(); });
    rejects([] { Primitive{}.to_conservative(); });
    rejects([] { Primitive{1, 0, 1, -1, 0}.to_conservative(); });
    grid.set_state(grid.begin(), state);
    grid.fill(grid.cell(grid.begin()).state);
    for (int i = 0; i < grid.size(); ++i)
        assert(grid.cell(i).state.energy == state.energy);
    apply_boundary_conditions(grid);
    for (int i = 0; i < grid.begin(); ++i) {
        const State& boundary = grid.cell(i).state;
        assert(boundary.rho == 1.0 && close(boundary.temperature(), 1.0));
        assert(boundary.w_plus == OMEGA_0);
        assert(close(boundary.velocity(), -0.5) && boundary.w_minus == 0.2);
        assert(close(boundary.energy,
            0.5 * 0.5 * 0.5 + THETA / (GAMMA - 1.0) + OMEGA_0 + 0.2));
        assert(grid.cell(grid.end() + i).state.energy == state.energy);
    }
    assert(grid.cell(grid.begin()).state.energy == state.energy);
    rejects([&] { grid.set_state(-1, state); });
    rejects([&] { grid.fill(State{}); });
    rejects([&] { State::from_primitive(1, 0, -1, 0, 0); });
    rejects([&] { grid.create(0); });
    rejects([&] { grid.create(numeric_limits<int>::max(), 0); });
    rejects([&] { grid.create(10, 2, 1, 30, 1); }); // Ghosts cross r = 0.
    apply_boundary_conditions(grid);
    assert(grid.size() == NX + 2 * NG); // Failed create preserves old grid.
    for (double ratio : {1.0, 1.0 + 1e-12, 0.99}) {
        grid.create(10, 2, 1, 2, ratio);
        for (int i = 0; i + 1 < grid.size(); ++i)
            assert(close(grid.cell(i + 1).width / grid.cell(i).width, ratio));
    }
    grid.create(1, 0, 1, 2, 1);
    set_initial_conditions(grid);
    apply_boundary_conditions(grid);
    assert(grid.size() == 1 && grid.cell(0).state.valid());
    const double volume_before_clear = grid.cell(0).volume;
    grid.clear_states();
    check_zero(grid.cell(0).state);
    assert(grid.cell(0).volume == volume_before_clear);
    cout << "Grid checks passed.\n";
}
