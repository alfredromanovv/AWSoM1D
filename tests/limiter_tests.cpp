// Includes solver internals; link with simulation sources except solver.cpp and main.cpp.
#include "../solver.cpp"
#include <cassert>

int main()
{
    for (auto limiter : {SlopeLimiter::Minmod, SlopeLimiter::VanLeer, SlopeLimiter::FirstOrder})
    {
        assert(limited_slope(1, -2, limiter) == 0);
        assert(limited_slope(-1, 2, limiter) == 0);
        assert(limited_slope(0, 2, limiter) == 0);
        assert(limited_slope(2, 0, limiter) == 0);
        const double expected = limiter == SlopeLimiter::Minmod ? 1.0 :
                                limiter == SlopeLimiter::VanLeer ? 1.5 : 0.0;
        assert(limited_slope(1, 3, limiter) == expected);
        assert(limited_slope(3, 1, limiter) == expected);
        assert(limited_slope(-1, -3, limiter) == -expected);
        assert(limited_slope(2, 2, limiter) == (limiter == SlopeLimiter::FirstOrder ? 0 : 2));
    }
    assert(isfinite(limited_slope(1e308, 1e308, SlopeLimiter::VanLeer)));

    // Linear data must reconstruct exactly on uniform and stretched grids.
    for (double ratio : {1.0, 1.1})
    {
        Grid grid;
        grid.create(10, 2, 1, 2, ratio);
        vector<Primitive> q(grid.size());
        for (int i = 0; i < grid.size(); ++i)
        {
            const double value = 3 + 2 * grid.cell(i).r;
            q[i] = {value, value, value, value, value};
        }
        for (int i = grid.begin(); i < grid.end(); ++i)
        {
            const auto edges = reconstruct(grid, q, i);
            const double left = SLOPE_LIMITER == SlopeLimiter::FirstOrder ? q[i].rho : 3 + 2 * grid.face(i).r;
            const double right = SLOPE_LIMITER == SlopeLimiter::FirstOrder ? q[i].rho : 3 + 2 * grid.face(i + 1).r;
            for (auto field : {&Primitive::rho, &Primitive::u, &Primitive::p, &Primitive::w_plus, &Primitive::w_minus})
            {
                assert(abs(edges.left.*field - left) < 1e-12);
                assert(abs(edges.right.*field - right) < 1e-12);
            }
        }
    }
    cout << "Limiter checks passed: " << LIMITER_NAME << '\n';
}
