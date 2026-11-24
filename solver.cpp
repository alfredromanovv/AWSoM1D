#include "functions.h"
#include <array>
#include <limits>

namespace
{
using Vector = array<double, 5>;
Vector components(const State &s)
{
    return {s.rho, s.momentum, s.energy, s.w_plus, s.w_minus};
}
State state(const Vector &v)
{
    return {v[0], v[1], v[2], v[3], v[4]};
}
struct Edges
{
    Primitive left, right;
};
struct Interface
{
    Vector flux;
    double velocity;
};
struct Bounds
{
    double cfl, gamma, residual;
};
struct Work
{
    vector<Primitive> q;
    vector<double> va;
    vector<Edges> edges;
    vector<Interface> faces;
    vector<Vector> rhs;
    vector<Bounds> bounds;
    explicit Work(int n) : q(n), va(n), edges(n), faces(n + 1), rhs(n), bounds(n) {}
};

double limited_slope(double a, double b, SlopeLimiter limiter = SLOPE_LIMITER)
{
    if (limiter == SlopeLimiter::FirstOrder || !((a > 0 && b > 0) || (a < 0 && b < 0)))
        return 0.0;
    const double lo = min(abs(a), abs(b)), hi = max(abs(a), abs(b));
    // van Leer: 2*a*b/(a+b), avoiding products of large slopes.
    const double magnitude = limiter == SlopeLimiter::VanLeer ? lo / (0.5 + 0.5 * (lo / hi)) : lo;
    return copysign(magnitude, a);
}

Primitive reconstruction_floors(Primitive q)
{
    q.rho = max(q.rho, 1e-12);
    q.p = max(q.p, 1e-12);
    q.w_plus = max(q.w_plus, 0.0);
    q.w_minus = max(q.w_minus, 1e-15);
    return q;
}

// Limited slopes use actual distances between neighboring centers.
Edges reconstruct(const Grid &grid, const vector<Primitive> &q, int i)
{
    if constexpr (SLOPE_LIMITER == SlopeLimiter::FirstOrder)
    {
        const Primitive edge = reconstruction_floors(q[i]);
        return {edge, edge};
    }
    const double dl = grid.cell(i).r - grid.cell(i - 1).r;
    const double dr = grid.cell(i + 1).r - grid.cell(i).r;
    Primitive left = q[i], right = q[i];
    for (auto field :
         {&Primitive::rho, &Primitive::u, &Primitive::p, &Primitive::w_plus, &Primitive::w_minus})
    {
        const double a = (q[i].*field - q[i - 1].*field) / dl;
        const double b = (q[i + 1].*field - q[i].*field) / dr;
        const double slope = limited_slope(a, b);
        left.*field += slope * (grid.face(i).r - grid.cell(i).r);
        right.*field += slope * (grid.face(i + 1).r - grid.cell(i).r);
    }
    return {reconstruction_floors(left), reconstruction_floors(right)};
}

struct Side
{
    Vector U, F;
    double fast;
};

double fast_speed(const Primitive &q, double va)
{
    const double cs2 = GAMMA * q.p / q.rho;
    const double cw2 = 0.75 * (q.w_plus + q.w_minus) / q.rho;
    return max(va, sqrt(cs2 + cw2));
}

double cell_cfl(const Cell &cell, const Primitive &q, double va)
{
    const double fast = fast_speed(q, va);
    const double rate = max(abs(q.u - fast), abs(q.u + fast)) / cell.width;
    return CFL_NUMBER / max(rate, 1e-10);
}

Side side(const Primitive &q, double area)
{
    const double va = B / (area * sqrt(q.rho));
    const double waves = q.w_plus + q.w_minus;
    const double pressure = q.p + 0.5 * waves;
    const double m = q.rho * q.u;
    const double E = q.p / (GAMMA - 1) + 0.5 * m * q.u + waves;
    return {{q.rho, m, E, q.w_plus, q.w_minus},
            {m, m * q.u + pressure, q.u * (E + pressure) + va * (q.w_plus - q.w_minus), (q.u + va) * q.w_plus,
             (q.u - va) * q.w_minus},
            fast_speed(q, va)};
}

Interface hll(const Primitive &left, const Primitive &right, double area)
{
    const Side l = side(left, area), r = side(right, area);
    const double sl = min(left.u - l.fast, right.u - r.fast);
    const double sr = max(left.u + l.fast, right.u + r.fast);
    if (sl >= 0)
        return {l.F, left.u};
    if (sr <= 0)
        return {r.F, right.u};
    Interface face{};
    for (int k = 0; k < 5; ++k)
        face.flux[k] = (sr * l.F[k] - sl * r.F[k] + sl * sr * (r.U[k] - l.U[k])) / (sr - sl);
    // HLL-weighted velocity for the wave compression source.
    face.velocity = (sr * left.u - sl * right.u) / (sr - sl);
    return face;
}

double imbalance(double plus, double minus)
{
    if (plus > 0 && plus >= 4 * minus)
        return 1 - 2 * sqrt(minus / plus);
    if (minus > 0 && minus >= 4 * plus)
        return 2 * sqrt(plus / minus) - 1;
    return 0;
}

double dissipation_coefficient(const Cell &cell, double density)
{
    return D / sqrt(cell.area * density);
}

double reflection_rate(const Grid &grid, const vector<double> &va, int i, double plus, double minus, double gamma)
{
    if constexpr (!ENABLE_REFLECTION)
        return 0.0;
    const double gradient = (va[i + 1] - va[i - 1]) / (grid.cell(i + 1).r - grid.cell(i - 1).r);
    return imbalance(plus, minus) * min(abs(gradient), gamma);
}

Bounds evaluate(const Grid &grid, Work &w)
{
    const int begin = grid.begin(), end = grid.end(), n = grid.size();
    if (begin < 1 || end <= begin)
        throw invalid_argument("Solver requires physical cells and a ghost layer.");
    const double infinity = numeric_limits<double>::infinity();
#ifdef _OPENMP
    const int threads = min(omp_get_max_threads(), max(1, n / 128));
#pragma omp parallel num_threads(threads)
#endif
    {
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int i = 0; i < n; ++i)
        {
            const Cell &c = grid.cell(i);
            const State &s = c.state;
            w.q[i] = {s.rho, s.velocity(), s.pressure(), s.w_plus, s.w_minus};
            w.va[i] = B / (c.area * sqrt(s.rho));
            w.edges[i] = {w.q[i], w.q[i]};
        }
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int i = begin; i < end; ++i)
            w.edges[i] = reconstruct(grid, w.q, i);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int i = begin; i <= end; ++i)
        {
            Primitive right = w.edges[i].left;
            if (i == begin)
                right = w.q[i];
            else if (i == end)
                right = reconstruction_floors(right); // The extrapolated ghost has zero slope.
            w.faces[i] = hll(w.edges[i - 1].right, right, grid.face(i).area);
        }
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int i = begin; i < end; ++i)
        {
            const Cell &c = grid.cell(i);
            const auto &q = w.q[i];
            const auto &l = w.faces[i];
            const auto &r = w.faces[i + 1];
            const double al = grid.face(i).area, ar = grid.face(i + 1).area;
            auto &rhs = w.rhs[i];
            // Divide the legacy AU equations by the time-independent cell area.
            const double area_width = c.area * c.width;
            for (int k = 0; k < 5; ++k)
                rhs[k] = (al * l.flux[k] - ar * r.flux[k]) / area_width;
            const double gravity = 1 / (c.r * c.r);
            const double area_gradient = SUPER_RADIAL_EXPANSION ? c.area_gradient / c.area : 2 / c.r;
            rhs[1] += (q.p + 0.5 * (q.w_plus + q.w_minus)) * area_gradient - q.rho * gravity;
            rhs[2] -= c.state.momentum * gravity;

            const double diss = dissipation_coefficient(c, q.rho);
            const double gamma = diss * sqrt(max(q.w_plus, q.w_minus));
            const double reflection = reflection_rate(grid, w.va, i, q.w_plus, q.w_minus, gamma);
            const double transfer = reflection * sqrt(q.w_plus) * sqrt(q.w_minus);
            const double sp = -transfer - diss * q.w_plus * sqrt(q.w_minus);
            const double sm = transfer - diss * q.w_minus * sqrt(q.w_plus);
            const double divergence = (ar * r.velocity - al * l.velocity) / area_width;
            rhs[3] -= 0.5 * q.w_plus * divergence;
            rhs[4] -= 0.5 * q.w_minus * divergence;
            double residual = 0;
            const Vector u = components(c.state);
            for (int k = 0; k < 5; ++k)
            {
                const double full_rhs = rhs[k] + (k == 3 ? sp : k == 4 ? sm : 0.0);
                if (!isfinite(full_rhs))
                {
                    residual = infinity;
                    break;
                }
                residual = max(residual, abs(full_rhs) / (1 + abs(u[k])));
            }
            w.bounds[i] = {cell_cfl(c, q, w.va[i]), gamma, residual};
        }
    }
    Bounds b{infinity, 0, 0};
    for (int i = begin; i < end; ++i)
    {
        const auto &v = w.bounds[i];
        b.cfl = min(b.cfl, v.cfl);
        b.gamma = max(b.gamma, v.gamma);
        b.residual = max(b.residual, v.residual);
    }
    return b;
}

bool apply_state_floors(State &s)
{
    for (double value : components(s))
        if (!isfinite(value))
            return false;
    s.rho = max(s.rho, 1e-14);
    s.w_plus = max(s.w_plus, 0.0);
    s.w_minus = max(s.w_minus, 1e-16);
    // Store the corrected energy so subsequent conversions see the pressure floor.
    constexpr double pressure_floor = 1e-14;
    if (s.pressure() < pressure_floor)
    {
        s.energy = ((0.5 * s.momentum * s.velocity() + pressure_floor / (GAMMA - 1)) + s.w_plus) + s.w_minus;
        if (s.pressure() < pressure_floor)
            s.energy = nextafter(s.energy, numeric_limits<double>::infinity());
    }
    return s.valid();
}

bool update_transport(Grid &dest, const Grid &base, const Grid &stage, const Work &w, double dt, bool second)
{
    int invalid = 0;
    const int first = base.begin(), end = base.end();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(| : invalid)                                             \
    num_threads(min(omp_get_max_threads(), max(1, (end - first) / 128)))
#endif
    for (int i = first; i < end; ++i)
    {
        auto v = components(stage.cell(i).state);
        const auto original = components(base.cell(i).state);
        for (int k = 0; k < 5; ++k)
        {
            v[k] += dt * w.rhs[i][k];
            if (second)
                v[k] = 0.5 * (original[k] + v[k]);
        }
        State s = state(v);
        if (apply_state_floors(s))
            dest.set_state(i, s);
        else
            invalid = 1;
    }
    if (invalid)
        return false;
    // The wave solve uses the ghost states from the start of this stage.
    for (int j = 0; j < first; ++j)
    {
        dest.set_state(j, stage.cell(j).state);
        dest.set_state(end + j, stage.cell(end + j).state);
    }
    return true;
}

bool update_wave_sources(Grid &grid, vector<double> &va, double dt)
{
    if constexpr (ENABLE_REFLECTION)
        for (int i = 0; i < grid.size(); ++i)
        {
            const Cell &c = grid.cell(i);
            va[i] = B / (c.area * sqrt(c.state.rho));
        }

    const int first = grid.begin(), end = grid.end();
    int invalid = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(| : invalid)                                             \
    num_threads(min(omp_get_max_threads(), max(1, (end - first) / 128)))
#endif
    for (int i = first; i < end; ++i)
    {
        const Cell &c = grid.cell(i);
        State s = c.state;
        const double plus = s.w_plus, minus = max(s.w_minus, 1e-16);
        const double ap = sqrt(plus), am = sqrt(minus);
        const double diss = dissipation_coefficient(c, s.rho);
        const double gamma = diss * max(ap, am);
        const double reflection = reflection_rate(grid, va, i, plus, minus, gamma);
        const double loss = diss * ap;

        // Solve the legacy 2x2 system for wave amplitudes, with a11 = 1.
        const double a12 = 0.5 * dt * (reflection + loss);
        const double a21 = -0.5 * dt * reflection;
        const double a22 = 1 + 0.5 * dt * loss;
        const double det = a22 - a12 * a21;
        if (!isfinite(det))
        {
            invalid = 1;
            continue;
        }
        const double ap_new = det > 1e-15 ? max((a22 * ap - a12 * am) / det, 0.0) : ap;
        const double am_new = det > 1e-15 ? max((am - a21 * ap) / det, 1e-8) : am;
        s.w_plus = ap_new * ap_new;
        s.w_minus = am_new * am_new;
        // Recover pressure from the same total energy and the updated waves.
        if (apply_state_floors(s))
            grid.set_state(i, s);
        else
            invalid = 1;
    }
    if (invalid)
        return false;
    apply_boundary_conditions(grid);
    return true;
}
} // namespace

double dissipation_stiffness(const Grid &grid)
{
    const int first = grid.begin(), end = grid.end();
    if (first == end)
        throw invalid_argument("Empty grid for dissipation diagnostic.");
    double cfl = numeric_limits<double>::infinity(), gamma = 0;
    for (int i = first; i < end; ++i)
    {
        const auto &c = grid.cell(i);
        const Primitive q = c.state.to_primitive();
        cfl = min(cfl, cell_cfl(c, q, B / (c.area * sqrt(q.rho))));
        gamma = max(gamma, dissipation_coefficient(c, q.rho) * sqrt(max(q.w_plus, q.w_minus)));
    }
    return cfl * gamma;
}

void calculate()
{
    Grid intermediate = g_grid, next = g_grid;
    Work first(g_grid.size()), second(g_grid.size());
    ofstream log("convergence.txt", RESTART_FILE[0] ? ios::app : ios::trunc);
    log.exceptions(ios::badbit | ios::failbit);
    log << "# step time dt residual eta_diss (residual=max_i,k |L_k|/(1+|U_k|))\n" << setprecision(17);
    auto next_event = [](double t, double interval) {
        const double index = floor(t / interval) + 1;
        const double event = index * interval;
        return event > t ? event : (index + 1) * interval;
    };
    double print_at = next_event(g_time, PRINT_INTERVAL), save_at = next_event(g_time, OUTPUT_INTERVAL);
    int consecutive = 0;
    double last_dt = 0;
    for (;;)
    {
        const Bounds b = evaluate(g_grid, first);
        if (!isfinite(b.residual))
            throw runtime_error("Nonfinite spatial operator.");
        if (last_dt == 0 || g_time >= print_at || g_time >= END_TIME || g_stop.load())
        {
            cout << "step=" << g_step << " t=" << g_time << " dt=" << last_dt << " residual=" << b.residual
                 << " eta_diss=" << b.cfl * b.gamma << endl;
            log << g_step << ' ' << g_time << ' ' << last_dt << ' ' << b.residual << ' ' << b.cfl * b.gamma
                << endl;
            consecutive = b.residual < CONVERGENCE_TOL ? consecutive + 1 : 0;
            print_at = next_event(g_time, PRINT_INTERVAL);
        }
        if (consecutive >= CONV_NEEDED)
        {
            cout << "Stationary residual tolerance reached.\n";
            break;
        }
        if (g_stop.load() || g_time >= END_TIME)
            break;
        const double remaining = END_TIME - g_time;
        const double dt = min(remaining, b.cfl);
        if (!(g_time + dt > g_time))
            throw runtime_error("Time step underflow.");
        if (!update_transport(intermediate, g_grid, g_grid, first, dt, false) ||
            !update_wave_sources(intermediate, second.va, dt))
            throw runtime_error("Invalid predictor state after applying floors.");
        const Bounds mid = evaluate(intermediate, second);
        if (!isfinite(mid.residual) ||
            !update_transport(next, g_grid, intermediate, second, dt, true) ||
            !update_wave_sources(next, second.va, 0.5 * dt))
            throw runtime_error("Invalid corrector state after applying floors.");
        swap(g_grid, next);
        g_time = dt == remaining ? END_TIME : g_time + dt;
        ++g_step;
        last_dt = dt;
        if (g_time >= save_at)
        {
            save_results(g_grid, g_time, g_step);
            save_at = next_event(g_time, OUTPUT_INTERVAL);
        }
    }
}
