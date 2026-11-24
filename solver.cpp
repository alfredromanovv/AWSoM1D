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
    double velocity, speed;
};
struct Bounds
{
    double cfl, source, gamma, residual;
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

// Limited slopes use actual distances between neighboring centers.
Edges reconstruct(const Grid &grid, const vector<Primitive> &q, int i)
{
    if constexpr (SLOPE_LIMITER == SlopeLimiter::FirstOrder)
        return {q[i], q[i]};
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
    return {left, right};
}

struct Side
{
    Vector U, F;
    double fast;
};
Side side(const Primitive &q, double area)
{
    const double va = B / (area * sqrt(q.rho));
    const double waves = q.w_plus + q.w_minus;
    const double pressure = q.p + 0.5 * waves;
    const double m = q.rho * q.u;
    const double E = q.p / (GAMMA - 1) + 0.5 * m * q.u + waves;
    // Use the same fast-speed estimate as VVmain.cpp.
    const double cs2 = GAMMA * q.p / q.rho;
    const double cw2 = 0.75 * waves / q.rho;
    const double fast = max(va, sqrt(cs2 + cw2));
    return {{q.rho, m, E, q.w_plus, q.w_minus},
            {m, m * q.u + pressure, q.u * (E + pressure) + va * (q.w_plus - q.w_minus), (q.u + va) * q.w_plus,
             (q.u - va) * q.w_minus},
            fast};
}

Interface hll(const Primitive &left, const Primitive &right, double area)
{
    const Side l = side(left, area), r = side(right, area);
    const double sl = min(left.u - l.fast, right.u - r.fast);
    const double sr = max(left.u + l.fast, right.u + r.fast);
    const double speed = max(abs(sl), abs(sr));
    if (sl >= 0)
        return {l.F, left.u, speed};
    if (sr <= 0)
        return {r.F, right.u, speed};
    Interface face{};
    for (int k = 0; k < 5; ++k)
        face.flux[k] = (sr * l.F[k] - sl * r.F[k] + sl * sr * (r.U[k] - l.U[k])) / (sr - sl);
    // (rho u)_star / rho_star; the common factor 1/(sr-sl) cancels.
    face.velocity =
        (sr * r.U[1] - sl * l.U[1] - r.F[1] + l.F[1]) / (sr * r.U[0] - sl * l.U[0] - r.F[0] + l.F[0]);
    face.speed = speed;
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
            if constexpr (ENABLE_REFLECTION)
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
            w.faces[i] = hll(w.edges[i - 1].right, w.edges[i].left, grid.face(i).area);
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
            for (int k = 0; k < 5; ++k)
                rhs[k] = (al * l.flux[k] - ar * r.flux[k]) / c.volume;
            const double gravity = c.gravity_integral / c.volume;
            rhs[1] += (q.p + 0.5 * (q.w_plus + q.w_minus)) * (ar - al) / c.volume - q.rho * gravity;
            rhs[2] -= c.state.momentum * gravity;

            const double sqrt_area = SUPER_RADIAL_EXPANSION ? sqrt(c.area) : c.r;
            const double gamma = D / sqrt_area * sqrt(max(q.w_plus, q.w_minus) / q.rho);
            double transfer = 0.0;
            if constexpr (ENABLE_REFLECTION)
            {
                // Second-order derivative on the stretched radial mesh.
                const double dl = c.r - grid.cell(i - 1).r, dr = grid.cell(i + 1).r - c.r;
                const double gradient =
                    (dr * (w.va[i] - w.va[i - 1]) / dl + dl * (w.va[i + 1] - w.va[i]) / dr) / (dl + dr);
                transfer =
                    imbalance(q.w_plus, q.w_minus) * min(abs(gradient), gamma) * sqrt(q.w_plus) * sqrt(q.w_minus);
            }
            const double diss = SUPER_RADIAL_EXPANSION ? D * c.dissipation_integral / c.volume : D * c.r * gravity;
            const double sp = -transfer - diss * q.w_plus * sqrt(q.w_minus / q.rho);
            const double sm = transfer - diss * q.w_minus * sqrt(q.w_plus / q.rho);
            const double divergence = (ar * r.velocity - al * l.velocity) / c.volume;
            rhs[3] += sp - 0.5 * q.w_plus * divergence;
            rhs[4] += sm - 0.5 * q.w_minus * divergence;
            double source = infinity;
            if (sp < 0)
                source = SOURCE_CFL * q.w_plus / (-sp);
            if (sm < 0)
                source = min(source, SOURCE_CFL * q.w_minus / (-sm));
            double residual = 0;
            const Vector u = components(c.state);
            for (int k = 0; k < 5; ++k)
            {
                if (!isfinite(rhs[k]))
                {
                    residual = infinity;
                    break;
                }
                residual = max(residual, abs(rhs[k]) / (1 + abs(u[k])));
            }
            w.bounds[i] = {CFL_NUMBER * c.width / max(l.speed, r.speed), source, gamma, residual};
        }
    }
    Bounds b{infinity, infinity, 0, 0};
    for (int i = begin; i < end; ++i)
    {
        const auto &v = w.bounds[i];
        b.cfl = min(b.cfl, v.cfl);
        b.source = min(b.source, v.source);
        b.gamma = max(b.gamma, v.gamma);
        b.residual = max(b.residual, v.residual);
    }
    if (!LIMIT_SOURCE_STEP)
        b.source = infinity;
    return b;
}

// Trial updates affect only scratch grids until both stages have succeeded.
bool update(Grid &dest, const Grid &base, const Grid &stage, const Work &w, double dt, bool second)
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
        const State s = state(v);
        if (s.valid())
            dest.set_state(i, s);
        else
            invalid = 1;
    }
    if (invalid)
        return false;
    apply_boundary_conditions(dest);
    return true;
}
} // namespace

double dissipation_stiffness(const Grid &grid)
{
    // Output may be requested before ghost filling, or on a physical-only
    // mesh. Reconstruct using the same photospheric data as the solver.
    const int first = grid.begin(), end = grid.end();
    if (first == end)
        throw invalid_argument("Empty grid for dissipation diagnostic.");
    vector<Primitive> q(grid.size());
    vector<Edges> edges(grid.size());
    for (int i = first; i < end; ++i)
        q[i] = grid.cell(i).state.to_primitive();
    const Primitive photosphere = photospheric_boundary_state(grid.cell(first).state).to_primitive();
    if (first > 0)
        q[first - 1] = photosphere;
    edges[first] = {q[first], q[first]};
    edges[end - 1] = {q[end - 1], q[end - 1]};
    for (int i = first + (first == 0 ? 1 : 0); i < end - 1; ++i)
        edges[i] = reconstruct(grid, q, i);
    vector<double> speed(grid.size() + 1);
    for (int i = first; i <= end; ++i)
        speed[i] = hll(i == first ? photosphere : edges[i - 1].right,
                       edges[min(end - 1, i)].left, grid.face(i).area).speed;
    double cfl = numeric_limits<double>::infinity(), gamma = 0;
    for (int i = first; i < end; ++i)
    {
        const auto &c = grid.cell(i);
        cfl = min(cfl, CFL_NUMBER * c.width / max(speed[i], speed[i + 1]));
        const double sqrt_area = SUPER_RADIAL_EXPANSION ? sqrt(c.area) : c.r;
        gamma = max(gamma, D / sqrt_area * sqrt(max(q[i].w_plus, q[i].w_minus) / q[i].rho));
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
        const double target = min({END_TIME, print_at, save_at});
        double dt = min({target - g_time, b.cfl, b.source});
        for (;;)
        {
            if (!(g_time + dt > g_time))
                throw runtime_error("Time step underflow.");
            if (update(intermediate, g_grid, g_grid, first, dt, false))
            {
                const Bounds mid = evaluate(intermediate, second);
                if (dt <= min(mid.cfl, mid.source) && isfinite(mid.residual) &&
                    update(next, g_grid, intermediate, second, dt, true))
                    break;
            }
            dt *= 0.5;
            if (g_stop.load())
                return;
        }
        swap(g_grid, next);
        g_time = dt == target - g_time ? target : g_time + dt;
        ++g_step;
        last_dt = dt;
        if (g_time >= save_at)
        {
            save_results(g_grid, g_time, g_step);
            save_at = next_event(g_time, OUTPUT_INTERVAL);
        }
    }
}
