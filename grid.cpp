#include "grid.h"
#include <limits>

Grid g_grid;

namespace {
double expansion_factor(double r)
{
    const double x = (r - EXPANSION_R) / EXPANSION_SIGMA;
    const double e = exp(-abs(x));
    return 1 + (EXPANSION_F_MAX - 1) * (x >= 0 ? 1.0 : e) / (1 + e);
}
}

bool Primitive::valid() const
{
    return isfinite(rho) && rho > 0.0 && isfinite(u) && isfinite(p) && p > 0.0 &&
        isfinite(w_plus) && w_plus >= 0.0 && isfinite(w_minus) && w_minus >= 0.0;
}

State Primitive::to_conservative() const
{
    if (!valid())
        throw invalid_argument("Invalid primitive state.");
    State state{rho, rho * u,
        0.5 * rho * u * u + p / (GAMMA - 1.0) + w_plus + w_minus, w_plus, w_minus};
    if (!state.valid())
        throw invalid_argument("Unrepresentable conservative state.");
    return state;
}

Primitive State::to_primitive() const
{
    if (!valid())
        throw invalid_argument("Invalid conservative state.");
    Primitive primitive{rho, velocity(), pressure(), w_plus, w_minus};
    if (!primitive.valid())
        throw invalid_argument("Unrepresentable primitive state.");
    return primitive;
}

double State::pressure() const
{
    return (GAMMA - 1.0) * (energy - 0.5 * momentum * velocity() - w_plus - w_minus);
}

bool State::valid() const
{
    return isfinite(rho) && rho > 0.0 && isfinite(momentum) &&
        isfinite(energy) && isfinite(w_plus) && w_plus >= 0.0 &&
        isfinite(w_minus) && w_minus >= 0.0 &&
        isfinite(pressure()) && pressure() > 0.0;
}

State State::from_primitive(double density, double u, double temperature,
                          double outgoing, double incoming)
{
    if (!isfinite(density) || density <= 0.0 || !isfinite(u) ||
        !isfinite(temperature) || temperature <= 0.0)
        throw invalid_argument("Invalid primitive state.");
    return Primitive{density, u, THETA * density * temperature, outgoing, incoming}.to_conservative();
}

void Grid::create(int count, int ghosts, double r_min, double r_max, double ratio)
{
    if (count <= 0 || count == numeric_limits<int>::max() || ghosts < 0 ||
        ghosts > (numeric_limits<int>::max() - count - 1) / 2 ||
        !isfinite(r_min) || !isfinite(r_max) || r_min <= 0.0 || r_max <= r_min ||
        !isfinite(ratio) || ratio <= 0.0)
        throw invalid_argument("Invalid grid parameters.");

    const int total = count + 2 * ghosts;
    const double log_ratio = log(ratio);
    const double denominator = expm1(count * log_ratio);
    if (!isfinite(denominator))
        throw invalid_argument("Grid stretching is too large.");
    vector<Face> faces(total + 1);
    vector<Cell> cells(total);
    const double area_norm = SUPER_RADIAL_EXPANSION ? r_min * r_min * expansion_factor(r_min) : 1.0;
    // Closed-form face positions avoid a serial prefix sum and are deterministic.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i <= total; ++i) {
        const int j = i - ghosts;
        const double fraction = ratio == 1.0 ? static_cast<double>(j) / count
            : expm1(j * log_ratio) / denominator;
        const double r = j == 0 ? r_min : (j == count ? r_max
            : r_min + (r_max - r_min) * fraction);
        faces[i] = {r, r * r};
        if constexpr (SUPER_RADIAL_EXPANSION)
            faces[i].area = j == 0 ? 1.0 : faces[i].area * (expansion_factor(r) / area_norm);
    }
    int invalid = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(|:invalid)
#endif
    for (int i = 0; i < total; ++i) {
        const double left = faces[i].r;
        const double right = faces[i + 1].r;
        Cell& c = cells[i];
        c.r = left + 0.5 * (right - left);
        c.width = right - left;
        c.area = c.r * c.r;
        c.volume = c.width * (left * left + left * right + right * right) / 3.0;
        c.gravity_integral = c.width;
        c.dissipation_integral = c.r * c.width;
        if constexpr (SUPER_RADIAL_EXPANSION) {
            c.area *= expansion_factor(c.r) / area_norm;
            c.volume = c.gravity_integral = c.dissipation_integral = 0.0;
            // Three-point Gauss-Legendre quadrature for the area and source integrals.
            const double half_width = 0.5 * c.width;
            const double offset = half_width * sqrt(3.0 / 5.0);
            auto integrate_at = [&](double r, double weight) {
                const double factor = expansion_factor(r) / area_norm;
                c.volume += weight * r * r * factor;
                c.gravity_integral += weight * factor;
                c.dissipation_integral += weight * r * sqrt(factor);
            };
            integrate_at(c.r - offset, 5.0 / 9);
            integrate_at(c.r, 8.0 / 9);
            integrate_at(c.r + offset, 5.0 / 9);
            c.volume *= half_width;
            c.gravity_integral *= half_width;
            c.dissipation_integral *= half_width;
        }
        if (!(left > 0.0) || !(c.width > 0.0) || !(c.volume > 0.0) ||
            !isfinite(c.volume) || !isfinite(c.gravity_integral) || !isfinite(c.dissipation_integral) ||
            !(faces[i].area > 0.0) || !isfinite(faces[i + 1].area))
            invalid = 1;
    }
    if (invalid)
        throw invalid_argument("Grid contains invalid or unresolved cells (including ghosts).");
    cells_.swap(cells);
    faces_.swap(faces);
    count_ = count;
    ghosts_ = ghosts;
}

void Grid::set_state(int i, const State& state)
{
    if (!state.valid())
        throw invalid_argument("Invalid cell state.");
    cells_.at(i).state = state;
}

void Grid::fill(const State& state)
{
    if (!state.valid())
        throw invalid_argument("Invalid cell state.");
    // Snapshot also permits fill(grid.cell(i).state) without aliasing writes.
    const State value = state;
    const int total = size();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < total; ++i)
        cells_[i].state = value;
}

void Grid::clear_states()
{
    const int total = size();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < total; ++i)
        cells_[i].state = State{};
}
