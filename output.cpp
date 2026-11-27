#include "functions.h"
#include <array>
#include <bit>
#include <charconv>
#include <limits>
#include <locale>

namespace {
constexpr uint32_t COLUMN_COUNT = 14;
using Row = array<double, COLUMN_COUNT>;
constexpr char COLUMNS[] = "r_left r_right r width area volume rho u p T momentum energy w_plus w_minus";
static_assert(sizeof(Row) == COLUMN_COUNT * sizeof(double));
static_assert(sizeof(double) == 8 && numeric_limits<double>::is_iec559);
static_assert(endian::native == endian::little);

vector<Row> prepare_rows(const Grid& grid, double time)
{
    if (!isfinite(time) || time < 0 || grid.size() == 0)
        throw invalid_argument("Output requires a nonempty grid and finite nonnegative time.");
    const int first = grid.begin();
    const int count = grid.end() - first;
    vector<Row> rows(count);
    int invalid = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(|:invalid)
#endif
    for (int j = 0; j < count; ++j) {
        const int i = first + j;
        const Cell& c = grid.cell(i);
        const State& s = c.state;
        if (!s.valid()) {
            invalid = 1;
            continue;
        }
        rows[j] = {grid.face(i).r, grid.face(i + 1).r, c.r, c.width, c.area, c.volume,
            s.rho, s.velocity(), s.pressure(), s.temperature(), s.momentum, s.energy,
            s.w_plus, s.w_minus};
        for (double value : rows[j])
            if (!isfinite(value))
                invalid = 1;
    }
    if (invalid)
        throw runtime_error("Cannot output invalid cell data.");
    return rows;
}

ofstream open_output(const string& filename, ios::openmode mode = ios::out)
{
    ofstream file;
    file.exceptions(ios::failbit | ios::badbit);
    file.imbue(locale::classic());
    file.open(filename, mode | ios::out | ios::trunc);
    file << setprecision(numeric_limits<double>::max_digits10);
    return file;
}

void write_parameters(const Grid& grid, double time, uint64_t step)
{
    auto file = open_output("parameters.txt");
    file << "# AWSoM1D run parameters; all physical quantities and time are dimensionless\n"
         << "final_time = " << time << "\nfinal_step = " << step << '\n'
         << "solver_implemented = true\n"
         << "# Active numerical method:\n";
#define WRITE_PARAMETER(name) file << #name " = " << name << '\n'
    WRITE_PARAMETER(RIEMANN_SOLVER);
    WRITE_PARAMETER(LIMITER_NAME);
    WRITE_PARAMETER(SCHEME_ORDER);
    WRITE_PARAMETER(TIME_INTEGRATOR);
    WRITE_PARAMETER(GAMMA);
    WRITE_PARAMETER(THETA);
    WRITE_PARAMETER(B);
    WRITE_PARAMETER(D);
    WRITE_PARAMETER(OMEGA_0);
    WRITE_PARAMETER(ENABLE_REFLECTION);
    WRITE_PARAMETER(SUPER_RADIAL_EXPANSION);
    if constexpr (SUPER_RADIAL_EXPANSION) {
        WRITE_PARAMETER(EXPANSION_F_MAX);
        WRITE_PARAMETER(EXPANSION_R);
        WRITE_PARAMETER(EXPANSION_SIGMA);
    }
    WRITE_PARAMETER(NX);
    WRITE_PARAMETER(NG);
    WRITE_PARAMETER(R_MIN);
    WRITE_PARAMETER(R_MAX);
    WRITE_PARAMETER(GRID_RATIO);
    WRITE_PARAMETER(U_PH_INITIAL);
    WRITE_PARAMETER(U_INITIAL_RISE);
    WRITE_PARAMETER(U_INITIAL_SCALE);
    WRITE_PARAMETER(RHO_INITIAL_SCALE);
    WRITE_PARAMETER(RHO_INITIAL_FLOOR);
    WRITE_PARAMETER(T_PH_INITIAL);
    WRITE_PARAMETER(T_INITIAL_RISE);
    WRITE_PARAMETER(T_INITIAL_SCALE);
    WRITE_PARAMETER(T_INITIAL_RADIAL_EXPONENT);
    WRITE_PARAMETER(W_PLUS_INITIAL_AREA_EXPONENT);
    WRITE_PARAMETER(W_MINUS_INITIAL_RATIO);
    WRITE_PARAMETER(W_MINUS_INITIAL_SCALE);
    WRITE_PARAMETER(CFL_NUMBER);
    WRITE_PARAMETER(SOURCE_CFL);
    WRITE_PARAMETER(LIMIT_SOURCE_STEP);
    WRITE_PARAMETER(END_TIME);
    WRITE_PARAMETER(PRINT_INTERVAL);
    WRITE_PARAMETER(OUTPUT_INTERVAL);
    WRITE_PARAMETER(CONVERGENCE_TOL);
    WRITE_PARAMETER(CONV_NEEDED);
    WRITE_PARAMETER(RESTART_FILE);
    WRITE_PARAMETER(SAVE_FILE);
#undef WRITE_PARAMETER
    const int first = grid.begin();
    const int last = grid.end() - 1;
    file << "actual_physical_cells = " << grid.end() - first
         << "\nactual_ghost_layers_per_side = " << first
         << "\nactual_r_min = " << grid.face(first).r
         << "\nactual_r_max = " << grid.face(grid.end()).r
         << "\nactual_first_width = " << grid.cell(first).width
         << "\nactual_last_width = " << grid.cell(last).width
         << "\nactual_first_width_ratio = ";
    if (last > first)
        file << grid.cell(first + 1).width / grid.cell(first).width;
    else
        file << "not_applicable";
    file << "\ngeometry = "
         << (SUPER_RADIAL_EXPANSION ? "Kopp-Holzer superradial; A(r_in)=1; volume=Gauss quadrature"
                                    : "radial 1D; A(r)=r^2; volume=integral(A dr)") << '\n'
         << "initial_conditions = " << (RESTART_FILE[0] ? "loaded from checkpoint" : "uniform state from include.h") << '\n'
         << "inner_boundary = fixed rho=1, T=1, w_plus=OMEGA_0; copied u, w_minus\n"
         << "outer_boundary = constant extrapolation\n"
         << "signal_speed_estimate = max(VA,sqrt(cs2+cw2)); "
            "cs2=GAMMA*p/rho; cw2=3*(w_plus+w_minus)/(4*rho)\n"
         << "output_cells = physical only (no ghosts)\n"
         << "binary_format = AWSoM1D version 2; little-endian IEEE754 float64\n"
         << "binary_header = magic[8]=AWS1DOUT, uint32 version, uint32 columns, "
            "uint64 cells, uint64 step, float64 time, float64 eta_diss (48 bytes total)\n"
         << "binary_rows = cell-major; 14 float64 values per cell\n"
         << "columns = " << COLUMNS << '\n';
#ifdef _OPENMP
    file << "OpenMP = " << _OPENMP << "\nOpenMP_max_threads = " << omp_get_max_threads() << '\n';
#else
    file << "OpenMP = disabled\n";
#endif
    file.close();
}
} // namespace

void save_results(const Grid& grid, double time, uint64_t step)
{
    const auto rows = prepare_rows(grid, time);
    // Shortest round-trippable spelling: distinct double times have distinct names.
    char buffer[64];
    const auto converted = to_chars(buffer, buffer + sizeof(buffer), time == 0 ? 0.0 : time);
    if (converted.ec != errc{})
        throw runtime_error("Cannot format output time.");
    const string filename = "results_" + string(buffer, converted.ptr);
    const double eta = dissipation_stiffness(grid);
    auto file = open_output(filename, ios::binary);
    file.write("AWS1DOUT", 8);
    const uint32_t version = 2;
    const uint64_t count = rows.size();
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    file.write(reinterpret_cast<const char*>(&COLUMN_COUNT), sizeof(COLUMN_COUNT));
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));
    file.write(reinterpret_cast<const char*>(&step), sizeof(step));
    file.write(reinterpret_cast<const char*>(&time), sizeof(time));
    file.write(reinterpret_cast<const char*>(&eta), sizeof(eta));
    file.write(reinterpret_cast<const char*>(rows.data()),
        static_cast<streamsize>(rows.size() * sizeof(Row)));
    file.close();
}

static void write_text_profile(const string& filename, const Grid& grid, double time, uint64_t step)
{
    const auto rows = prepare_rows(grid, time);
    // Format in memory and issue one bulk write instead of many small disk writes.
    ostringstream text;
    text.imbue(locale::classic());
    text << setprecision(numeric_limits<double>::max_digits10)
         << "# dimensionless_time = " << time << "\n# step = " << step
         << "\n# eta_diss = " << dissipation_stiffness(grid)
         << "\n# " << COLUMNS << '\n';
    for (const Row& row : rows) {
        for (size_t j = 0; j < row.size(); ++j) {
            if (j != 0) text << '\t';
            text << row[j];
        }
        text << '\n';
    }
    const string data = text.str();
    auto file = open_output(filename);
    file.write(data.data(), static_cast<streamsize>(data.size()));
    file.close();
}

void save_initial_results(const Grid& grid, double time, uint64_t step)
{
    save_results(grid, time, step);
    write_text_profile("initial_conditions.txt", grid, time, step);
}

void save_final_results(const Grid& grid, double time, uint64_t step)
{
    write_text_profile("results.txt", grid, time, step);
    write_parameters(grid, time, step);
}

void output_results()
{
    save_results(g_grid, g_time, g_step);
    save_final_results(g_grid, g_time, g_step);
    cout << "Saved final results and parameters at t = " << g_time << ".\n";
}
