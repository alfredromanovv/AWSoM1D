#include "functions.h"
#include <filesystem>
#include <limits>
#include <locale>

double g_time = 0.0;
uint64_t g_step = 0;

void load_parameters()
{
    static_assert(NX > 0 && NG >= 1 && R_MIN >= 1 && R_MAX > R_MIN && GRID_RATIO > 0);
    static_assert(GAMMA > 1 && THETA > 0 && D >= 0 && CFL_NUMBER > 0 && CFL_NUMBER <= 1);
    static_assert(END_TIME > 0);
    static_assert(PRINT_INTERVAL > 0 && OUTPUT_INTERVAL > 0 && CONVERGENCE_TOL > 0 && CONV_NEEDED > 0);
    if (FLUX_METHOD == FluxMethod::HLLC)
        throw logic_error("HLLC is reserved but not implemented. Select FluxMethod::HLL in include.h.");
}

void create_grid()
{
    g_grid.create();
    cout << "Created grid: " << NX << " physical cells, " << NG << " ghost layers.\n";
}

// Full-precision decimal checkpoint, written only at shutdown. Store each pair
// of faces so a restart cannot silently reinterpret data on a different mesh.
void save_state(const string &filename)
{
    const string temporary = filename + ".tmp";
    ofstream file(temporary);
    file.exceptions(ios::failbit | ios::badbit);
    file.imbue(locale::classic());
    file << setprecision(numeric_limits<double>::max_digits10)
         << (SUPER_RADIAL_EXPANSION ? "AWS1D_RESTART_2 " : "AWS1D_RESTART_1 ")
         << g_grid.end() - g_grid.begin() << ' ' << g_time << ' ' << g_step;
    if constexpr (SUPER_RADIAL_EXPANSION)
        file << ' ' << EXPANSION_F_MAX << ' ' << EXPANSION_R << ' ' << EXPANSION_SIGMA;
    file << '\n';
    for (int i = g_grid.begin(); i < g_grid.end(); ++i)
    {
        const State &s = g_grid.cell(i).state;
        file << g_grid.face(i).r << ' ' << g_grid.face(i + 1).r << ' ' << s.rho << ' ' << s.momentum << ' '
             << s.energy << ' ' << s.w_plus << ' ' << s.w_minus << '\n';
    }
    file.close();
    filesystem::rename(temporary, filename);
}

void load_state(const string &filename)
{
    ifstream file(filename);
    file.imbue(locale::classic());
    string magic;
    int count;
    double time;
    uint64_t step;
    if (!(file >> magic >> count >> time >> step) ||
        count != g_grid.end() - g_grid.begin() || !isfinite(time) || time < 0)
        throw runtime_error("Invalid restart header: " + filename);
    if (magic != (SUPER_RADIAL_EXPANSION ? "AWS1D_RESTART_2" : "AWS1D_RESTART_1"))
        throw runtime_error("Restart geometry or format does not match this build: " + filename);
    if constexpr (SUPER_RADIAL_EXPANSION)
    {
        double f_max, r_exp, sigma;
        if (!(file >> f_max >> r_exp >> sigma) ||
            f_max != EXPANSION_F_MAX || r_exp != EXPANSION_R || sigma != EXPANSION_SIGMA)
            throw runtime_error("Restart expansion parameters do not match this build: " + filename);
    }
    Grid restored = g_grid;
    for (int i = restored.begin(); i < restored.end(); ++i)
    {
        State s;
        double left, right;
        if (!(file >> left >> right >> s.rho >> s.momentum >> s.energy >> s.w_plus >> s.w_minus) ||
            left != restored.face(i).r || right != restored.face(i + 1).r || !s.valid())
            throw runtime_error("Invalid restart geometry or state: " + filename);
        restored.set_state(i, s);
    }
    string trailing;
    if (file >> trailing)
        throw runtime_error("Extra records in restart: " + filename);
    apply_boundary_conditions(restored);
    g_grid = move(restored);
    g_time = time;
    g_step = step;
}
