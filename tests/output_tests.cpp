#include "../functions.h"
#include <cassert>
#include <filesystem>
#include <limits>

// Link against the simulation sources, excluding main.cpp.


template<class T> static T read_value(ifstream& file)
{
    T value{};
    file.read(reinterpret_cast<char*>(&value), sizeof(value));
    assert(file.good());
    return value;
}

int main()
{
    Grid grid;
    grid.create(10, 2, 1, 2, 1);
    set_initial_conditions(grid);
    apply_boundary_conditions(grid);
    save_results(grid, 0.125, 7);
    assert(filesystem::file_size("results_0.125") == 48 + 10 * 14 * 8);
    ifstream binary("results_0.125", ios::binary);
    char magic[8];
    binary.read(magic, 8);
    assert(string(magic, 8) == "AWS1DOUT");
    assert(read_value<uint32_t>(binary) == 2);
    assert(read_value<uint32_t>(binary) == 14);
    assert(read_value<uint64_t>(binary) == 10);
    assert(read_value<uint64_t>(binary) == 7);
    assert(read_value<double>(binary) == 0.125);
    const double eta = read_value<double>(binary);
    assert(isfinite(eta) && eta > 0);
    save_final_results(grid, 0.125, 7);
    ifstream text("results.txt");
    string line;
    getline(text, line);
    assert(line == "# dimensionless_time = 0.125");
    getline(text, line);
    assert(line == "# step = 7");
    getline(text, line);
    assert(line.starts_with("# eta_diss = "));
    assert(stod(line.substr(13)) == eta);
    getline(text, line);
    int count = 0;
    while (getline(text, line)) {
        istringstream row(line);
        for (int j = 0; j < 14; ++j) {
            double value;
            assert(static_cast<bool>(row >> value));
            assert(value == read_value<double>(binary));
        }
        double extra;
        assert(!(row >> extra));
        ++count;
    }
    assert(count == 10);
    ifstream parameters("parameters.txt");
    const string info((istreambuf_iterator<char>(parameters)), {});
    assert(info.find("actual_physical_cells = 10\n") != string::npos);
    assert(info.find("SCHEME_ORDER = " + to_string(SCHEME_ORDER) + "\n") != string::npos);
    assert(info.find(string("LIMITER_NAME = ") + LIMITER_NAME + "\n") != string::npos);
    assert(info.find("RIEMANN_SOLVER = HLL\n") != string::npos);
    assert(info.find("final_step = 7\n") != string::npos);
    save_results(grid, nextafter(0.125, 1.0), 8);
    assert(filesystem::exists("results_0.12500000000000003"));
    bool rejected = false;
    try { save_results(grid, numeric_limits<double>::quiet_NaN(), 0); }
    catch (const invalid_argument&) { rejected = true; }
    assert(rejected);
    grid.clear_states();
    rejected = false;
    try { save_results(grid, 1, 0); }
    catch (const runtime_error&) { rejected = true; }
    assert(rejected && !filesystem::exists("results_1"));
    // Opening an output path must fail loudly if it is a directory.
    set_initial_conditions(grid);
    filesystem::create_directory("results_2");
    rejected = false;
    try { save_results(grid, 2, 0); }
    catch (const ios_base::failure&) { rejected = true; }
    assert(rejected);
    cout << "Output checks passed.\n";
}
