#include "functions.h"
#include "mms.h"
#include <filesystem>

static void enter_run_directory()
{
    const filesystem::path output_root = OUTPUT_ROOT;
    if (!output_root.empty()) {
        if (!output_root.is_absolute())
            throw invalid_argument("OUTPUT_ROOT must be an absolute path");
        filesystem::create_directories(output_root);
    }

    for (uint64_t number = 1; ; ++number) {
        const filesystem::path path = output_root / ("run" + to_string(number));
        error_code error;
        if (filesystem::create_directory(path, error)) {
            filesystem::current_path(path);
            cout << "Output directory: " << filesystem::current_path() << endl;
            return;
        }
        if (error && error != errc::file_exists)
            throw filesystem::filesystem_error("Cannot create output directory", path, error);
    }
}

int main()
{
    try {
        install_signal_handlers();

        if constexpr (MMS_TEST) {
            enter_run_directory();
            mms::run_study();
            return EXIT_SUCCESS;
        }

        // 1. Parameters and grid geometry.
        load_parameters();
        create_grid();

        // 2. Initial conditions or restart from a saved state.
        if (RESTART_FILE[0] != '\0') {
            load_state(RESTART_FILE);
        } else {
            set_initial_conditions(g_grid);
        }
        apply_boundary_conditions(g_grid);
        // Resolve restart input in the launch directory before switching to output.
        enter_run_directory();
        save_initial_results(g_grid, g_time, g_step);

        // 3. The calculation must return when g_stop becomes true.
        if (!g_stop.load(memory_order_relaxed)) {
            calculate();
        }

        // 4. Common shutdown path, including a Ctrl+C stop.
        if (g_stop.load(memory_order_relaxed)) {
            cout << "Stop requested. Proceeding to output and save.\n";
        }
        output_results();
        save_state(SAVE_FILE);
        return EXIT_SUCCESS;
    } catch (const exception& error) {
        cerr << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
