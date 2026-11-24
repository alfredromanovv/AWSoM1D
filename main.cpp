#include "functions.h"

int main()
{
    try {
        install_signal_handlers();

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
        save_results(g_grid, g_time, g_step);

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
