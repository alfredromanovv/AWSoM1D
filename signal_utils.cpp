#include "functions.h"
#include <csignal>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#endif

atomic<bool> g_stop{false};
static_assert(atomic<bool>::is_always_lock_free);

static void on_signal_posix(int)
{
    // The handler only requests a stop.
    g_stop.store(true, memory_order_relaxed);
}

#ifdef _WIN32
static BOOL WINAPI on_console_ctrl(DWORD code)
{
    if (code == CTRL_C_EVENT || code == CTRL_CLOSE_EVENT ||
        code == CTRL_BREAK_EVENT || code == CTRL_LOGOFF_EVENT ||
        code == CTRL_SHUTDOWN_EVENT) {
        // On console close or session shutdown, Windows may terminate the process
        // before main saves the state. This flag does not guarantee saving.
        g_stop.store(true, memory_order_relaxed);
        return TRUE;
    }
    return FALSE;
}
#endif

void install_signal_handlers()
{
    if (signal(SIGINT, on_signal_posix) == SIG_ERR ||
        signal(SIGTERM, on_signal_posix) == SIG_ERR) {
        HANDLE_ERROR("Cannot install signal handlers.");
    }

#ifdef _WIN32
    if (!SetConsoleCtrlHandler(on_console_ctrl, TRUE)) {
        HANDLE_ERROR("Cannot install console control handler.");
    }
#endif
}
