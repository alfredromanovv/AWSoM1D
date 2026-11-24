#pragma once

#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <random>
#include <vector>
#include <atomic>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

#define HANDLE_ERROR(msg) \
    do { \
        cerr << "Error: " << msg << "\nFile: " << __FILE__ \
                  << "\nLine: " << __LINE__ << endl; \
        throw runtime_error(msg); \
    } while (0)

extern atomic<bool> g_stop;

// All physical quantities, coordinates, and times below are dimensionless.
// Computational example from Section 9 of the model document.
constexpr double GAMMA = 5.0 / 3.0;         // Adiabatic index.
constexpr double THETA = 0.0036;            // Equation of state: p = THETA * rho * T.
constexpr double B = 1.50;                  // Alfven speed coefficient: VA = B / (A * sqrt(rho)).
constexpr double D = 35.0;                  // Wave dissipation coefficient d.
constexpr double OMEGA_0 = 0.08;            // Initial outgoing wave energy density at r = 1.
constexpr bool ENABLE_REFLECTION = true;  // Enable energy transfer between the wave populations.

// One-dimensional radial grid. Flux tube cross section: A(r) = r^2.
constexpr int NX = 600;                    // Number of physical cells.
constexpr int NG = 2;                      // Number of ghost layers on each side.
constexpr double R_MIN = 1.0;               // Inner boundary at the solar surface.
constexpr double R_MAX = 30.0;              // Outer boundary in solar-radius units.
constexpr double GRID_RATIO = 1.008;        // Adjacent cell width ratio; 1 gives a uniform grid.

// Trial initial profiles, evaluated at physical cell centers at t = 0.
// Here r >= 1, A(1) = 1, and exp denotes the exponential function.

// Velocity: u(r) = U_BASE + U_AMPLITUDE * (1 - exp(-(r - 1) / U_SCALE)).
constexpr double U_BASE = 0.02;             // Velocity at r = 1.
constexpr double U_AMPLITUDE = 1.45;        // Added velocity far from the base; u tends to 1.47.
constexpr double U_SCALE = 3.5;             // Radial e-folding length of the velocity increase.

// Density: rho(r) = max(U_BASE / (A(r) * u(r))
//   + (1 - U_BASE / u(r)) * exp(-(r - 1) / RHO_SCALE), INITIAL_RHO_MIN).
// This gives rho(1) = 1 and approximately A*rho*u = U_BASE far from the base,
// until the density floor becomes active.
constexpr double RHO_SCALE = 0.25;          // Decay length of the near-base density correction.
constexpr double INITIAL_RHO_MIN = 1e-8;    // Density floor for initialization only.

// Temperature: T(r) = 1 + T_AMPLITUDE * (1 - exp(-(r - 1) / T_SCALE)) * r^T_POWER.
// T(1) = 1; the added term first rises and then decays with radius.
constexpr double T_AMPLITUDE = 20.0;        // Multiplier of the temperature excess, not its peak value.
constexpr double T_SCALE = 0.8;             // Radial e-folding length of the temperature rise.
constexpr double T_POWER = -0.3;            // Power controlling the outer temperature excess decay.

// Wave energy densities (not wave amplitudes):
// w_plus(r)  = OMEGA_0 * A(r)^W_AREA_POWER.
// w_minus(r) = W_MINUS_FRACTION * w_plus(r) * (1 - exp(-(r - 1) / W_MINUS_SCALE)).
// Initially w_plus(1) = OMEGA_0 and w_minus(1) = 0.
constexpr double W_AREA_POWER = -1.2;       // Outgoing energy scales as r^-2.4 when A(r) = r^2.
constexpr double W_MINUS_FRACTION = 0.02;   // Far-field ratio w_minus / w_plus.
constexpr double W_MINUS_SCALE = 0.5;       // Radial e-folding length of the incoming wave fraction rise.

// Derived initial quantities: p = THETA * rho * T, momentum density = rho * u,
// E = rho*u*u/2 + p/(GAMMA - 1) + w_plus + w_minus.
// These profiles are initial guesses, not a stationary solution or fixed boundary data.
// In these comments, ^ denotes exponentiation (use pow in C++).

// Dimensionless solver settings.
enum class SlopeLimiter { Minmod, VanLeer, FirstOrder };
// ===== ВЫБОР ОГРАНИЧИТЕЛЯ =====
// В строке SLOPE_LIMITER ниже поставьте один из вариантов:
//   SlopeLimiter::Minmod     — minmod, более сглаживающий ограничитель.
//   SlopeLimiter::VanLeer    — van Leer, менее сглаживающий ограничитель.
//   SlopeLimiter::FirstOrder — первый порядок по пространству (нулевые наклоны).
// Minmod и VanLeer дают второй порядок по пространству на гладких участках.
// По времени во всех вариантах используется SSP-RK2.
// После изменения пересоберите проект и запустите расчёт заново.
constexpr SlopeLimiter SLOPE_LIMITER = SlopeLimiter::VanLeer;

// Название для parameters.txt определяется автоматически.
static_assert(SLOPE_LIMITER == SlopeLimiter::Minmod || SLOPE_LIMITER == SlopeLimiter::VanLeer ||
              SLOPE_LIMITER == SlopeLimiter::FirstOrder);
constexpr const char* LIMITER_NAME = SLOPE_LIMITER == SlopeLimiter::Minmod ? "minmod" :
                                   SLOPE_LIMITER == SlopeLimiter::VanLeer ? "vanleer" : "first_order";
enum class FluxMethod { HLL, HLLC };
constexpr FluxMethod FLUX_METHOD = FluxMethod::HLL; // HLLC is reserved.
constexpr const char* RIEMANN_SOLVER = FLUX_METHOD == FluxMethod::HLL ? "HLL" : "HLLC";
constexpr bool LIMIT_SOURCE_STEP = false; // Enable the explicit source time-step restriction.
constexpr double SOURCE_CFL = 0.5;
constexpr double END_TIME = 1000.0; // Absolute final time, including restarted runs.
inline constexpr char TIME_INTEGRATOR[] = "SSP-RK2";
constexpr int SCHEME_ORDER = SLOPE_LIMITER == SlopeLimiter::FirstOrder ? 1 : 2; // Spatial order; time remains SSP-RK2.
constexpr double CFL_NUMBER = 0.45;
constexpr double PRINT_INTERVAL = 0.5;      // Simulation-time interval between progress reports.
constexpr double OUTPUT_INTERVAL = 0.5;     // Simulation-time interval between profile outputs.
constexpr double CONVERGENCE_TOL = 1e-9;     // Threshold for max |L_k|/(1+|U_k|) over cells and components.
constexpr int CONV_NEEDED = 5;              // Required consecutive successful convergence checks.

// An empty restart path selects initialization from the profiles above.
inline constexpr char RESTART_FILE[] = "";
inline constexpr char SAVE_FILE[] = "restart.dat";
