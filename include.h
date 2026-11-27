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

//// All physical quantities, coordinates, and times below are dimensionless.
//// Computational example from Section 9 of the model document.
//constexpr double GAMMA = 5.0 / 3.0;         // Adiabatic index.
//constexpr double THETA = 0.00363215;            // Equation of state: p = THETA * rho * T.
//constexpr double B = 0.111670;                  // Alfven speed coefficient: VA = B / (A * sqrt(rho)).
//constexpr double D = 293.3329;                  // Wave dissipation coefficient d.
//constexpr double OMEGA_0 = 0.0003534224;            // Initial outgoing wave energy density at r = 1.
//constexpr bool ENABLE_REFLECTION = true;  // Enable energy transfer between the wave populations.
//
//// Optional Kopp-Holzer flux-tube expansion, normalized at the inner boundary.
//constexpr bool SUPER_RADIAL_EXPANSION = true;
//constexpr double EXPANSION_F_MAX = 6.5;
//constexpr double EXPANSION_R = 1.3;
//constexpr double EXPANSION_SIGMA = 0.25;
//static_assert(EXPANSION_F_MAX >= 1 && EXPANSION_R > 0 && EXPANSION_SIGMA > 0);

// Solar wind calculator: R*=R_sun, U*=sqrt(GM/R*)
// rho* = 3.3452438519e-10 kg/m^3; T* = 50000 K
// B* = 0.0005 T; L_perp,0 = 4743379.7887534 m
// w_+,0 = 0.022553373276327 J/m^3; mu = 0.6
constexpr double GAMMA = 5.0 / 3.0;
constexpr double THETA = 0.0036320362607286;
constexpr double B = 0.055834135972868;
constexpr double D = 293.33514539549;
constexpr double OMEGA_0 = 0.00035341173263479;
constexpr bool ENABLE_REFLECTION = true;

constexpr bool SUPER_RADIAL_EXPANSION = true;
constexpr double EXPANSION_F_MAX = 6.5;
constexpr double EXPANSION_R = 1.3;
constexpr double EXPANSION_SIGMA = 0.25;

// One-dimensional radial grid.
constexpr int NX = 700;                    // Number of physical cells.
constexpr int NG = 2;                      // Number of ghost layers on each side.
constexpr double R_MIN = 1.0;               // Inner boundary at the solar surface.
constexpr double R_MAX = 70.0;              // Outer boundary in solar-radius units.
constexpr double GRID_RATIO = 1.008;        // Adjacent cell width ratio; 1 gives a uniform grid.


// Dimensionless initial gas and wave profiles.
constexpr double U_PH_INITIAL = 0.015;
constexpr double U_INITIAL_RISE = 1.45;
constexpr double U_INITIAL_SCALE = 3.5;
constexpr double RHO_INITIAL_SCALE = 0.15;
constexpr double RHO_INITIAL_FLOOR = 1e-8;
constexpr double T_PH_INITIAL = 1.0;
constexpr double T_INITIAL_RISE = 25.0;
constexpr double T_INITIAL_SCALE = 0.3;
constexpr double T_INITIAL_RADIAL_EXPONENT = -0.25;
constexpr double W_PLUS_INITIAL_AREA_EXPONENT = -1.2;
constexpr double W_MINUS_INITIAL_RATIO = 0.02;
constexpr double W_MINUS_INITIAL_SCALE = 0.5;
static_assert(U_PH_INITIAL > 0 && U_INITIAL_RISE >= 0 && U_INITIAL_SCALE > 0);
static_assert(RHO_INITIAL_SCALE > 0 && RHO_INITIAL_FLOOR > 0);
static_assert(T_PH_INITIAL > 0 && T_INITIAL_RISE >= 0 && T_INITIAL_SCALE > 0);
static_assert(W_MINUS_INITIAL_RATIO >= 0 && W_MINUS_INITIAL_SCALE > 0);

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
constexpr double END_TIME = 30.0; // Absolute final time, including restarted runs.
inline constexpr char TIME_INTEGRATOR[] = "SSP-RK2";
constexpr int SCHEME_ORDER = SLOPE_LIMITER == SlopeLimiter::FirstOrder ? 1 : 2; // Spatial order; time remains SSP-RK2.
constexpr double CFL_NUMBER = 0.45;
constexpr double PRINT_INTERVAL = 0.5;      // Simulation-time interval between progress reports.
constexpr double OUTPUT_INTERVAL = 0.1;     // Simulation-time interval between profile outputs.
constexpr double CONVERGENCE_TOL = 1e-9;     // Threshold for max |L_k|/(1+|U_k|) over cells and components.
constexpr int CONV_NEEDED = 5;              // Required consecutive successful convergence checks.

// Parent directory for run1, run2, etc. Empty uses the launch directory.
// Example: LR"(D:\AWSoM1D\Results)"
inline constexpr wchar_t OUTPUT_ROOT[] = LR"(F:\Results\AWSoM1D)";

// An empty restart path selects initialization from the profiles above.
inline constexpr char RESTART_FILE[] = "";
inline constexpr char SAVE_FILE[] = "restart.dat";
