// ==============================================================================
// AWSoM-1D: Квазиодномерный нестационарный солвер солнечного ветра
//           с турбулентным альфвеновским нагревом, отражением волн
//           и моделью переходного слоя (теплопроводность + излучение + прогонка Томаса)
// Компиляция: g++ -O3 -fopenmp awsom_1d.cpp -o awsom_1d
// ==============================================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <map>
#include <algorithm>
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#endif

// ==============================================================================
// 1. СТРУКТУРА ПАРАМЕТРОВ И ПАРСЕР ФАЙЛА param.ini
// ==============================================================================

struct Params {
    // Режим ввода: "dimensionless" или "physical"
    std::string input_mode = "dimensionless";

    // Сетка
    double r_in = 1.0;
    double r_out = 30.0;
    int N_cells = 600;
    double grid_stretch = 1.008;

    // Геометрия трубки
    int geom_type = 1;
    double f_max = 6.5;
    double r_exp = 1.3;
    double sigma_exp = 0.25;

    // Модель переходного слоя (Transition Region)
    int enable_tr = 1;
    double chi_0 = 2.0e-6;
    double lambda_rad = 0.35;
    double tau_mod = 6.0;
    double r_col = 5.0;
    double kappa_0_SI = 9.0e-12;
    double T_mod_SI = 3.0e5;

    // Безразмерные параметры (основные расчётные комплексы)
    double gamma = 5.0 / 3.0;
    double theta_0 = 0.0036;
    double V_A0 = 1.50;
    double alpha_d = 35.0;
    double w_plus0 = 0.08;

    // Физические константы и размерные параметры СИ (для пересчёта)
    double G_grav = 6.67430e-11;
    double M_sun = 1.98847e30;
    double R_sun = 6.95700e8;
    double mu_0 = 1.25663706212e-6;
    double k_B = 1.380649e-23;
    double m_u = 1.66053906660e-27;
    double mu_mol = 0.60;

    double rho_0 = 2.0e-11;
    double T_0 = 5.0e4;
    double B_0 = 8.0e-4;
    double L_perp0 = 4.0e7;
    double w_plus0_SI = 0.30;

    // Масштабы обезразмеривания (заполняются автоматически)
    double R_star = 6.95700e8;
    double V_star = 436675.0;
    double rho_star = 2.0e-11;
    double T_star = 5.0e4;
    double p_star = 1.0;
    double B_star = 8.0e-4;

    // Численные настройки
    std::string flux_scheme = "HLL";
    std::string limiter = "vanleer";
    double cfl = 0.45;
    double t_end = 60.0;
    int max_steps = 500000;
    double tol_steady = 1.0e-7;
    int output_every = 5000;
    int smooth_reflection = 0;
    double eps_reg = 1.0e-6;
    int omp_threads = 4;

    std::string output_profile = "solution_1d.csv";
    std::string output_history = "convergence_history.csv";
};

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

bool load_params_from_ini(const std::string& filename, Params& P) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[Ошибка] Не удалось открыть файл параметров: " << filename << std::endl;
        return false;
    }

    std::map<std::string, std::string> kv;
    std::string line;
    while (std::getline(file, line)) {
        size_t comm = line.find_first_of("#;");
        if (comm != std::string::npos) line = line.substr(0, comm);
        line = trim(line);
        if (line.empty() || line[0] == '[') continue;

        size_t eq = line.find('=');
        if (eq != std::string::npos) {
            std::string key = trim(line.substr(0, eq));
            std::string val = trim(line.substr(eq + 1));
            kv[key] = val;
        }
    }

    auto get_str = [&](const std::string& k, std::string& target) {
        if (kv.count(k)) target = kv[k];
    };
    auto get_dbl = [&](const std::string& k, double& target) {
        if (kv.count(k)) target = std::stod(kv[k]);
    };
    auto get_int = [&](const std::string& k, int& target) {
        if (kv.count(k)) target = std::stoi(kv[k]);
    };

    get_str("input_mode", P.input_mode);
    get_dbl("r_in", P.r_in);
    get_dbl("r_out", P.r_out);
    get_int("N_cells", P.N_cells);
    get_dbl("grid_stretch", P.grid_stretch);

    get_int("geom_type", P.geom_type);
    get_dbl("f_max", P.f_max);
    get_dbl("r_exp", P.r_exp);
    get_dbl("sigma_exp", P.sigma_exp);

    get_int("enable_tr", P.enable_tr);
    get_dbl("chi_0", P.chi_0);
    get_dbl("lambda_rad", P.lambda_rad);
    get_dbl("tau_mod", P.tau_mod);
    get_dbl("r_col", P.r_col);
    get_dbl("kappa_0_SI", P.kappa_0_SI);
    get_dbl("T_mod_SI", P.T_mod_SI);

    get_dbl("gamma", P.gamma);
    get_dbl("theta_0", P.theta_0);
    get_dbl("V_A0", P.V_A0);
    get_dbl("alpha_d", P.alpha_d);
    get_dbl("w_plus0", P.w_plus0);

    get_dbl("G_grav", P.G_grav);
    get_dbl("M_sun", P.M_sun);
    get_dbl("R_sun", P.R_sun);
    get_dbl("mu_0", P.mu_0);
    get_dbl("k_B", P.k_B);
    get_dbl("m_u", P.m_u);
    get_dbl("mu_mol", P.mu_mol);

    get_dbl("rho_0", P.rho_0);
    get_dbl("T_0", P.T_0);
    get_dbl("B_0", P.B_0);
    get_dbl("L_perp0", P.L_perp0);
    get_dbl("w_plus0_SI", P.w_plus0_SI);

    get_str("flux_scheme", P.flux_scheme);
    get_str("limiter", P.limiter);
    get_dbl("cfl", P.cfl);
    get_dbl("t_end", P.t_end);
    get_int("max_steps", P.max_steps);
    get_dbl("tol_steady", P.tol_steady);
    get_int("output_every", P.output_every);
    get_int("smooth_reflection", P.smooth_reflection);
    get_dbl("eps_reg", P.eps_reg);
    get_int("omp_threads", P.omp_threads);

    get_str("output_profile", P.output_profile);
    get_str("output_history", P.output_history);

    // Вычисление характерных масштабов
    P.R_star = P.R_sun;
    P.V_star = std::sqrt(P.G_grav * P.M_sun / P.R_star);

    const double Lambda_star = 1.0e-34; // Характерный масштаб функции охлаждения [Вт * м^3]
    const double c_rad_factor = 0.61;   // Фактор n_e * n_H / (rho / m_u)^2

    if (P.input_mode == "physical") {
        P.rho_star = P.rho_0;
        P.T_star   = P.T_0;
        P.p_star   = P.rho_star * P.V_star * P.V_star;
        P.B_star   = P.B_0;

        // Пересчёт физических параметров в безразмерные комплексы
        P.theta_0    = (P.k_B * P.T_0) / (P.mu_mol * P.m_u * P.V_star * P.V_star);
        P.V_A0       = P.B_0 / (P.V_star * std::sqrt(P.mu_0 * P.rho_0));
        P.alpha_d    = 2.0 * P.R_star / P.L_perp0;
        P.w_plus0    = P.w_plus0_SI / P.p_star;
        P.chi_0      = (P.kappa_0_SI * std::pow(P.T_0, 3.5)) / (P.R_star * P.rho_star * std::pow(P.V_star, 3.0));
        P.lambda_rad = (c_rad_factor * P.R_star * P.rho_star * Lambda_star) / (P.m_u * P.m_u * std::pow(P.V_star, 3.0));
        P.tau_mod    = P.T_mod_SI / P.T_0;
    } else {
        // Если заданы безразмерные параметры, вычисляем эквивалентные размерные масштабы
        P.rho_star = P.rho_0;
        P.p_star   = P.rho_star * P.V_star * P.V_star;
        P.T_star   = P.theta_0 * (P.mu_mol * P.m_u * P.V_star * P.V_star) / P.k_B;
        P.B_star   = P.V_A0 * P.V_star * std::sqrt(P.mu_0 * P.rho_star);
    }

    return true;
}

// ==============================================================================
// 2. ПОЛЬЗОВАТЕЛЬСКОЕ ЗАДАНИЕ ФОРМЫ МАГНИТНОЙ ТРУБКИ A(r) И ЕЁ ПРОИЗВОДНОЙ dA/dr
// ==============================================================================

double user_area_A(double r, const Params& P) {
    if (P.geom_type == 0) {
        return r * r;
    } 
    else if (P.geom_type == 1) {
        double exp_r = std::exp((P.r_exp - r) / P.sigma_exp);
        double exp_1 = std::exp((P.r_exp - 1.0) / P.sigma_exp);
        double f_num = P.f_max + exp_r;
        double f_den = 1.0 + exp_r;
        double f1_num = P.f_max + exp_1;
        double f1_den = 1.0 + exp_1;
        double f_r = (f_num / f_den) * (f1_den / f1_num);
        return r * r * f_r;
    } 
    else {
        return r * r * (1.0 + 0.5 * (r - 1.0));
    }
}

double user_area_dAdr(double r, const Params& P) {
    if (P.geom_type == 0) {
        return 2.0 * r;
    } 
    else if (P.geom_type == 1) {
        double exp_r = std::exp((P.r_exp - r) / P.sigma_exp);
        double exp_1 = std::exp((P.r_exp - 1.0) / P.sigma_exp);
        double f1_factor = (1.0 + exp_1) / (P.f_max + exp_1);

        double f_raw  = (P.f_max + exp_r) / (1.0 + exp_r);
        double df_raw = (P.f_max - 1.0) * exp_r / (P.sigma_exp * (1.0 + exp_r) * (1.0 + exp_r));

        double f_r  = f_raw * f1_factor;
        double df_r = df_raw * f1_factor;
        return 2.0 * r * f_r + r * r * df_r;
    } 
    else {
        return 2.0 * r * (1.0 + 0.5 * (r - 1.0)) + 0.5 * r * r;
    }
}

// ==============================================================================
// 3. ФУНКЦИЯ РАДИАЦИОННЫХ ПОТЕРЬ ПЕРЕХОДНОГО СЛОЯ (Rosner-Tucker-Vaiana / Klimchuk)
// ==============================================================================
// Возвращает нормированную функцию охлаждения Lambda(T) / (10^-34 Вт * м^3)
// с гладким выключением у хромосферной температуры T <= T_star (T_dimless <= 1.05).

inline double cooling_curve_hat(double T_dimless, const Params& P) {
    if (T_dimless <= 1.02) return 0.0;

    double T_K = T_dimless * P.T_star;
    double L_val = 0.0; // В единицах 10^-34 Вт * м^3 (= 10^-21 эрг * см^3 / с)

    if (T_K < 1.0e5) {
        // Пик излучения ионов углерода/кислорода в нижней части переходного слоя
        L_val = 1.0 * std::pow(T_K / 1.0e5, 2.0);
    } else if (T_K < 3.0e5) {
        L_val = 1.0 * std::pow(T_K / 1.0e5, -0.5);
    } else if (T_K < 2.0e6) {
        L_val = 0.58 * std::pow(T_K / 3.0e5, -0.5);
    } else {
        L_val = 0.22 * std::pow(T_K / 2.0e6, 0.5);
    }

    // Плавный рамп-фильтр у T_dimless = 1.0 для равновесия хромосферного резервуара
    double ramp = std::tanh((T_dimless - 1.02) / 0.15);
    return std::max(0.0, L_val * ramp);
}

// Модифицированная скорость объёмного охлаждения Q_rad с регуляризацией Линкера-Лионелло-Соколова
inline double compute_Qrad_modified(double rho, double T, const Params& P) {
    if (P.enable_tr == 0 || T <= 1.02) return 0.0;
    double L_hat = cooling_curve_hat(T, P);
    double s_mod = (T < P.tau_mod) ? std::pow(P.tau_mod / std::max(T, 0.5), 2.5) : 1.0;
    return P.lambda_rad * rho * rho * (L_hat / s_mod);
}

// ==============================================================================
// 4. СТРУКТУРЫ СОСТОЯНИЯ (ПРИМИТИВНЫЕ И КОНСЕРВАТИВНЫЕ ПЕРЕМЕННЫЕ)
// ==============================================================================

struct Prim {
    double rho;   // Плотность плазмы
    double u;     // Продольная скорость
    double p;     // Тепловое давление плазмы
    double wp;    // Плотность энергии выходящих волн w_+
    double wm;    // Плотность энергии входящих волн w_-
};

struct Cons {
    double U1;    // A * rho
    double U2;    // A * rho * u
    double U3;    // A * E, где E = rho*u^2/2 + p/(gamma-1) + wp + wm (полная энергия)
    double U4;    // A * wp
    double U5;    // A * wm
};

inline Cons prim_to_cons(const Prim& W, double A, double gamma) {
    Cons U;
    U.U1 = A * W.rho;
    U.U2 = A * W.rho * W.u;
    double E_g = 0.5 * W.rho * W.u * W.u + W.p / (gamma - 1.0);
    double E_tot = E_g + W.wp + W.wm;
    U.U3 = A * E_tot;
    U.U4 = A * W.wp;
    U.U5 = A * W.wm;
    return U;
}

inline Prim cons_to_prim(const Cons& U, double A, double gamma) {
    const double eps_floor = 1.0e-14;
    Prim W;
    W.rho = std::max(U.U1 / A, eps_floor);
    W.u   = (U.U2 / A) / W.rho;
    W.wp  = std::max(U.U4 / A, 0.0);
    W.wm  = std::max(U.U5 / A, eps_floor * 0.01);

    double E_tot = U.U3 / A;
    double E_kin = 0.5 * W.rho * W.u * W.u;
    double E_th  = E_tot - E_kin - W.wp - W.wm;
    W.p = std::max((gamma - 1.0) * E_th, eps_floor);
    return W;
}

inline Cons compute_flux(const Prim& W, double A, const Params& P) {
    double VA = P.V_A0 / (A * std::sqrt(W.rho));
    double PA = 0.5 * (W.wp + W.wm);
    double E_g = 0.5 * W.rho * W.u * W.u + W.p / (P.gamma - 1.0);

    Cons F;
    F.U1 = A * W.rho * W.u;
    F.U2 = A * (W.rho * W.u * W.u + W.p + PA);
    F.U3 = A * (W.u * (E_g + W.p + PA) + (W.u + VA) * W.wp + (W.u - VA) * W.wm);
    F.U4 = A * (W.u + VA) * W.wp;
    F.U5 = A * (W.u - VA) * W.wm;
    return F;
}

inline void wave_speeds(const Prim& W, double A, const Params& P, double& s_min, double& s_max) {
    double cs2 = P.gamma * W.p / W.rho;
    double VA  = P.V_A0 / (A * std::sqrt(W.rho));
    double cw2 = 0.75 * (W.wp + W.wm) / W.rho;
    double c_fast = std::max(VA, std::sqrt(cs2 + cw2));
    s_min = W.u - c_fast;
    s_max = W.u + c_fast;
}

inline double apply_limiter(double a, double b, const std::string& lim_type) {
    if (lim_type == "first_order") return 0.0;
    if (a * b <= 0.0) return 0.0;

    if (lim_type == "minmod") {
        return (a > 0.0) ? std::min(a, b) : std::max(a, b);
    } 
    else if (lim_type == "superbee") {
        double s = (a > 0.0) ? 1.0 : -1.0;
        double aa = std::abs(a), bb = std::abs(b);
        return s * std::max(std::min(2.0 * aa, bb), std::min(aa, 2.0 * bb));
    } 
    else {
        return (2.0 * a * b) / (a + b);
    }
}

inline double compute_C_imbalance(double wp, double wm) {
    const double eps = 1.0e-30;
    if (wp >= 4.0 * wm) {
        return 1.0 - 2.0 * std::sqrt(wm / (wp + eps));
    } else if (wm >= 4.0 * wp) {
        return 2.0 * std::sqrt(wp / (wm + eps)) - 1.0;
    } else {
        return 0.0;
    }
}

// ==============================================================================
// 5. ГЛАВНАЯ ПРОГРАММА И ЦИКЛ ИНТЕГРАЦИИ ПО ВРЕМЕНИ (RK2 + IMEX + ПРОГОНКА ТОМАСА)
// ==============================================================================

int main(int argc, char* argv[]) {
    std::string ini_file = "param.ini";
    if (argc > 1) ini_file = argv[1];

    Params P;
    if (!load_params_from_ini(ini_file, P)) {
        return 1;
    }

#ifdef _OPENMP
    if (P.omp_threads > 0) {
        omp_set_num_threads(P.omp_threads);
    }
#endif

    const int N = P.N_cells;
    const int ng = 2; // 2 фиктивные ячейки с каждого края
    const int N_tot = N + 2 * ng;

    std::vector<double> r_face(N_tot + 1);
    std::vector<double> r_cell(N_tot), dr(N_tot);
    std::vector<double> A_cell(N_tot), dAdr_cell(N_tot), A_face(N_tot + 1);

    double L_dom = P.r_out - P.r_in;
    double dr0;
    if (std::abs(P.grid_stretch - 1.0) < 1.0e-7) {
        dr0 = L_dom / N;
    } else {
        dr0 = L_dom * (P.grid_stretch - 1.0) / (std::pow(P.grid_stretch, N) - 1.0);
    }

    r_face[ng] = P.r_in;
    double cur_dr = dr0;
    for (int i = ng; i < ng + N; ++i) {
        r_face[i + 1] = r_face[i] + cur_dr;
        cur_dr *= P.grid_stretch;
    }
    r_face[ng + N] = P.r_out;

    double dr_left = r_face[ng + 1] - r_face[ng];
    r_face[ng - 1] = r_face[ng] - dr_left;
    r_face[ng - 2] = r_face[ng - 1] - dr_left;

    double dr_right = r_face[ng + N] - r_face[ng + N - 1];
    r_face[ng + N + 1] = r_face[ng + N] + dr_right;
    r_face[ng + N + 2] = r_face[ng + N + 1] + dr_right;

    double A_norm = user_area_A(P.r_in, P);

    for (int i = 0; i <= N_tot; ++i) {
        A_face[i] = user_area_A(r_face[i], P) / A_norm;
    }
    for (int i = 0; i < N_tot; ++i) {
        r_cell[i]    = 0.5 * (r_face[i] + r_face[i + 1]);
        dr[i]        = r_face[i + 1] - r_face[i];
        A_cell[i]    = user_area_A(r_cell[i], P) / A_norm;
        dAdr_cell[i] = user_area_dAdr(r_cell[i], P) / A_norm;
    }

    std::cout << "====================================================================\n";
    std::cout << "  AWSoM-1D: Нестационарный солвер солнечного ветра с переходным слоем\n";
    std::cout << "====================================================================\n";
    std::cout << "  Режим ввода параметров : " << P.input_mode << "\n";
    std::cout << "  Сетка                  : N = " << N << ", r in [" << P.r_in << ", " << P.r_out << "] R_sun\n";
    std::cout << "  Шаг сетки dr           : dr_min = " << dr[ng] << ", dr_max = " << dr[ng + N - 1] << "\n";
    std::cout << "  Переходный слой (TR)   : " << (P.enable_tr ? "ВКЛЮЧЕН (Спитцер + излучение + прогонка Томаса)" : "ВЫКЛЮЧЕН (q=0, Q_rad=0)") << "\n";
    std::cout << "  Численная схема        : " << P.flux_scheme << " + MUSCL (" << P.limiter << "), CFL = " << P.cfl << "\n";
    std::cout << "--------------------------------------------------------------------\n";
    std::cout << "  Безразмерные комплексы модели:\n";
    std::cout << "    theta_0    = " << P.theta_0    << "  (T_0 = " << P.T_star << " K)\n";
    std::cout << "    V_A0       = " << P.V_A0       << "  (V_A,0 / V_*)\n";
    std::cout << "    alpha_d    = " << P.alpha_d    << "  (2 R_sun / L_perp,0)\n";
    std::cout << "    w_plus0    = " << P.w_plus0    << "  (w_+,0 / (rho_0 V_*^2))\n";
    if (P.enable_tr) {
        std::cout << "    chi_0      = " << P.chi_0      << "  (параметр теплопроводности Спитцера)\n";
        std::cout << "    lambda_rad = " << P.lambda_rad << "  (параметр радиационных потерь)\n";
        std::cout << "    tau_mod    = " << P.tau_mod    << "  (порог регуляризации T_mod / T_0)\n";
    }
    std::cout << "====================================================================\n";

    std::vector<Prim> W(N_tot);
    std::vector<Cons> U(N_tot), U0(N_tot), RHS(N_tot);
    std::vector<Cons> F_face(N_tot + 1);
    std::vector<double> Au_face(N_tot + 1);
    std::vector<double> VA_cell(N_tot), R_ref_cell(N_tot), Qw_cell(N_tot);

    // Буферы для трёхдиагональной прогонки Томаса (размер N)
    std::vector<double> c_star(N, 0.0), d_star(N, 0.0), T_new(N, 1.0);

    // --------------------------------------------------------------------------
    // Инициализация начального состояния
    // --------------------------------------------------------------------------
    for (int i = 0; i < N_tot; ++i) {
        double r = std::max(r_cell[i], P.r_in);
        double u_init = 0.015 + 1.45 * (1.0 - std::exp(-(r - 1.0) / 3.5));
        double m_init = 0.015;
        double rho_init = (m_init / (A_cell[i] * u_init)) + (1.0 - m_init / u_init) * std::exp(-(r - 1.0) / 0.15);
        rho_init = std::max(rho_init, 1.0e-8);

        double T_init = 1.0 + 25.0 * (1.0 - std::exp(-(r - 1.0) / 0.3)) * std::pow(r, -0.25);
        double p_init = P.theta_0 * rho_init * T_init;

        double wp_init = P.w_plus0 * std::pow(A_cell[i], -1.2);
        double wm_init = 0.02 * wp_init * (1.0 - std::exp(-(r - 1.0) / 0.5));

        W[i] = {rho_init, u_init, p_init, wp_init, wm_init};
        U[i] = prim_to_cons(W[i], A_cell[i], P.gamma);
    }

    // --------------------------------------------------------------------------
    // Граничные условия
    // --------------------------------------------------------------------------
    auto apply_boundary_conditions = [&](std::vector<Prim>& W_arr, std::vector<Cons>& U_arr) {
        double u_in  = std::max(W_arr[ng].u, 1.0e-5);
        double wm_in = std::max(W_arr[ng].wm, 1.0e-12);

        for (int g = 0; g < ng; ++g) {
            W_arr[g].rho = 1.0;
            W_arr[g].p   = P.theta_0;
            W_arr[g].wp  = P.w_plus0;
            W_arr[g].u   = u_in;
            W_arr[g].wm  = wm_in;
            U_arr[g]     = prim_to_cons(W_arr[g], A_cell[g], P.gamma);
        }

        for (int g = ng + N; g < N_tot; ++g) {
            W_arr[g] = W_arr[ng + N - 1];
            U_arr[g] = prim_to_cons(W_arr[g], A_cell[g], P.gamma);
        }
    };

    // --------------------------------------------------------------------------
    // Пространственный гиперболический оператор (MUSCL + HLL/Rusanov)
    // --------------------------------------------------------------------------
    auto compute_spatial_rhs = [&](const std::vector<Prim>& W_arr, std::vector<Cons>& rhs_out) {
        #pragma omp parallel for schedule(static)
        for (int f = ng; f <= ng + N; ++f) {
            int il = f - 1;
            int ir = f;

            Prim WL, WR;
            if (f == ng) {
                WL.rho = 1.0;
                WL.p   = P.theta_0;
                WL.wp  = P.w_plus0;
                WL.u   = std::max(W_arr[ng].u, 1.0e-5);
                WL.wm  = std::max(W_arr[ng].wm, 1.0e-12);
                WR     = W_arr[ng];
            } else {
                auto reconstruct = [&](int c, double sign_dir) -> Prim {
                    Prim Wc = W_arr[c];
                    Prim Wm = W_arr[c - 1];
                    Prim Wp = W_arr[c + 1];

                    double dx_m = r_cell[c] - r_cell[c - 1];
                    double dx_p = r_cell[c + 1] - r_cell[c];
                    double half_dx = 0.5 * dr[c] * sign_dir;

                    Prim W_rec;
                    W_rec.rho = Wc.rho + half_dx * apply_limiter((Wc.rho - Wm.rho)/dx_m, (Wp.rho - Wc.rho)/dx_p, P.limiter);
                    W_rec.u   = Wc.u   + half_dx * apply_limiter((Wc.u   - Wm.u  )/dx_m, (Wp.u   - Wc.u  )/dx_p, P.limiter);
                    W_rec.p   = Wc.p   + half_dx * apply_limiter((Wc.p   - Wm.p  )/dx_m, (Wp.p   - Wc.p  )/dx_p, P.limiter);
                    W_rec.wp  = Wc.wp  + half_dx * apply_limiter((Wc.wp  - Wm.wp )/dx_m, (Wp.wp  - Wc.wp )/dx_p, P.limiter);
                    W_rec.wm  = Wc.wm  + half_dx * apply_limiter((Wc.wm  - Wm.wm )/dx_m, (Wp.wm  - Wc.wm )/dx_p, P.limiter);

                    W_rec.rho = std::max(W_rec.rho, 1.0e-12);
                    W_rec.p   = std::max(W_rec.p,   1.0e-12);
                    W_rec.wp  = std::max(W_rec.wp,  0.0);
                    W_rec.wm  = std::max(W_rec.wm,  1.0e-15);
                    return W_rec;
                };

                WL = reconstruct(il, +1.0);
                WR = reconstruct(ir, -1.0);
            }

            double Af = A_face[f];
            Cons UL = prim_to_cons(WL, Af, P.gamma);
            Cons UR = prim_to_cons(WR, Af, P.gamma);
            Cons FL = compute_flux(WL, Af, P);
            Cons FR = compute_flux(WR, Af, P);

            double sl_min, sl_max, sr_min, sr_max;
            wave_speeds(WL, Af, P, sl_min, sl_max);
            wave_speeds(WR, Af, P, sr_min, sr_max);

            double sL = std::min(sl_min, sr_min);
            double sR = std::max(sl_max, sr_max);

            Cons F_num;
            if (P.flux_scheme == "Rusanov" || P.flux_scheme == "rusanov") {
                double s_max = std::max(std::abs(sL), std::abs(sR));
                F_num.U1 = 0.5 * (FL.U1 + FR.U1 - s_max * (UR.U1 - UL.U1));
                F_num.U2 = 0.5 * (FL.U2 + FR.U2 - s_max * (UR.U2 - UL.U2));
                F_num.U3 = 0.5 * (FL.U3 + FR.U3 - s_max * (UR.U3 - UL.U3));
                F_num.U4 = 0.5 * (FL.U4 + FR.U4 - s_max * (UR.U4 - UL.U4));
                F_num.U5 = 0.5 * (FL.U5 + FR.U5 - s_max * (UR.U5 - UL.U5));
                Au_face[f] = 0.5 * Af * (WL.u + WR.u);
            } else {
                if (sL >= 0.0) {
                    F_num = FL;
                    Au_face[f] = Af * WL.u;
                } else if (sR <= 0.0) {
                    F_num = FR;
                    Au_face[f] = Af * WR.u;
                } else {
                    double inv_ds = 1.0 / (sR - sL);
                    F_num.U1 = (sR * FL.U1 - sL * FR.U1 + sL * sR * (UR.U1 - UL.U1)) * inv_ds;
                    F_num.U2 = (sR * FL.U2 - sL * FR.U2 + sL * sR * (UR.U2 - UL.U2)) * inv_ds;
                    F_num.U3 = (sR * FL.U3 - sL * FR.U3 + sL * sR * (UR.U3 - UL.U3)) * inv_ds;
                    F_num.U4 = (sR * FL.U4 - sL * FR.U4 + sL * sR * (UR.U4 - UL.U4)) * inv_ds;
                    F_num.U5 = (sR * FL.U5 - sL * FR.U5 + sL * sR * (UR.U5 - UL.U5)) * inv_ds;
                    Au_face[f] = (sR * Af * WL.u - sL * Af * WR.u) * inv_ds;
                }
            }
            F_face[f] = F_num;
        }

        #pragma omp parallel for schedule(static)
        for (int i = ng; i < ng + N; ++i) {
            double inv_dr = 1.0 / dr[i];
            double r  = r_cell[i];
            double A  = A_cell[i];
            double dA = dAdr_cell[i];

            const Prim& Wi = W_arr[i];
            double PA = 0.5 * (Wi.wp + Wi.wm);
            double div_Au = (Au_face[i + 1] - Au_face[i]) * inv_dr;

            rhs_out[i].U1 = -(F_face[i + 1].U1 - F_face[i].U1) * inv_dr;
            rhs_out[i].U2 = -(F_face[i + 1].U2 - F_face[i].U2) * inv_dr 
                            + (Wi.p + PA) * dA - A * Wi.rho / (r * r);
            rhs_out[i].U3 = -(F_face[i + 1].U3 - F_face[i].U3) * inv_dr 
                            - A * Wi.rho * Wi.u / (r * r);
            rhs_out[i].U4 = -(F_face[i + 1].U4 - F_face[i].U4) * inv_dr 
                            - 0.5 * Wi.wp * div_Au;
            rhs_out[i].U5 = -(F_face[i + 1].U5 - F_face[i].U5) * inv_dr 
                            - 0.5 * Wi.wm * div_Au;
        }
    };

    // --------------------------------------------------------------------------
    // Локальный полунеявный (IMEX) шаг для волновых источников R и Gamma_+-
    // --------------------------------------------------------------------------
    auto apply_wave_sources_imex = [&](std::vector<Prim>& W_arr, std::vector<Cons>& U_arr, double dt_step) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N_tot; ++i) {
            VA_cell[i] = P.V_A0 / (A_cell[i] * std::sqrt(W_arr[i].rho));
        }

        #pragma omp parallel for schedule(static)
        for (int i = ng; i < ng + N; ++i) {
            double A   = A_cell[i];
            double rho = W_arr[i].rho;
            double wp  = std::max(W_arr[i].wp, 0.0);
            double wm  = std::max(W_arr[i].wm, 1.0e-16);

            double dVA_dr = (VA_cell[i + 1] - VA_cell[i - 1]) / (r_cell[i + 1] - r_cell[i - 1]);
            double grad_VA = std::abs(dVA_dr);
            if (P.smooth_reflection == 1) {
                grad_VA = std::sqrt(dVA_dr * dVA_dr + P.eps_reg * P.eps_reg);
            }

            double k_d = P.alpha_d / std::sqrt(A * rho);
            double Gamma_p = k_d * std::sqrt(wm);
            double Gamma_m = k_d * std::sqrt(wp);

            double Gamma_max;
            if (P.smooth_reflection == 1) {
                Gamma_max = 0.5 * (Gamma_p + Gamma_m + std::sqrt((Gamma_p - Gamma_m)*(Gamma_p - Gamma_m) + P.eps_reg*P.eps_reg));
            } else {
                Gamma_max = std::max(Gamma_p, Gamma_m);
            }

            double C_imb = compute_C_imbalance(wp, wm);
            double R_rate;
            if (P.smooth_reflection == 1) {
                R_rate = (grad_VA * Gamma_max) / std::sqrt(grad_VA * grad_VA + Gamma_max * Gamma_max + 1.0e-30);
            } else {
                R_rate = std::min(grad_VA, Gamma_max);
            }
            double R_ref = C_imb * R_rate;

            R_ref_cell[i] = R_ref;
            Qw_cell[i]    = Gamma_p * wp + Gamma_m * wm;

            double ap = std::sqrt(wp);
            double am = std::sqrt(wm);

            double a11 = 1.0;
            double a12 = 0.5 * dt_step * (R_ref + k_d * ap);
            double a21 = -0.5 * dt_step * R_ref;
            double a22 = 1.0 + 0.5 * dt_step * k_d * ap;

            double det = a11 * a22 - a12 * a21;
            double ap_new = ap, am_new = am;
            if (det > 1.0e-15) {
                ap_new = (a22 * ap - a12 * am) / det;
                am_new = (-a21 * ap + a11 * am) / det;
            }
            ap_new = std::max(ap_new, 0.0);
            am_new = std::max(am_new, 1.0e-8);

            W_arr[i].wp = ap_new * ap_new;
            W_arr[i].wm = am_new * am_new;

            U_arr[i].U4 = A * W_arr[i].wp;
            U_arr[i].U5 = A * W_arr[i].wm;

            W_arr[i] = cons_to_prim(U_arr[i], A, P.gamma);
        }
    };

    // --------------------------------------------------------------------------
    // Неявный шаг теплопроводности Спитцера и радиационных потерь (прогонка Томаса)
    // --------------------------------------------------------------------------
    auto apply_thermal_tr_implicit = [&](std::vector<Prim>& W_arr, std::vector<Cons>& U_arr, double dt_step) {
        if (P.enable_tr == 0) return;

        const double cv_fac = P.theta_0 / (P.gamma - 1.0);

        // Коэффициент модифицированной теплопроводности K_{f} = A_f * f_c(r_f) * chi_0 * max(T_f, tau_mod)^2.5
        auto get_K_face = [&](int f) -> double {
            double rf = r_face[f];
            double Af = A_face[f];
            double fc = 1.0 / (1.0 + (rf / P.r_col) * (rf / P.r_col));
            double TL = W_arr[f - 1].p / (P.theta_0 * W_arr[f - 1].rho);
            double TR = W_arr[f].p     / (P.theta_0 * W_arr[f].rho);
            double Tf = 0.5 * (TL + TR);
            double T_eff = std::max(Tf, P.tau_mod);
            return Af * fc * P.chi_0 * std::pow(T_eff, 2.5);
        };

        // Прямой ход прогонки Томаса по внутренним ячейкам k = 0 .. N-1 (глобальный индекс i = ng + k)
        for (int k = 0; k < N; ++k) {
            int i = ng + k;
            double rho_i = W_arr[i].rho;
            double T_old = std::max(W_arr[i].p / (P.theta_0 * rho_i), 0.5);

            double KL = get_K_face(i);
            double KR = get_K_face(i + 1);

            double dist_L = (k == 0) ? (r_cell[i] - r_face[ng]) : (r_cell[i] - r_cell[i - 1]);
            double dist_R = (k == N - 1) ? (r_face[ng + N] - r_cell[i]) : (r_cell[i + 1] - r_cell[i]);

            double a_k = dt_step * KL / (dr[i] * dist_L);
            double c_k = dt_step * KR / (dr[i] * dist_R);

            // Полунеявная аппроксимация радиационных потерь Q_rad = (Q_rad_old / T_old) * T_new
            double Q_rad_old = compute_Qrad_modified(rho_i, T_old, P);
            double rad_impl  = dt_step * A_cell[i] * (Q_rad_old / T_old);

            double m_cap = cv_fac * A_cell[i] * rho_i;
            double b_k   = m_cap + a_k + c_k + rad_impl;
            double d_k   = m_cap * T_old;

            if (k == 0) {
                // Левая граница r = 1: условие Дирихле T(1) = 1.0
                d_k += a_k * 1.0;
                a_k = 0.0;
            }
            if (k == N - 1) {
                // Правая граница r = r_out: условие Неймана dT/dr = 0 (T_N = T_{N-1})
                b_k -= c_k;
                c_k = 0.0;
            }

            if (k == 0) {
                c_star[k] = c_k / b_k;
                d_star[k] = d_k / b_k;
            } else {
                double denom = b_k - a_k * c_star[k - 1];
                c_star[k] = c_k / denom;
                d_star[k] = (d_k + a_k * d_star[k - 1]) / denom;
            }
        }

        // Обратный ход прогонки Томаса
        T_new[N - 1] = std::max(d_star[N - 1], 0.5);
        for (int k = N - 2; k >= 0; --k) {
            T_new[k] = std::max(d_star[k] + c_star[k] * T_new[k + 1], 0.5);
        }

        // Обновление давления и полной энергии в ячейках
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < N; ++k) {
            int i = ng + k;
            W_arr[i].p = P.theta_0 * W_arr[i].rho * T_new[k];
            U_arr[i]   = prim_to_cons(W_arr[i], A_cell[i], P.gamma);
        }
    };

    // --------------------------------------------------------------------------
    // Основной цикл установления по времени (TVD Runge-Kutta 2 + IMEX + Томас)
    // --------------------------------------------------------------------------
    apply_boundary_conditions(W, U);

    std::ofstream hist_file(P.output_history);
    hist_file << "step,time,dt,res_rho,res_u,u_in,u_out,T_max\n";

    double t = 0.0;
    int step = 0;
    double res_u = 1.0, res_rho = 1.0;

    while (t < P.t_end && step < P.max_steps) {
        double max_rate = 1.0e-10;
        #pragma omp parallel for reduction(max:max_rate) schedule(static)
        for (int i = ng; i < ng + N; ++i) {
            double s_min, s_max;
            wave_speeds(W[i], A_cell[i], P, s_min, s_max);
            double local_s = std::max(std::abs(s_min), std::abs(s_max));
            max_rate = std::max(max_rate, local_s / dr[i]);
        }

        double dt = P.cfl / max_rate;
        if (t + dt > P.t_end) dt = P.t_end - t;

        std::vector<double> u_old(N), rho_old(N);
        #pragma omp parallel for schedule(static)
        for (int i = ng; i < ng + N; ++i) {
            U0[i] = U[i];
            u_old[i - ng]   = W[i].u;
            rho_old[i - ng] = W[i].rho;
        }

        // --- Стадия 1 RK2 (Предиктор) ---
        compute_spatial_rhs(W, RHS);
        #pragma omp parallel for schedule(static)
        for (int i = ng; i < ng + N; ++i) {
            U[i].U1 = U0[i].U1 + dt * RHS[i].U1;
            U[i].U2 = U0[i].U2 + dt * RHS[i].U2;
            U[i].U3 = U0[i].U3 + dt * RHS[i].U3;
            U[i].U4 = U0[i].U4 + dt * RHS[i].U4;
            U[i].U5 = U0[i].U5 + dt * RHS[i].U5;
            W[i] = cons_to_prim(U[i], A_cell[i], P.gamma);
        }
        apply_wave_sources_imex(W, U, dt);
        apply_boundary_conditions(W, U);

        // --- Стадия 2 RK2 (Корректор) ---
        compute_spatial_rhs(W, RHS);
        #pragma omp parallel for schedule(static)
        for (int i = ng; i < ng + N; ++i) {
            U[i].U1 = 0.5 * (U0[i].U1 + U[i].U1 + dt * RHS[i].U1);
            U[i].U2 = 0.5 * (U0[i].U2 + U[i].U2 + dt * RHS[i].U2);
            U[i].U3 = 0.5 * (U0[i].U3 + U[i].U3 + dt * RHS[i].U3);
            U[i].U4 = 0.5 * (U0[i].U4 + U[i].U4 + dt * RHS[i].U4);
            U[i].U5 = 0.5 * (U0[i].U5 + U[i].U5 + dt * RHS[i].U5);
            W[i] = cons_to_prim(U[i], A_cell[i], P.gamma);
        }
        apply_wave_sources_imex(W, U, 0.5 * dt);

        // --- Стадия 3: Неявный шаг электронной теплопроводности и радиационных потерь ---
        apply_thermal_tr_implicit(W, U, dt);
        apply_boundary_conditions(W, U);

        t += dt;
        step++;

        if (step % 200 == 0 || step == 1 || t >= P.t_end) {
            double sum_du2 = 0.0, sum_drho2 = 0.0, T_max = 0.0;
            #pragma omp parallel for reduction(+:sum_du2,sum_drho2) reduction(max:T_max) schedule(static)
            for (int i = ng; i < ng + N; ++i) {
                double du   = (W[i].u - u_old[i - ng]) / dt;
                double drho = (W[i].rho - rho_old[i - ng]) / (dt * W[i].rho);
                sum_du2   += du * du;
                sum_drho2 += drho * drho;
                double Ti = W[i].p / (P.theta_0 * W[i].rho);
                T_max = std::max(T_max, Ti);
            }
            res_u   = std::sqrt(sum_du2 / N);
            res_rho = std::sqrt(sum_drho2 / N);

            hist_file << step << "," << t << "," << dt << "," 
                      << res_rho << "," << res_u << "," 
                      << W[ng].u << "," << W[ng + N - 1].u << "," << T_max << "\n";

            if (step % P.output_every == 0 || step == 1 || t >= P.t_end) {
                std::cout << "  Шаг " << std::setw(7) << step 
                          << " | t = " << std::fixed << std::setprecision(3) << std::setw(7) << t
                          << " | res(u) = " << std::scientific << std::setprecision(2) << res_u
                          << " | u(1) = " << std::fixed << std::setprecision(4) << W[ng].u
                          << " | u(out) = " << std::setprecision(3) << W[ng + N - 1].u
                          << " | T_max/T_0 = " << std::setprecision(1) << T_max << "\n";
            }

            if (step > 2000 && res_u < P.tol_steady && res_rho < P.tol_steady) {
                std::cout << "\n[Успех] Достигнут критерий стационарности tol = " << P.tol_steady 
                          << " на шаге " << step << " (t = " << t << ").\n";
                break;
            }
        }
    }
    hist_file.close();

    // --------------------------------------------------------------------------
    // Сохранение итогового пространственного профиля в CSV
    // --------------------------------------------------------------------------
    std::ofstream prof_file(P.output_profile);
    prof_file << std::scientific << std::setprecision(8);
    prof_file << "r,A,dAdr,rho,u,p,T,wp,wm,VA,cs,R_ref,Gamma_p,Gamma_m,Qw,Qrad,q_cond,mass_flux,Bernoulli,"
              << "r_Rsun,u_kms,VA_kms,cs_kms,T_K,rho_SI,B_Gauss\n";

    double int_A_Qrad = 0.0;
    for (int i = ng; i < ng + N; ++i) {
        double r   = r_cell[i];
        double A   = A_cell[i];
        double dA  = dAdr_cell[i];
        double rho = W[i].rho;
        double u   = W[i].u;
        double p   = W[i].p;
        double T   = p / (P.theta_0 * rho);
        double wp  = W[i].wp;
        double wm  = W[i].wm;
        double VA  = P.V_A0 / (A * std::sqrt(rho));
        double cs  = std::sqrt(P.gamma * p / rho);

        double k_d     = P.alpha_d / std::sqrt(A * rho);
        double Gamma_p = k_d * std::sqrt(wm);
        double Gamma_m = k_d * std::sqrt(wp);
        double Qw      = Gamma_p * wp + Gamma_m * wm;
        double R_ref   = R_ref_cell[i];

        // Тепловой поток Спитцера и радиационные потери
        double T_prev = W[i - 1].p / (P.theta_0 * W[i - 1].rho);
        double T_next = W[i + 1].p / (P.theta_0 * W[i + 1].rho);
        double dT_dr  = (T_next - T_prev) / (r_cell[i + 1] - r_cell[i - 1]);
        double fc     = 1.0 / (1.0 + (r / P.r_col) * (r / P.r_col));
        double q_cond = (P.enable_tr) ? (-fc * P.chi_0 * std::pow(std::max(T, P.tau_mod), 2.5) * dT_dr) : 0.0;
        double Qrad   = compute_Qrad_modified(rho, T, P);

        int_A_Qrad += A * Qrad * dr[i];

        double mass_flux = A * rho * u;
        double Bernoulli = 0.5 * u * u - 1.0 / r + (P.gamma / (P.gamma - 1.0)) * (p / rho)
                         + 1.5 * (wp + wm) / rho + VA * (wp - wm) / (rho * u)
                         + (A * q_cond + int_A_Qrad) / std::max(mass_flux, 1.0e-12);

        double u_kms   = u * P.V_star * 1.0e-3;
        double VA_kms  = VA * P.V_star * 1.0e-3;
        double cs_kms  = cs * P.V_star * 1.0e-3;
        double T_K     = T * P.T_star;
        double rho_SI  = rho * P.rho_star;
        double B_Gauss = (P.B_star / A) * 1.0e4;

        prof_file << r << "," << A << "," << dA << ","
                  << rho << "," << u << "," << p << "," << T << ","
                  << wp << "," << wm << "," << VA << "," << cs << ","
                  << R_ref << "," << Gamma_p << "," << Gamma_m << "," << Qw << ","
                  << Qrad << "," << q_cond << "," << mass_flux << "," << Bernoulli << ","
                  << r << "," << u_kms << "," << VA_kms << "," << cs_kms << ","
                  << T_K << "," << rho_SI << "," << B_Gauss << "\n";
    }
    prof_file.close();

    std::cout << "Итоговый профиль сохранён в файл: " << P.output_profile << "\n";
    std::cout << "История установления сохранена в: " << P.output_history << "\n";
    return 0;
}