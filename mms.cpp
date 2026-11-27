#include "mms.h"
#include "functions.h"
#include <filesystem>

namespace mms {
using Values = array<double, 5>;

static Primitive exact(double r, double t)
{
    const double a = r - 1 - t, b = r - 1 + t;
    return {1 + 0.1*sin(a), 0.3 + 0.05*cos(a), 0.2 + 0.03*sin(b),
            0.09 + 0.01*cos(b), 0.01 + 0.002*sin(a)};
}

Values source(double r, double t)
{
    const auto [rho,u,p,wp,wm] = exact(r,t);
    const double a = r-1-t, b = r-1+t;
    const double rho_r = 0.1*cos(a), u_r = -0.05*sin(a);
    const double p_r = 0.03*cos(b), wp_r = -0.01*sin(b), wm_r = 0.002*cos(a);
    const double div = u_r + 2*u/r, va = B/(r*r*sqrt(rho));
    const double density_gradient = rho_r/(2*rho);
    const double reflection = (sqrt(wp*wm)-2*wm)*va*(2/r + density_gradient);
    const double pressure = p + (wp+wm)/2, pressure_r = p_r + (wp_r+wm_r)/2;
    const double energy = rho*u*u/2 + p/(GAMMA-1) + wp + wm;
    const double kinetic_r = u*u*rho_r/2 + rho*u*u_r;
    const double energy_t = -kinetic_r + p_r/(GAMMA-1) + wp_r - wm_r;
    const double energy_r = kinetic_r + p_r/(GAMMA-1) + wp_r + wm_r;
    const double mass = (u-1)*rho_r + rho*div;
    return {mass, u*mass + rho*(u-1)*u_r + pressure_r + rho/(r*r),
            energy_t + div*(energy+pressure) + u*(energy_r+pressure_r)
                + va*(wp_r-wm_r-(wp-wm)*density_gradient) + rho*u/(r*r),
            (1+u+va)*wp_r + 1.5*wp*div - va*wp*density_gradient + reflection + D/r*sqrt(wm/rho)*wp,
            (u-va-1)*wm_r + 1.5*wm*div + va*wm*density_gradient - reflection + D/r*sqrt(wp/rho)*wm};
}

static Values components(const State& s)
{
    return {s.rho,s.momentum,s.energy,s.w_plus,s.w_minus};
}

// Three-point Gauss-Legendre quadrature of r^2 U divided by cell volume.
static Values average(const Cell& cell, double t)
{
    Values sum{};
    const double offset = sqrt(0.6)*cell.width/2;
    for (int j = -1; j <= 1; ++j) {
        const double r = cell.r + j*offset;
        const auto v = components(exact(r,t).to_conservative());
        const double weight = (j == 0 ? 8.0 : 5.0)*cell.width*r*r/(18*cell.volume);
        for (int k = 0; k < 5; ++k) sum[k] += weight*v[k];
    }
    return sum;
}

static void set_exact(Grid& grid, int i, double time)
{
    auto v = average(grid.cell(i),time);
    grid.set_state(i,{v[0],v[1],v[2],v[3],v[4]});
}

void fill_ghosts(Grid& grid, double time)
{
    for (int j = 0; j < grid.begin(); ++j) {
        set_exact(grid,j,time);
        set_exact(grid,grid.end()+j,time);
    }
}

using Norms = array<Values,3>;
static Norms errors(const Grid& grid, double time)
{
    Norms norms{};
    double volume = 0;
    for (int i=grid.begin(); i<grid.end(); ++i) {
        const auto& cell = grid.cell(i);
        const auto ref = average(cell,time), num = components(cell.state);
        volume += cell.volume;
        for (int k=0; k<5; ++k) {
            const double e = abs(num[k]-ref[k]);
            norms[0][k] += cell.volume*e;
            norms[1][k] += cell.volume*e*e;
            norms[2][k] = max(norms[2][k],e);
        }
    }
    for (int k=0; k<5; ++k) {
        norms[0][k] /= volume;
        norms[1][k] = sqrt(norms[1][k]/volume);
    }
    return norms;
}

void run_study()
{
    static_assert(!MMS_TEST || ENABLE_REFLECTION);
    load_parameters();
    const auto root = filesystem::current_path();
    ofstream report("mms_errors.tsv");
    report.exceptions(ios::failbit | ios::badbit);
    report << setprecision(17) << "mesh\tstep_factor\tN\tq\tt\tcomponent\tL1\tL2\tLinf\tm1\tm2\tminf\n";
    const char* names[] = {"rho","rho_u","E","w_plus","w_minus"};
    for (bool stretched : {false,true}) {
        for (double factor : {1.0,0.5}) {
            Norms previous{};
            for (int n : {100,200,400,800}) {
                const string mesh = stretched ? "stretched" : "uniform";
                const double q = stretched ? pow(1.008,100.0/n) : 1.0;
                const auto dir = root / (mesh+"_"+to_string(n)+(factor==1 ? "_full" : "_half"));
                filesystem::create_directory(dir);
                filesystem::current_path(dir);
                g_grid.create(n,NG,1,4,q);
                g_time = 0;
                g_step = 0;
                for (int i=0; i<g_grid.size(); ++i) set_exact(g_grid,i,0);
                save_initial_results(g_grid,0,0);
                calculate(factor);
                save_final_results(g_grid,g_time,g_step);
                filesystem::current_path(root);
                if (g_time != END_TIME) return;
                const Norms current = errors(g_grid,g_time);
                for (int k=0; k<5; ++k) {
                    report << mesh << '\t' << factor << '\t' << n << '\t' << q << '\t' << g_time << '\t' << names[k];
                    for (const auto& norm : current) report << '\t' << norm[k];
                    for (int j=0; j<3; ++j) {
                        report << '\t';
                        if (n>100 && previous[j][k]>0 && current[j][k]>0)
                            report << log2(previous[j][k]/current[j][k]);
                        else report << "NA";
                    }
                    report << '\n';
                }
                report.flush();
                previous = current;
                if (g_stop.load()) return;
            }
        }
    }
}
}
