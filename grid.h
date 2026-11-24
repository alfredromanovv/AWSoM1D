#pragma once

#include "include.h"

struct State;

// Primitive variables: density, velocity, gas pressure, wave energy densities.
struct Primitive {
    double rho = 0.0;
    double u = 0.0;
    double p = 0.0;
    double w_plus = 0.0;
    double w_minus = 0.0;

    double temperature() const { return p / (THETA * rho); }
    bool valid() const;
    State to_conservative() const;
};

// Densities per unit volume; energy includes both wave populations.
struct State {
    double rho = 0.0;
    double momentum = 0.0;
    double energy = 0.0;
    double w_plus = 0.0;
    double w_minus = 0.0;

    double velocity() const { return momentum / rho; }
    double pressure() const;
    double temperature() const { return pressure() / (THETA * rho); }
    bool valid() const;
    Primitive to_primitive() const;
    // Convenience overload accepting temperature instead of pressure.
    static State from_primitive(double rho, double u, double temperature,
                                double w_plus, double w_minus);
};

struct Face {
    double r = 0.0;
    double area = 0.0; // A(r) = r^2.
};

struct Cell {
    double r = 0.0;     // Arithmetic midpoint.
    double width = 0.0;
    double area = 0.0;  // A at the midpoint.
    double volume = 0.0; // Exact integral of A(r) over the cell.
    State state;
};

class Grid {
public:
    // Physical cells: [begin(), end()); ghosts are outside this interval.
    // Cell i is bounded by faces i and i+1. Rebuilding zeros ALL states, including ghosts.
    void create(int count = NX, int ghosts = NG, double r_min = R_MIN,
                double r_max = R_MAX, double ratio = GRID_RATIO);
    int begin() const { return ghosts_; }
    int end() const { return ghosts_ + count_; }
    int size() const { return static_cast<int>(cells_.size()); }
    const Cell& cell(int i) const { return cells_.at(i); }
    const Face& face(int i) const { return faces_.at(i); }
    void set_state(int i, const State& state);
    void fill(const State& state); // Includes ghost cells.
    void clear_states(); // Zero storage; initial conditions must follow before conversion/use.

private:
    int count_ = 0;
    int ghosts_ = 0;
    vector<Cell> cells_;
    vector<Face> faces_;
};

extern Grid g_grid;
