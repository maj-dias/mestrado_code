#!/usr/bin/env python3
"""
Optimal Control — SIR Model (Realistic Economic Cost)
======================================================
Applies the Pontryagin Maximum Principle to the SIR model using a
cost function grounded in Brazil 2020 economic and epidemiological data.

Cost functional (R$):
    J = integral_0^T [c_I*I + c_G*u1 + (r1/2)*u1^2
                              + c_V*u2 + (r2/2)*u2^2] dt
      + (VSL*IFR)*I(T)

where:
    c_I = (VSL*IFR + c_H*h) * gamma_SIR  [R$/infected/day]
    c_G = GDP_daily * alpha_GDP           [R$/day/lockdown unit]
    c_V = 75 R$/person vaccinated

Note: c_I is lower than in SEIR because gamma_SIR < gamma_SEIR,
reflecting the longer effective infectious period in the SIR model.
The total cost per infected individual is the same (R$77,500).

Parameters loaded from: config/realistic_params.yaml

Usage:
    python scripts/optimal_control_pontryagin_sir.py
"""

import yaml
import json
import numpy as np
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.sir import SIRControlledModel
from src.control.cost_functions import RealisticCostSIR
from src.control.pontryagin import GenericPontryaginSolver
from src.plots.control_plots import (
    plot_control_trajectories_generic,
    plot_compartments_generic,
    plot_costates_generic,
)

MODEL_NAME = 'SIR'
FIGURES_DIR = ROOT_DIR / 'results' / 'figures' / 'control' / 'pontryagin_sir'
RESULTS_DIR = ROOT_DIR / 'results' / 'control' / 'pontryagin_sir'

STATE_LABELS = {
    'S': 'Susceptible S(t)',
    'I': 'Infectious I(t)',
    'R': 'Removed R(t)',
}
COSTATE_LABELS = {
    'p_S': 'Co-state p_S(t)  [R$/person]',
    'p_I': 'Co-state p_I(t)  [R$/person]',
    'p_R': 'Co-state p_R(t)  [R$/person]  ≡ 0',
}


def load_config():
    cfg_path = ROOT_DIR / 'config' / 'realistic_params.yaml'
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    cfg = load_config()
    sir_cfg = cfg['sir']
    econ = cfg['economic']
    reg = cfg['regularization']
    sol_cfg = cfg['solver']
    fig_cfg = cfg['output']['figures']

    print("=" * 70)
    print("SIR OPTIMAL CONTROL — REALISTIC ECONOMIC COST")
    print("Pontryagin Maximum Principle | Shooting Method")
    print("=" * 70)

    # --- Model parameters ---
    beta  = sir_cfg['model']['beta']
    gamma = sir_cfg['model']['gamma']
    N     = sir_cfg['model']['N']

    print(f"\nModel parameters (literature, Brazil 2020):")
    print(f"  beta  = {beta}   (transmission rate)")
    print(f"  gamma = {gamma}   (recovery rate, 10-day period)")
    print(f"  R0    = {beta/gamma:.2f}")
    print(f"  N     = {N:,.0f}")

    # --- Initial conditions ---
    ic = sir_cfg['initial_conditions']
    x0 = np.array([ic['S0'], ic['I0'], ic['R0_ic']])
    print(f"\nInitial conditions:")
    for k, v in zip(['S', 'I', 'R'], x0):
        print(f"  {k}0 = {v:>14,.0f}")

    # --- Cost function ---
    cost = RealisticCostSIR(
        beta=beta, gamma=gamma, N=N,
        VSL=econ['VSL'], IFR=econ['IFR'],
        h=econ['h'], c_H=econ['c_H'],
        GDP_daily=econ['GDP_daily'], alpha_GDP=econ['alpha_GDP'],
        c_vaccine=econ['c_vaccine'],
        r1=reg['r1_lockdown'], r2=reg['r2_vaccination'],
    )

    print(f"\nCost function: {cost}")
    print(f"  c_I   = {cost.c_I:.2e}  R$/infected/day")
    print(f"  c_G   = {cost.c_G:.2e}  R$/day  (full lockdown cost)")
    print(f"  c_V   = {cost.c_V}  R$/person vaccinated")
    print(f"  phi   = {cost.terminal_coeff:.2e}  R$/infected at T")
    print(f"\n  Note: c_I(SIR)={cost.c_I:.0f} < c_I(SEIR)~10150 because")
    print(f"        gamma_SIR={gamma} vs gamma_SEIR~0.10.")
    print(f"        Total cost/infected = R${cost.c_I/gamma:,.0f} (same for both).")

    # --- Model ---
    model = SIRControlledModel(beta, gamma, N)

    # --- Control bounds ---
    ctrl = sir_cfg['control']
    u_bounds = {
        'u1': (ctrl['u1_lockdown']['min'], ctrl['u1_lockdown']['max']),
        'u2': (ctrl['u2_vaccination']['min'], ctrl['u2_vaccination']['max']),
    }
    print(f"\nControl bounds: u1 in {u_bounds['u1']}, u2 in {u_bounds['u2']} persons/day")

    # --- Solver ---
    T = float(sol_cfg['horizon_days'])
    solver = GenericPontryaginSolver(model, cost, u_bounds)

    print(f"\nSolving over T = {T:.0f} days ...")
    solution = solver.solve(
        x0=x0, T=T,
        method='fbs',
        n_points=sol_cfg['n_time_points'],
        tolerance=sol_cfg['tolerance'],
        max_iterations=sol_cfg['max_iterations'],
    )

    # --- Baseline (no control) ---
    def zero_u(t):
        return np.array([0.0, 0.0])

    nc = model.simulate_with_control(x0, zero_u, (0, T), solution['t'])

    # --- Print summary ---
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total cost J         : {solution['cost']:.4e}  R$")
    print(f"  Mean lockdown u1     : {np.mean(solution['u1'])*100:.1f}%")
    print(f"  Mean vaccination u2  : {np.mean(solution['u2']):,.0f}  persons/day")
    print(f"  Peak I (controlled)  : {np.max(solution['I']):,.0f}")
    peak_nc = np.max(nc['I'])
    print(f"  Peak I (no control)  : {peak_nc:,.0f}")
    peak_red = (peak_nc - np.max(solution['I'])) / peak_nc * 100
    print(f"  Peak reduction       : {peak_red:.1f}%")

    # --- Save JSON ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(RESULTS_DIR / 'trajectories.json', {
        't': solution['t'].tolist(),
        'u1': solution['u1'].tolist(),
        'u2': solution['u2'].tolist(),
        'S': solution['S'].tolist(),
        'I': solution['I'].tolist(),
        'R': solution['R'].tolist(),
        'cost_total': float(solution['cost']),
        'peak_reduction_pct': float(peak_red),
        'model': 'SIR', 'cost_type': 'realistic_economic',
        'c_I': float(cost.c_I), 'c_G': float(cost.c_G),
        'c_V': float(cost.c_V),
    })
    print(f"\nResults saved to {RESULTS_DIR}")

    # --- Plots ---
    if cfg['output']['save_figures']:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        dpi = fig_cfg['dpi']
        fmt = fig_cfg['format']
        fsize = tuple(fig_cfg['figsize'])

        plot_control_trajectories_generic(
            solution, FIGURES_DIR, MODEL_NAME, dpi=dpi, fmt=fmt, figsize=fsize)

        plot_compartments_generic(
            solution, nc,
            state_keys=['S', 'I', 'R'],
            state_labels=STATE_LABELS,
            save_dir=FIGURES_DIR, model_name=MODEL_NAME,
            dpi=dpi, fmt=fmt, figsize=fsize)

        plot_costates_generic(
            solution,
            costate_keys=['p_S', 'p_I', 'p_R'],
            costate_labels=COSTATE_LABELS,
            save_dir=FIGURES_DIR, model_name=MODEL_NAME,
            dpi=dpi, fmt=fmt, figsize=fsize)

        print(f"Figures saved to {FIGURES_DIR}")

    print("\nDone.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
