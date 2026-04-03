#!/usr/bin/env python3
"""
Optimal Control — SIDARTHE Model (Realistic Economic Cost)
===========================================================
Applies the Pontryagin Maximum Principle to the 8-compartment SIDARTHE
model using a cost function grounded in Brazil 2020 data.

Cost functional (R$):
    J = integral_0^T [c_T_total*T + c_G*u1 + (r1/2)*u1^2
                                  + c_V*u2 + (r2/2)*u2^2] dt
      + p_T_terminal * T(T)

where:
    c_T_total    = VSL*tau + c_T_icu  [R$/ICU-patient/day]
    c_G          = GDP_daily * alpha_GDP
    p_T_terminal = VSL*tau / (nu + tau)   [expected future deaths per T patient]

Advantage over SIR/SEIR: T(t) and E(t) are explicit model states,
so no approximation via IFR is needed — deaths are directly tracked.

Parameters loaded from: config/realistic_params.yaml

Usage:
    python scripts/optimal_control_pontryagin_sidarthe.py
"""

import yaml
import json
import numpy as np
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.sidarthe import SIDARTHEControlledModel
from src.control.cost_functions import RealisticCostSIDARTHE
from src.control.pontryagin import GenericPontryaginSolver
from src.plots.control_plots import (
    plot_control_trajectories_generic,
    plot_compartments_generic,
    plot_costates_generic,
)

MODEL_NAME = 'SIDARTHE'
FIGURES_DIR = ROOT_DIR / 'results' / 'figures' / 'control' / 'pontryagin_sidarthe'
RESULTS_DIR = ROOT_DIR / 'results' / 'control' / 'pontryagin_sidarthe'

# SIDARTHE state labels (8 compartments split into 2 figures of 4)
STATE_LABELS = {
    'S':     'Susceptible S(t)',
    'I':     'Infected asymptomatic undetected I(t)',
    'D':     'Diagnosed asymptomatic D(t)',
    'A':     'Ailing symptomatic undetected A(t)',
    'Rs':    'Recognized symptomatic detected Rs(t)',
    'T':     'Threatened — severe/ICU T(t)',
    'H':     'Healed H(t)',
    'E_ext': 'Extinct (deaths) E(t)',
}

# Co-state labels (p_H = p_E_ext = 0 for all t)
COSTATE_LABELS = {
    'p_S':     'p_S(t)  [R$/person]',
    'p_I':     'p_I(t)  [R$/person]',
    'p_D':     'p_D(t)  [R$/person]',
    'p_A':     'p_A(t)  [R$/person]',
    'p_Rs':    'p_Rs(t)  [R$/person]',
    'p_T':     'p_T(t)  [R$/ICU-patient]',
    'p_H':     'p_H(t) ≡ 0',
    'p_E_ext': 'p_E(t) ≡ 0',
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
    sid_cfg = cfg['sidarthe']
    econ = cfg['economic']
    reg = cfg['regularization']
    sol_cfg = cfg['solver']
    fig_cfg = cfg['output']['figures']

    print("=" * 70)
    print("SIDARTHE OPTIMAL CONTROL — REALISTIC ECONOMIC COST")
    print("Pontryagin Maximum Principle | Shooting Method")
    print("=" * 70)

    # --- Model parameters ---
    mp = sid_cfg['model']
    N = mp['N']
    params = {
        'alpha':    mp['alpha'],
        'beta':     mp['beta'],
        'epsilon':  mp['epsilon'],
        'zeta':     mp['zeta'],
        'eta':      mp['eta'],
        'lambda_r': mp['lambda_r'],
        'mu':       mp['mu'],
        'rho':      mp['rho'],
        'theta':    mp['theta'],
        'kappa':    mp['kappa'],
        'nu':       mp['nu'],
        'tau':      mp['tau'],
    }

    print(f"\nModel parameters (Brazil 2020 identification + Giordano et al. 2020):")
    for k, v in params.items():
        print(f"  {k:<10} = {v}")
    print(f"  R0         = {mp['R0']}")
    print(f"  N          = {N:,.0f}")

    # --- Initial conditions ---
    ic = sid_cfg['initial_conditions']
    x0 = np.array([
        ic['S0'], ic['I0'], ic['D0'], ic['A0'],
        ic['R0_ic'], ic['T0'], ic['H0'], ic['E0_ic']
    ])
    print(f"\nInitial conditions:")
    for k, v in zip(['S', 'I', 'D', 'A', 'Rs', 'T', 'H', 'E'], x0):
        print(f"  {k}0 = {v:>14,.0f}")

    # --- Cost function ---
    cost = RealisticCostSIDARTHE(
        alpha=params['alpha'], beta=params['beta'],
        epsilon=params['epsilon'], zeta=params['zeta'], eta=params['eta'],
        lambda_r=params['lambda_r'], mu=params['mu'], rho=params['rho'],
        theta=params['theta'], kappa=params['kappa'],
        nu=params['nu'], tau=params['tau'],
        N=N,
        VSL=econ['VSL'], c_T_icu=econ['c_T_icu'],
        GDP_daily=econ['GDP_daily'], alpha_GDP=econ['alpha_GDP'],
        c_vaccine=econ['c_vaccine'],
        r1=reg['r1_lockdown'], r2=reg['r2_vaccination'],
    )

    print(f"\nCost function: {cost}")
    print(f"  c_T_total    = {cost.c_T_total:.2e}  R$/ICU-patient/day")
    print(f"     = VSL*tau ({econ['VSL']:.0e} x {params['tau']:.4f})"
          f" + c_T_icu ({econ['c_T_icu']:.0f})")
    print(f"  c_G          = {cost.c_G:.2e}  R$/day  (full lockdown cost)")
    print(f"  c_V          = {cost.c_V}  R$/person vaccinated")
    print(f"  p_T_terminal = {cost.p_T_terminal:.2e}  R$/T-patient at T")

    # --- Model ---
    model = SIDARTHEControlledModel(params, N)

    # --- Control bounds ---
    ctrl = sid_cfg['control']
    u_bounds = {
        'u1': (ctrl['u1_lockdown']['min'], ctrl['u1_lockdown']['max']),
        'u2': (ctrl['u2_vaccination']['min'], ctrl['u2_vaccination']['max']),
    }
    print(f"\nControl bounds: u1 in {u_bounds['u1']}, u2 in {u_bounds['u2']} persons/day")

    # --- Solver ---
    T = float(sol_cfg['horizon_days'])
    solver = GenericPontryaginSolver(model, cost, u_bounds)

    print(f"\nSolving over T = {T:.0f} days ...")
    print("(8-state SIDARTHE: using Forward-Backward Sweep for numerical stability)")
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
    print(f"  Peak T (controlled)  : {np.max(solution['T']):,.0f}  (ICU)")
    peak_nc = np.max(nc['T'])
    print(f"  Peak T (no control)  : {peak_nc:,.0f}  (ICU)")
    if peak_nc > 0:
        peak_red = (peak_nc - np.max(solution['T'])) / peak_nc * 100
        print(f"  Peak T reduction     : {peak_red:.1f}%")
    total_deaths = solution['E_ext'][-1]
    total_deaths_nc = nc['E'][-1]
    print(f"  Total deaths (ctrl)  : {total_deaths:,.0f}")
    print(f"  Total deaths (no ctrl): {total_deaths_nc:,.0f}")

    # --- Save JSON ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(RESULTS_DIR / 'trajectories.json', {
        't':     solution['t'].tolist(),
        'u1':    solution['u1'].tolist(),
        'u2':    solution['u2'].tolist(),
        'S':     solution['S'].tolist(),
        'I':     solution['I'].tolist(),
        'D':     solution['D'].tolist(),
        'A':     solution['A'].tolist(),
        'Rs':    solution['Rs'].tolist(),
        'T':     solution['T'].tolist(),
        'H':     solution['H'].tolist(),
        'E_ext': solution['E_ext'].tolist(),
        'cost_total': float(solution['cost']),
        'model': 'SIDARTHE', 'cost_type': 'realistic_economic',
        'c_T_total':    float(cost.c_T_total),
        'c_G':          float(cost.c_G),
        'c_V':          float(cost.c_V),
        'p_T_terminal': float(cost.p_T_terminal),
    })
    print(f"\nResults saved to {RESULTS_DIR}")

    # --- Plots ---
    if cfg['output']['save_figures']:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        dpi = fig_cfg['dpi']
        fmt = fig_cfg['format']
        fsize = tuple(fig_cfg['figsize'])

        # Control trajectories (1 figure, 2 subplots)
        plot_control_trajectories_generic(
            solution, FIGURES_DIR, MODEL_NAME, dpi=dpi, fmt=fmt, figsize=fsize)

        # Compartments: 8 states → 2 figures (4+4)
        # Map solution keys to plot-friendly names
        sol_plot = dict(solution)
        nc_plot = {
            't': nc['t'], 'S': nc['S'], 'I': nc['I'], 'D': nc['D'],
            'A': nc['A'], 'Rs': nc['Rs'], 'T': nc['T'],
            'H': nc['H'], 'E_ext': nc['E']
        }
        plot_compartments_generic(
            sol_plot, nc_plot,
            state_keys=['S', 'I', 'D', 'A', 'Rs', 'T', 'H', 'E_ext'],
            state_labels=STATE_LABELS,
            save_dir=FIGURES_DIR, model_name=MODEL_NAME,
            dpi=dpi, fmt=fmt, figsize=fsize)

        # Co-states: 8 → 2 figures (4+4)
        plot_costates_generic(
            solution,
            costate_keys=['p_S', 'p_I', 'p_D', 'p_A', 'p_Rs', 'p_T', 'p_H', 'p_E_ext'],
            costate_labels=COSTATE_LABELS,
            save_dir=FIGURES_DIR, model_name=MODEL_NAME,
            dpi=dpi, fmt=fmt, figsize=fsize)

        print(f"Figures saved to {FIGURES_DIR}")
        print(f"  - {MODEL_NAME.lower()}_control_trajectories.{fmt}       (1 fig, 2 subplots)")
        print(f"  - {MODEL_NAME.lower()}_compartments_part1.{fmt}         (4 compartments)")
        print(f"  - {MODEL_NAME.lower()}_compartments_part2.{fmt}         (4 compartments)")
        print(f"  - {MODEL_NAME.lower()}_costates_part1.{fmt}             (4 co-states)")
        print(f"  - {MODEL_NAME.lower()}_costates_part2.{fmt}             (4 co-states)")

    print("\nDone.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
