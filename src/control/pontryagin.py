"""
Solver de Pontryagin para Controle Otimo

Este modulo implementa o Principio do Maximo de Pontryagin (PMP) para resolver
problemas de controle otimo usando o metodo shooting para TPBVP (Two-Point
Boundary Value Problem).

Algoritmo:
1. Chutar condicoes iniciais dos co-estados lambda(0)
2. Integrar forward: estados x(t) e co-estados lambda(t) de 0 a T
3. Verificar se lambda(T) satisfaz condicoes terminais
4. Ajustar lambda(0) usando Newton-Raphson ate convergencia
"""

import numpy as np
from scipy.optimize import root
from scipy.integrate import solve_ivp
from typing import Dict, Tuple
from .cost_functions import QuadraticCost
from ..models.seir_controlled import SEIRControlledModel


class GenericPontryaginSolver:
    """
    Generic Pontryagin Maximum Principle solver for any compartmental model.

    Requires model with:
        model.derivatives(t, x, u) -> np.ndarray

    Requires cost with:
        cost.running_cost(x, u, t)      -> float
        cost.terminal_cost(x)           -> float
        cost.terminal_costates(x)       -> np.ndarray
        cost.costate_derivatives(x,u,p) -> np.ndarray
        cost.optimal_control(x, p)      -> np.ndarray  (before clipping)
    """

    def __init__(self, model, cost, u_bounds: Dict[str, Tuple[float, float]]):
        self.model = model
        self.cost = cost
        self.u_bounds = u_bounds

    def _clip_control(self, u: np.ndarray) -> np.ndarray:
        u1_min, u1_max = self.u_bounds['u1']
        u2_min, u2_max = self.u_bounds['u2']
        return np.array([np.clip(u[0], u1_min, u1_max),
                         np.clip(u[1], u2_min, u2_max)])

    def solve_shooting(
        self,
        x0: np.ndarray,
        T: float,
        n_points: int = 180,
        lam0_guess: np.ndarray = None,
        tolerance: float = 1e-4,
        max_iterations: int = 300
    ) -> Dict:
        print("\n" + "=" * 70)
        print("GENERIC SHOOTING METHOD - PONTRYAGIN MAXIMUM PRINCIPLE")
        print("=" * 70)

        n_states = len(x0)
        t_eval = np.linspace(0, T, n_points)

        if lam0_guess is None:
            lam0_guess = self.cost.terminal_costates(x0)
        print(f"Initial guess p(0): {lam0_guess}")

        iter_count = [0]

        def residual(lam0):
            iter_count[0] += 1
            y0 = np.concatenate([x0, lam0])

            def aug_ode(t, y):
                x = y[:n_states]
                p = y[n_states:]
                u = self._clip_control(self.cost.optimal_control(x, p))
                return np.concatenate([
                    self.model.derivatives(t, x, u),
                    self.cost.costate_derivatives(x, u, p)
                ])

            sol = solve_ivp(aug_ode, [0, T], y0, t_eval=t_eval,
                            method='RK45', rtol=1e-7, atol=1e-9)
            if not sol.success:
                return np.ones(n_states) * 1e10

            p_final = sol.y[n_states:, -1]
            x_final = sol.y[:n_states, -1]
            res = p_final - self.cost.terminal_costates(x_final)
            norm = np.linalg.norm(res)
            if iter_count[0] % 10 == 0 or norm < tolerance:
                print(f"  Iter {iter_count[0]:4d}: ||residual|| = {norm:.3e}")
            return res

        print("\nStarting optimization...")
        result = root(residual, lam0_guess, method='hybr', tol=tolerance,
                      options={'maxfev': max_iterations * 10})

        if not result.success:
            print(f"\nWarning: did not fully converge. {result.message}")
        else:
            print(f"\nConverged in {iter_count[0]} iterations.")

        # Final integration
        lam0_opt = result.x
        y0_opt = np.concatenate([x0, lam0_opt])

        def aug_final(t, y):
            x = y[:n_states]
            p = y[n_states:]
            u = self._clip_control(self.cost.optimal_control(x, p))
            return np.concatenate([
                self.model.derivatives(t, x, u),
                self.cost.costate_derivatives(x, u, p)
            ])

        sol_f = solve_ivp(aug_final, [0, T], y0_opt, t_eval=t_eval,
                          method='RK45', rtol=1e-7, atol=1e-9)

        x_arr = sol_f.y[:n_states]
        p_arr = sol_f.y[n_states:]

        u_arr = np.array([
            self._clip_control(self.cost.optimal_control(x_arr[:, i], p_arr[:, i]))
            for i in range(len(sol_f.t))
        ]).T

        total_cost = self._compute_total_cost(sol_f.t, x_arr, u_arr)
        print(f"\nTOTAL COST J = {total_cost:.4e}  R$")

        solution = {
            't': sol_f.t,
            'u1': u_arr[0],
            'u2': u_arr[1],
            'cost': total_cost,
            'success': result.success,
            'iterations': iter_count[0]
        }

        state_keys = self._state_keys(n_states)
        for k, key in enumerate(state_keys):
            solution[key] = x_arr[k]
        for k, key in enumerate(state_keys):
            solution[f'p_{key}'] = p_arr[k]

        return solution

    @staticmethod
    def _state_keys(n_states: int):
        if n_states == 3:
            return ['S', 'I', 'R']
        if n_states == 4:
            return ['S', 'E', 'I', 'R']
        if n_states == 8:
            return ['S', 'I', 'D', 'A', 'Rs', 'T', 'H', 'E_ext']
        return [f'x{i}' for i in range(n_states)]

    def _compute_total_cost(self, t, x, u) -> float:
        running = [self.cost.running_cost(x[:, i], u[:, i], t[i])
                   for i in range(len(t))]
        return float(np.trapz(running, t) + self.cost.terminal_cost(x[:, -1]))

    def solve_forward_backward_sweep(
        self,
        x0: np.ndarray,
        T: float,
        n_points: int = 180,
        max_iterations: int = 500,
        tolerance: float = 1e-4,
        **_ignored
    ) -> Dict:
        """
        Forward-Backward Sweep (FBS) method for Pontryagin optimality.

        Reference: Lenhart & Workman (2007), Optimal Control Applied to
        Biological Models.

        Algorithm:
        1. Forward-integrate states with current control u(t).
        2. Backward-integrate costates from p(T) = terminal_costates(x(T)).
        3. Update u using Hamiltonian optimality; blend with old u (alpha=0.5).
        4. Repeat until ||u_new - u_old||_inf < tolerance.
        """
        print("\n" + "=" * 70)
        print("FORWARD-BACKWARD SWEEP — PONTRYAGIN MAXIMUM PRINCIPLE")
        print("=" * 70)

        n_states = len(x0)
        t_eval = np.linspace(0, T, n_points)

        # Initialise controls to zero
        u1_arr = np.zeros(n_points)
        u2_arr = np.zeros(n_points)

        # Pre-allocate arrays
        x_arr = np.zeros((n_states, n_points))
        p_arr = np.zeros((n_states, n_points))

        converged = False
        for iteration in range(max_iterations):
            u1_old = u1_arr.copy()
            u2_old = u2_arr.copy()

            # ---- Forward sweep ------------------------------------------------
            def fwd_ode(t, x):
                u1 = float(np.interp(t, t_eval, u1_arr))
                u2 = float(np.interp(t, t_eval, u2_arr))
                u = self._clip_control(np.array([u1, u2]))
                return self.model.derivatives(t, x, u)

            sol_fwd = solve_ivp(fwd_ode, [0, T], x0, t_eval=t_eval,
                                method='LSODA', rtol=1e-6, atol=1.0)
            if not sol_fwd.success:
                print(f"  Warning: forward integration failed at iter {iteration}")
                break
            x_arr = sol_fwd.y  # (n_states, n_points)

            # ---- Backward sweep -----------------------------------------------
            p_T = self.cost.terminal_costates(x_arr[:, -1])

            def bwd_ode(t, p):
                x = np.array([float(np.interp(t, t_eval, x_arr[k]))
                              for k in range(n_states)])
                u1 = float(np.interp(t, t_eval, u1_arr))
                u2 = float(np.interp(t, t_eval, u2_arr))
                u = self._clip_control(np.array([u1, u2]))
                return self.cost.costate_derivatives(x, u, p)

            sol_bwd = solve_ivp(bwd_ode, [T, 0], p_T,
                                t_eval=t_eval[::-1],
                                method='LSODA', rtol=1e-6, atol=1e-4)
            if not sol_bwd.success:
                print(f"  Warning: backward integration failed at iter {iteration}")
                break
            p_arr = sol_bwd.y[:, ::-1]  # reverse to match t_eval order

            # ---- Control update (with convex relaxation alpha=0.5) -----------
            u_raw = np.array([
                self._clip_control(
                    self.cost.optimal_control(x_arr[:, i], p_arr[:, i])
                )
                for i in range(n_points)
            ]).T  # (2, n_points)

            u1_arr = 0.5 * u1_old + 0.5 * u_raw[0]
            u2_arr = 0.5 * u2_old + 0.5 * u_raw[1]

            # ---- Convergence check (relative to control bounds) --------------
            u1_max = self.u_bounds['u1'][1]
            u2_max = self.u_bounds['u2'][1]
            delta = max(np.max(np.abs(u1_arr - u1_old)) / (u1_max + 1e-12),
                        np.max(np.abs(u2_arr - u2_old)) / (u2_max + 1e-12))
            if iteration % 20 == 0 or delta < tolerance:
                print(f"  Iter {iteration:4d}: rel ||du||_inf = {delta:.3e}")
            if delta < tolerance:
                converged = True
                print(f"\nConverged in {iteration + 1} iterations.")
                break

        if not converged:
            print(f"\nWarning: FBS did not fully converge after {max_iterations} iterations.")

        total_cost = self._compute_total_cost(t_eval, x_arr,
                                              np.vstack([u1_arr, u2_arr]))
        print(f"\nTOTAL COST J = {total_cost:.4e}  R$")

        solution = {
            't': t_eval,
            'u1': u1_arr,
            'u2': u2_arr,
            'cost': total_cost,
            'success': converged,
            'iterations': iteration + 1,
        }
        state_keys = self._state_keys(n_states)
        for k, key in enumerate(state_keys):
            solution[key] = x_arr[k]
        for k, key in enumerate(state_keys):
            solution[f'p_{key}'] = p_arr[k]
        return solution

    def solve(self, x0: np.ndarray, T: float, method: str = 'shooting',
              **kwargs) -> Dict:
        if method == 'fbs':
            return self.solve_forward_backward_sweep(x0, T, **kwargs)
        return self.solve_shooting(x0, T, **kwargs)


class PontryaginSolver:
    """
    Solver para problemas de controle otimo usando Pontryagin Maximum Principle

    Resolve o problema:
        min J = integral_0^T L(x,u,t) dt + Phi(x(T))
        s.t. dx/dt = f(x,u,t)
             u em [u_min, u_max]

    Parametros
    ----------
    model : SEIRControlledModel
        Modelo SEIR controlado
    cost : QuadraticCost
        Funcao de custo
    u_bounds : dict
        Limites dos controles {'u1': (min, max), 'u2': (min, max)}
    """

    def __init__(
        self,
        model: SEIRControlledModel,
        cost: QuadraticCost,
        u_bounds: Dict[str, Tuple[float, float]]
    ):
        self.model = model
        self.cost = cost
        self.u_bounds = u_bounds

    def costate_derivatives(
        self,
        x: np.ndarray,
        u: np.ndarray,
        lam: np.ndarray,
        t: float
    ) -> np.ndarray:
        """
        Calcula dlambda/dt = -dH/dx (equacoes adjuntas)

        Derivadas analiticas do Hamiltoniano:
            dlambdaS/dt = beta(1-u1)I/N*(lambdaE-lambdaS)
            dlambdaE/dt = sigma*(lambdaI-lambdaE)
            dlambdaI/dt = -w1 + beta(1-u1)S/N*(lambdaE-lambdaS) + gamma*(lambdaR-lambdaI)
            dlambdaR/dt = 0

        Parametros
        ----------
        x : np.ndarray
            Estado [S, E, I, R]
        u : np.ndarray
            Controle [u1, u2]
        lam : np.ndarray
            Co-estados [lambdaS, lambdaE, lambdaI, lambdaR]
        t : float
            Tempo

        Retorna
        -------
        np.ndarray
            Derivadas [dlambdaS/dt, dlambdaE/dt, dlambdaI/dt, dlambdaR/dt]
        """
        S, E, I, R = x
        u1, u2 = u
        lam_S, lam_E, lam_I, lam_R = lam

        beta = self.model.beta
        sigma = self.model.sigma
        gamma = self.model.gamma
        N = self.model.N

        # dlambdaS/dt = -dH/dS
        dlam_S = beta * (1 - u1) * I / N * (lam_E - lam_S)

        # dlambdaE/dt = -dH/dE
        dlam_E = sigma * (lam_I - lam_E)

        # dlambdaI/dt = -dH/dI
        dlam_I = (-self.cost.w1
                  + beta * (1 - u1) * S / N * (lam_E - lam_S)
                  + gamma * (lam_R - lam_I))

        # dlambdaR/dt = -dH/dR
        dlam_R = 0.0

        return np.array([dlam_S, dlam_E, dlam_I, dlam_R])

    def optimal_control(
        self,
        x: np.ndarray,
        lam: np.ndarray,
        t: float
    ) -> np.ndarray:
        """
        Calcula controle otimo u*(t) = argmin H(x, u, lambda, t)

        Usando condicoes de primeira ordem dH/du = 0:
            u1* = clip(-betaSI/(2w2N)*(lambdaS-lambdaE), 0, u1_max)
            u2* = clip(-(lambdaR-lambdaS)/(2w3), 0, u2_max)

        Parametros
        ----------
        x : np.ndarray
            Estado [S, E, I, R]
        lam : np.ndarray
            Co-estados [lambdaS, lambdaE, lambdaI, lambdaR]
        t : float
            Tempo

        Retorna
        -------
        np.ndarray
            Controle otimo [u1*, u2*]
        """
        S, E, I, R = x
        lam_S, lam_E, lam_I, lam_R = lam

        beta = self.model.beta
        N = self.model.N

        # Controle sem limites (condicoes de primeira ordem)
        u1_unbounded = -beta * S * I / (2 * self.cost.w2 * N) * (lam_S - lam_E)
        u2_unbounded = -(lam_R - lam_S) / (2 * self.cost.w3)

        # Projetar em limites (bang-bang ou singular)
        u1_min, u1_max = self.u_bounds['u1']
        u2_min, u2_max = self.u_bounds['u2']

        u1_optimal = np.clip(u1_unbounded, u1_min, u1_max)
        u2_optimal = np.clip(u2_unbounded, u2_min, u2_max)

        return np.array([u1_optimal, u2_optimal])

    def solve_shooting(
        self,
        x0: np.ndarray,
        T: float,
        n_points: int = 365,
        lam0_guess: np.ndarray = None,
        tolerance: float = 1e-6,
        max_iterations: int = 100
    ) -> Dict:
        """
        Resolve o problema de controle otimo usando shooting method

        Algoritmo:
        1. Chutar lambda(0)
        2. Integrar [x, lambda] forward de 0 a T com controle u*(x, lambda)
        3. Verificar lambda(T) vs condicao terminal
        4. Ajustar lambda(0) com scipy.optimize.root
        5. Convergir

        Parametros
        ----------
        x0 : np.ndarray
            Condicoes iniciais [S0, E0, I0, R0]
        T : float
            Horizonte de tempo
        n_points : int
            Numero de pontos de avaliacao
        lam0_guess : np.ndarray, optional
            Chute inicial para lambda(0). Se None, usa zeros.
        tolerance : float
            Tolerancia de convergencia
        max_iterations : int
            Maximo de iteracoes

        Retorna
        -------
        dict
            Solucao com trajetorias otimas
        """
        print("\n" + "="*70)
        print("SHOOTING METHOD - PONTRYAGIN MAXIMUM PRINCIPLE")
        print("="*70)

        # Malha temporal
        t_eval = np.linspace(0, T, n_points)

        # Chute inicial para lambda(0)
        if lam0_guess is None:
            # Usar heuristica: comecar com valores pequenos
            lam0_guess = np.array([0.0, 0.0, -self.cost.w1, 0.0])

        print(f"Chute inicial lambda(0): {lam0_guess}")

        # Contador de iteracoes para callback
        iteration_count = [0]

        def shooting_residual(lam0):
            """
            Funcao residual para shooting:
            residual = lambda(T) - lambda_target(T)
            """
            iteration_count[0] += 1

            # Estado aumentado: y = [x, lambda] (8 dimensoes)
            y0 = np.concatenate([x0, lam0])

            def augmented_derivatives(t, y):
                """Derivadas do sistema aumentado [dx/dt, dlambda/dt]"""
                x = y[:4]  # [S, E, I, R]
                lam = y[4:]  # [lambdaS, lambdaE, lambdaI, lambdaR]

                # Calcular controle otimo
                u = self.optimal_control(x, lam, t)

                # dx/dt
                dx = self.model.derivatives(t, x, u)

                # dlambda/dt
                dlam = self.costate_derivatives(x, u, lam, t)

                return np.concatenate([dx, dlam])

            # Integrar de 0 a T
            sol = solve_ivp(
                augmented_derivatives,
                [0, T],
                y0,
                t_eval=t_eval,
                method='RK45',
                rtol=1e-8,
                atol=1e-10
            )

            if not sol.success:
                print(f"  Aviso: Integracao falhou na iteracao {iteration_count[0]}")
                return np.ones(4) * 1e10

            # Extrair lambda(T)
            lam_final = sol.y[4:, -1]

            # Condicao terminal desejada
            x_final = sol.y[:4, -1]
            lam_target = self.cost.terminal_costates(x_final)

            # Residual
            residual = lam_final - lam_target

            # Monitorar progresso
            residual_norm = np.linalg.norm(residual)
            if iteration_count[0] % 5 == 0 or residual_norm < tolerance:
                print(f"  Iteracao {iteration_count[0]:3d}: ||residual|| = {residual_norm:.2e}")

            return residual

        # Resolver usando scipy.optimize.root
        print("\nIniciando otimizacao...")
        result = root(
            shooting_residual,
            lam0_guess,
            method='hybr',  # Hybrid Newton/Levenberg-Marquardt
            tol=tolerance,
            options={'maxfev': max_iterations * 10}
        )

        if not result.success:
            print(f"\nAviso: Shooting method nao convergiu completamente")
            print(f"Mensagem: {result.message}")
            print(f"Continuando com melhor solucao encontrada...")

        # Solucao final
        lam0_optimal = result.x
        print(f"\nConvergencia atingida!")
        print(f"lambda(0) otimo: {lam0_optimal}")
        print(f"Total de iteracoes: {iteration_count[0]}")

        # Integrar novamente com lambda(0) otimo para obter trajetorias completas
        y0_optimal = np.concatenate([x0, lam0_optimal])

        def augmented_final(t, y):
            x = y[:4]
            lam = y[4:]
            u = self.optimal_control(x, lam, t)

            dx = self.model.derivatives(t, x, u)
            dlam = self.costate_derivatives(x, u, lam, t)

            return np.concatenate([dx, dlam])

        sol_final = solve_ivp(
            augmented_final,
            [0, T],
            y0_optimal,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )

        # Extrair estados e co-estados da solucao
        x_array = sol_final.y[:4]  # (4, n_points)
        lam_array = sol_final.y[4:]  # (4, n_points)

        # Calcular controles nos pontos de tempo avaliados
        u_array = np.array([self.optimal_control(x_array[:, i], lam_array[:, i], sol_final.t[i])
                            for i in range(len(sol_final.t))]).T  # (2, n_points)

        # Calcular custo total
        total_cost = self._compute_total_cost(sol_final.t, x_array, u_array)

        print(f"\n" + "="*70)
        print(f"CUSTO TOTAL: {total_cost:.4e}")
        print("="*70)

        return {
            't': sol_final.t,
            'S': x_array[0],
            'E': x_array[1],
            'I': x_array[2],
            'R': x_array[3],
            'lam_S': lam_array[0],
            'lam_E': lam_array[1],
            'lam_I': lam_array[2],
            'lam_R': lam_array[3],
            'u1': u_array[0],
            'u2': u_array[1],
            'cost': total_cost,
            'success': result.success,
            'iterations': iteration_count[0]
        }

    def _compute_total_cost(
        self,
        t: np.ndarray,
        x: np.ndarray,
        u: np.ndarray
    ) -> float:
        """Calcula custo total J = integral L dt + Phi(x(T))"""
        # Custo de running (integracao numerica usando trapezios)
        running_costs = []
        for i in range(len(t)):
            xi = x[:, i]
            ui = u[:, i]
            running_costs.append(self.cost.running_cost(xi, ui, t[i]))

        running_cost_integral = np.trapz(running_costs, t)

        # Custo terminal
        x_final = x[:, -1]
        terminal_cost = self.cost.terminal_cost(x_final)

        # Custo total
        total_cost = running_cost_integral + terminal_cost

        return total_cost

    def solve(self, x0: np.ndarray, T: float, **kwargs) -> Dict:
        """
        Interface principal para resolver controle otimo

        Parametros
        ----------
        x0 : np.ndarray
            Condicoes iniciais
        T : float
            Horizonte de tempo
        **kwargs : dict
            Argumentos opcionais para solve_shooting

        Retorna
        -------
        dict
            Solucao do problema de controle otimo
        """
        method = kwargs.pop('method', 'shooting')

        if method == 'shooting':
            return self.solve_shooting(x0, T, **kwargs)
        else:
            raise ValueError(f"Metodo '{method}' nao implementado. Use 'shooting'.")
