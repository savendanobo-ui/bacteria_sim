# src/bacteria_sim/cli_calibration.py

import cyclopts
import numpy as np
from pathlib import Path
from typing import Optional

from bacteria_sim.calibration import (
    load_experimental_data, calibrate, validate,
)
from bacteria_sim.config import SimulationConfig

app = cyclopts.App(
    help="Calibración y validación del autómata celular."
)


def _check_config(**kwargs) -> None:
    """Valida parámetros con SimulationConfig y sale con un mensaje limpio."""
    try:
        SimulationConfig(**kwargs)
    except ValueError as e:
        raise SystemExit(f"Error: {e}")


def _parse_range(s: str) -> list:
    """Parsea un rango tipo '0.05,0.5,0.05' a una lista."""
    parts = s.split(",")
    if len(parts) == 3:
        start, end, step = map(float, parts)
        if step <= 0:
            raise ValueError(f"El paso del rango debe ser > 0 (recibido: {step})")
        return np.arange(start, end + 1e-9, step).round(2).tolist()
    return [float(x) for x in parts]


@app.command()
def calibrate_cmd(
    data: str,
    output: str = "resultados_calibracion.csv",
    output_dir: str = "resultados_calibracion",
    n_jobs: Optional[int] = None,
    # Parámetros de simulación
    L: int = 100,
    steps: int = 480,
    init_cells: float = 0.005,
    init_substrate: float = 0.5,
    seed: int = 42,
    # Rangos de parámetros
    N0_values: str = "2,3,4,5,6",
    P_grow_range: str = "0.05,0.50,0.05",
    P_div_range: str = "0.05,0.50,0.05",
    # Guardado de figuras
    save_top: int = 10,
    save_worst: int = 5,
    snapshot_interval: int = 50,
    # Validación automática
    auto_validate: bool = False,
):
    """Ejecuta la calibración con barrido de parámetros en paralelo."""
    exp_time, exp_pop = load_experimental_data(Path(data))

    try:
        N0_list = [int(x) for x in N0_values.split(",")]
        P_grow_list = _parse_range(P_grow_range)
        P_div_list = _parse_range(P_div_range)
    except ValueError as e:
        raise SystemExit(f"Error en los rangos de parámetros: {e}")

    # Validar todo antes de lanzar los procesos en paralelo
    _check_config(L=L, steps=steps, init_cells=init_cells,
                  init_substrate=init_substrate, seed=seed)
    for n in N0_list:
        _check_config(N0=n)
    for p in P_grow_list:
        _check_config(P_grow=p)
    for p in P_div_list:
        _check_config(P_div=p)
    if n_jobs is not None and n_jobs < 1:
        raise SystemExit(f"Error: n_jobs debe ser >= 1 (recibido: {n_jobs})")

    # --- Fase 1: Calibración ---
    print("=" * 60)
    print("FASE 1: CALIBRACIÓN")
    print("=" * 60)
    df = calibrate(
        experimental_data=(exp_time, exp_pop),
        N0_values=N0_list,
        P_grow_values=P_grow_list,
        P_div_values=P_div_list,
        L=L,
        steps=steps,
        init_cells=init_cells,
        init_substrate=init_substrate,
        seed=seed,
        n_jobs=n_jobs,
        output_csv=Path(output),
        output_dir=Path(output_dir),
        save_top_n=save_top,
        save_worst_n=save_worst,
        snapshot_interval=snapshot_interval,
    )

    # --- Fase 2: Validación automática ---
    if auto_validate:
        best = df.iloc[0]
        print("\n" + "=" * 60)
        print("FASE 2: VALIDACIÓN AUTOMÁTICA")
        print("=" * 60)
        print(f"Usando los mejores parámetros: "
              f"N0={int(best['N0'])}, "
              f"P_grow={best['P_grow']}, "
              f"P_div={best['P_div']}")

        config = SimulationConfig(
            L=L, steps=steps,
            init_cells=init_cells,
            init_substrate=init_substrate,
            N0=int(best["N0"]),
            P_grow=best["P_grow"],
            P_div=best["P_div"],
            seed=seed,
        )

        metrics = validate(config, (exp_time, exp_pop))

        print(f"\n--- Resultados de validación ---")
        print(f"MSE (calibración) = {best['MSE']:.6f}")
        print(f"MSE (validación)  = {metrics['MSE']:.6f}")
        print(f"R²                = {metrics['R2']:.6f}")
        print(f"MAE               = {metrics['MAE']:.6f}")

        diff = abs(best['MSE'] - metrics['MSE']) / max(best['MSE'], 1e-12) * 100
        print(f"\nDiferencia MSE cal/val: {diff:.2f}%")
        if diff < 20:
            print("El modelo es reproducible (diferencia < 20%).")
        else:
            print("Diferencia grande entre calibración y validación.")


@app.command()
def validate_cmd(
    data: str,
    N0: int,
    P_grow: float,
    P_div: float,
    L: int = 100,
    steps: int = 480,
    init_cells: float = 0.005,
    init_substrate: float = 1.0,
    seed: int = 42,
    show_progress: bool = True,
):
    """Valida el modelo con parámetros específicos."""
    exp_time, exp_pop = load_experimental_data(Path(data))
    _check_config(L=L, steps=steps, init_cells=init_cells,
                  init_substrate=init_substrate, N0=N0,
                  P_grow=P_grow, P_div=P_div, seed=seed)
    config = SimulationConfig(
        L=L, steps=steps,
        init_cells=init_cells,
        init_substrate=init_substrate,
        N0=N0, P_grow=P_grow, P_div=P_div,
        seed=seed,
    )
    metrics = validate(config, (exp_time, exp_pop), show_progress=show_progress,)
    print(f"\n--- Resultados de validación ---")
    print(f"N0 = {N0}, P_grow = {P_grow}, P_div = {P_div}")
    print(f"L = {L}, steps = {steps}")
    print(f"init_cells = {init_cells}, init_substrate = {init_substrate}")
    print(f"MSE = {metrics['MSE']:.6f}")
    print(f"R²  = {metrics['R2']:.6f}")
    print(f"MAE = {metrics['MAE']:.6f}")