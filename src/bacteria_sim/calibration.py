"""
Módulo de calibración del autómata celular.

Realiza un barrido sistemático de los parámetros (N0, P_grow, P_div) y
selecciona la combinación que minimiza el Error Cuadrático Medio Normalizado
(MSE) entre la curva simulada y los datos experimentales.

Opcionalmente, guarda las curvas de crecimiento e imágenes de las mejores
(y peores) simulaciones para análisis visual.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from itertools import product
from typing import Optional, Tuple
from .model import BacteriaCellularAutomaton


EPSILON = 1e-6


def load_experimental_data(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Carga datos experimentales desde CSV con columnas 'time' y 'population'."""
    df = pd.read_csv(filepath)
    return df["time"].to_numpy(dtype=float), df["population"].to_numpy(dtype=float)


def normalize_curve(curve: np.ndarray) -> np.ndarray:
    """Normaliza una curva dividiéndola por su valor inicial."""
    if curve[0] == 0:
        raise ValueError("El valor inicial de la curva es cero.")
    return curve / curve[0]


def run_simulation(
    N0: int,
    P_grow: float,
    P_div: float,
    L: int = 100,
    steps: int = 300,
    init_cells: float = 0.01,
    init_substrate: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Ejecuta una simulación y retorna la curva de crecimiento."""
    if seed is not None:
        np.random.seed(seed)

    model = BacteriaCellularAutomaton(
        L=L, init_cells=init_cells, init_substrate=init_substrate,
        N0=N0, P_grow=P_grow, P_div=P_div,
    )

    population = [model.get_cell_count()]
    for _ in range(steps):
        model.step()
        population.append(model.get_cell_count())

    return np.array(population, dtype=float)


def compute_mse(sim_curve: np.ndarray, exp_curve: np.ndarray) -> float:
    """Calcula el MSE normalizado entre dos curvas."""
    if len(exp_curve) != len(sim_curve):
        exp_curve = np.interp(
            np.linspace(0, 1, len(sim_curve)),
            np.linspace(0, 1, len(exp_curve)),
            exp_curve,
        )
    sim_norm = normalize_curve(sim_curve)
    exp_norm = normalize_curve(exp_curve)
    errors = ((sim_norm - exp_norm) / (exp_norm + EPSILON)) ** 2
    return float(np.mean(errors))


def _save_growth_curve(
    sim_curve: np.ndarray,
    exp_curve: np.ndarray,
    N0: int, P_grow: float, P_div: float, MSE: float,
    output_path: Path,
):
    """Guarda un gráfico de la curva simulada vs experimental."""
    sim_norm = normalize_curve(sim_curve)
    exp_norm = normalize_curve(exp_curve)
    time = np.arange(len(sim_norm))

    plt.figure(figsize=(8, 5))
    plt.plot(time, exp_norm, "o-", label="Experimental", color="black", markersize=4)
    plt.plot(time, sim_norm, "-", label="Simulación", color="steelblue", linewidth=2)
    plt.xlabel("Tiempo (min)")
    plt.ylabel("Población normalizada")
    plt.title(f"N0={N0}, P_grow={P_grow}, P_div={P_div}\nMSE = {MSE:.6f}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def _save_snapshots(
    N0: int, P_grow: float, P_div: float,
    L: int, steps: int, init_cells: float, init_substrate: float,
    seed: Optional[int], output_dir: Path, snapshot_interval: int = 50,
):
    """Guarda imágenes del estado del autómata cada N pasos."""
    from matplotlib.colors import ListedColormap, BoundaryNorm

    if seed is not None:
        np.random.seed(seed)

    model = BacteriaCellularAutomaton(
        L=L, init_cells=init_cells, init_substrate=init_substrate,
        N0=N0, P_grow=P_grow, P_div=P_div,
    )

    cmap_cells = ListedColormap(["white", "blue", "red"])
    bounds_cells = [0, 1, 2, 3]
    norm_cells = BoundaryNorm(bounds_cells, cmap_cells.N)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Guardar estado inicial
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(model.G, cmap=cmap_cells, norm=norm_cells, interpolation="nearest")
    ax1.set_title("Células (paso 0)")
    ax2.imshow(model.S, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    ax2.set_title("Sustrato")
    plt.savefig(output_dir / "step_0000.png", dpi=80, bbox_inches="tight")
    plt.close(fig)

    for step in range(1, steps + 1):
        model.step()
        if step % snapshot_interval == 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(model.G, cmap=cmap_cells, norm=norm_cells, interpolation="nearest")
            ax1.set_title(f"Células (paso {step})")
            ax2.imshow(model.S, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
            ax2.set_title("Sustrato")
            plt.savefig(output_dir / f"step_{step:04d}.png", dpi=80, bbox_inches="tight")
            plt.close(fig)


def calibrate(
    experimental_data: Tuple[np.ndarray, np.ndarray],
    N0_values: list = [2, 3, 4, 5, 6],
    P_grow_values: list = None,
    P_div_values: list = None,
    L: int = 100,
    steps: int = 300,
    init_cells: float = 0.01,
    init_substrate: float = 0.5,
    seed: Optional[int] = 42,
    output_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    save_top_n: int = 10,
    save_worst_n: int = 10,
    snapshot_interval: int = 50,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Realiza el barrido sistemático de parámetros.

    Si output_dir se especifica, guarda:
        - Curvas de crecimiento para las mejores y peores N simulaciones.
        - Snapshots del autómata cada `snapshot_interval` pasos.
        - Un archivo metadata.json con los parámetros y métricas.
    """
    if P_grow_values is None:
        P_grow_values = np.arange(0.05, 0.51, 0.05).round(2).tolist()
    if P_div_values is None:
        P_div_values = np.arange(0.05, 0.51, 0.05).round(2).tolist()

    exp_time, exp_pop = experimental_data
    exp_pop_norm = normalize_curve(exp_pop)
    exp_interp = np.interp(
        np.linspace(0, 1, steps + 1),
        np.linspace(0, 1, len(exp_pop_norm)),
        exp_pop_norm,
    )

    total = len(N0_values) * len(P_grow_values) * len(P_div_values)
    results = []
    counter = 0

    if verbose:
        print(f"Iniciando calibración: {total} combinaciones a evaluar.")

    for N0, P_grow, P_div in product(N0_values, P_grow_values, P_div_values):
        sim_curve = run_simulation(
            N0=N0, P_grow=P_grow, P_div=P_div,
            L=L, steps=steps,
            init_cells=init_cells, init_substrate=init_substrate,
            seed=seed,
        )
        mse = compute_mse(sim_curve, exp_interp)
        results.append({"N0": N0, "P_grow": P_grow, "P_div": P_div, "MSE": mse})

        counter += 1
        if verbose and counter % 50 == 0:
            print(f"  Progreso: {counter}/{total} ({100*counter/total:.1f}%)")

    df = pd.DataFrame(results).sort_values("MSE", ascending=True).reset_index(drop=True)

    # Guardar CSV con TODAS las métricas
    if output_csv is not None:
        df.to_csv(output_csv, index=False)
        if verbose:
            print(f"Resultados guardados en {output_csv}")

    # Guardar curvas y snapshots de las mejores y peores simulaciones
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Guardar top N
        top_dir = output_dir / f"top_{save_top_n}"
        top_dir.mkdir(exist_ok=True)
        for rank, (_, row) in enumerate(df.head(save_top_n).iterrows(), start=1):
            sim_dir = top_dir / f"rank_{rank:02d}_N0-{row['N0']}_Pg-{row['P_grow']}_Pd-{row['P_div']}"
            sim_dir.mkdir(exist_ok=True)

            sim_curve = run_simulation(
                N0=int(row["N0"]), P_grow=row["P_grow"], P_div=row["P_div"],
                L=L, steps=steps, init_cells=init_cells,
                init_substrate=init_substrate, seed=seed,
            )
            _save_growth_curve(sim_curve, exp_interp,
                               int(row["N0"]), row["P_grow"], row["P_div"], row["MSE"],
                               sim_dir / "curva_crecimiento.png")

            _save_snapshots(
                N0=int(row["N0"]), P_grow=row["P_grow"], P_div=row["P_div"],
                L=L, steps=steps, init_cells=init_cells,
                init_substrate=init_substrate, seed=seed,
                output_dir=sim_dir / "snapshots",
                snapshot_interval=snapshot_interval,
            )

            with open(sim_dir / "metadata.json", "w") as f:
                json.dump({
                    "rank": rank,
                    "N0": int(row["N0"]),
                    "P_grow": float(row["P_grow"]),
                    "P_div": float(row["P_div"]),
                    "MSE": float(row["MSE"]),
                    "L": L, "steps": steps,
                    "init_cells": init_cells, "init_substrate": init_substrate,
                }, f, indent=2)

        # Guardar peores N (últimas filas del df ordenado)
        worst_dir = output_dir / f"worst_{save_worst_n}"
        worst_dir.mkdir(exist_ok=True)
        for rank, (_, row) in enumerate(df.tail(save_worst_n).iterrows(), start=1):
            sim_dir = worst_dir / f"rank_{rank:02d}_N0-{row['N0']}_Pg-{row['P_grow']}_Pd-{row['P_div']}"
            sim_dir.mkdir(exist_ok=True)

            sim_curve = run_simulation(
                N0=int(row["N0"]), P_grow=row["P_grow"], P_div=row["P_div"],
                L=L, steps=steps, init_cells=init_cells,
                init_substrate=init_substrate, seed=seed,
            )
            _save_growth_curve(sim_curve, exp_interp,
                               int(row["N0"]), row["P_grow"], row["P_div"], row["MSE"],
                               sim_dir / "curva_crecimiento.png")

        if verbose:
            print(f"Curvas y snapshots guardados en {output_dir}")

    if verbose:
        print("\n--- Mejor combinación ---")
        best = df.iloc[0]
        print(f"N0 = {best['N0']}, P_grow = {best['P_grow']}, P_div = {best['P_div']}")
        print(f"MSE = {best['MSE']:.6f}")

    return df


def validate(
    best_params: dict,
    experimental_data: Tuple[np.ndarray, np.ndarray],
    L: int = 100,
    steps: int = 300,
    init_cells: float = 0.01,
    init_substrate: float = 0.5,
    seed: Optional[int] = 42,
) -> dict:
    """
    Valida el modelo con los mejores parámetros contra datos experimentales
    independientes (no usados en la calibración).

    Retorna un diccionario con las métricas: MSE, R², MAE.
    """
    sim_curve = run_simulation(
        N0=int(best_params["N0"]),
        P_grow=best_params["P_grow"],
        P_div=best_params["P_div"],
        L=L, steps=steps,
        init_cells=init_cells, init_substrate=init_substrate,
        seed=seed,
    )

    exp_time, exp_pop = experimental_data
    exp_pop_norm = normalize_curve(exp_pop)

    # Interpolar a la misma longitud
    exp_interp = np.interp(
        np.linspace(0, 1, steps + 1),
        np.linspace(0, 1, len(exp_pop_norm)),
        exp_pop_norm,
    )
    sim_norm = normalize_curve(sim_curve)

    # MSE
    mse = compute_mse(sim_curve, exp_interp)

    # R²
    ss_res = np.sum((exp_interp - sim_norm) ** 2)
    ss_tot = np.sum((exp_interp - np.mean(exp_interp)) ** 2)
    r2 = 1 - ss_res / (ss_tot + EPSILON)

    # MAE
    mae = np.mean(np.abs(sim_norm - exp_interp))

    return {"MSE": float(mse), "R2": float(r2), "MAE": float(mae)}