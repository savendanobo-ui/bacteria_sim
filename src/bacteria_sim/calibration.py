"""
Módulo de calibración y validación del autómata celular.

Este módulo contiene la lógica de calibración y validación, pero NO
se encarga de parsear argumentos (eso lo hace cli_calibration.py).
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path
from itertools import product
from typing import Optional, Tuple
from multiprocessing import Pool, cpu_count

from bacteria_sim.model import BacteriaCellularAutomaton
from bacteria_sim.config import SimulationConfig

from tqdm import tqdm #para añadir barra de progreso

EPSILON = 1e-6

# Variables globales para compartir datos con los workers de multiprocessing
_EXP_INTERP = None
_CONFIG_TEMPLATE = None


def _init_worker(exp_interp, config_template):
    """Inicializa las variables globales en cada worker de multiprocessing."""
    global _EXP_INTERP, _CONFIG_TEMPLATE
    _EXP_INTERP = exp_interp
    _CONFIG_TEMPLATE = config_template


def load_experimental_data(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Carga datos experimentales desde CSV con columnas 'time' y 'population'."""
    df = pd.read_csv(filepath)
    return df["time"].to_numpy(dtype=float), df["population"].to_numpy(dtype=float)


def normalize_curve(curve: np.ndarray) -> np.ndarray:
    """Normaliza una curva dividiéndola por su valor inicial."""
    if curve[0] == 0:
        raise ValueError("El valor inicial de la curva es cero.")
    return curve / curve[0]


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


def _run_simulation_no_viz(
    config: SimulationConfig,
    show_progress: bool = False,
    return_substrate: bool = False,
):
    """Ejecuta la simulación sin visualización (para calibración y validación)."""
    if config.seed is not None:
        np.random.seed(config.seed)

    model = BacteriaCellularAutomaton(
        L=config.L,
        init_cells=config.init_cells,
        init_substrate=config.init_substrate,
        N0=config.N0,
        P_grow=config.P_grow,
        P_div=config.P_div,
        P_maint=config.P_maint,
        starve_threshold=config.starve_threshold,
        verbose=False, #desactivar mensajes de depuración
    )

    population = [model.get_cell_count()]
    substrate = [model.get_substrate_amount()] if return_substrate else []

    iterador = range(config.steps)
    if show_progress:
        iterador = tqdm(
            iterador,
            desc="Validando",
            unit="paso",
            leave=False, #para que la barra de progreso desaparezca al acabar la validación
        )

    for _ in iterador:
        model.step()
        population.append(model.get_cell_count())
        if return_substrate:
            substrate.append(model.get_substrate_amount())

    if return_substrate:
        return np.array(population, dtype=float), np.array(substrate, dtype=float)
    return np.array(population, dtype=float)


def _run_one_config(config_tuple: tuple) -> tuple:
    """
    Ejecuta UNA simulación con una combinación de parámetros y devuelve el MSE.

    Esta función debe estar a nivel de módulo para ser picklable por
    multiprocessing.
    """
    N0, P_grow, P_div, init_substrate, P_maint, starve_threshold = config_tuple

    # Usar el config_template global (no valores hardcodeados)
    config = SimulationConfig(
        L=_CONFIG_TEMPLATE.L,
        steps=_CONFIG_TEMPLATE.steps,
        init_cells=_CONFIG_TEMPLATE.init_cells,
        init_substrate=init_substrate,
        N0=N0, P_grow=P_grow, P_div=P_div,
        P_maint=P_maint,
        starve_threshold=int(starve_threshold),
        seed=_CONFIG_TEMPLATE.seed,
    )

    sim_curve = _run_simulation_no_viz(config)
    mse = compute_mse(sim_curve, _EXP_INTERP)
    return (N0, P_grow, P_div, init_substrate, P_maint, int(starve_threshold), mse)


def calibrate(
    experimental_data: Tuple[np.ndarray, np.ndarray],
    N0_values: list = None,
    P_grow_values: list = None,
    P_div_values: list = None,
    init_substrate_values: list = None,
    P_maint_values: list = None,
    starve_threshold_values: list = None,
    P_maint: float = 0.0,
    starve_threshold: int = 3,
    steps: int = 480,
    L: int = 100,
    init_cells: Optional[float] = None,
    init_substrate: float = 1.0,
    seed: int = 42,
    n_jobs: int = None,
    output_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    save_top_n: int = 0,
    save_worst_n: int = 0,
    snapshot_interval: int = 0,
    plot_substrate: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Realiza el barrido de parámetros EN PARALELO y devuelve la matriz de resultados.
    """
    exp_time, exp_pop = experimental_data

    # Cálculo automático de init_cells si no fue especificado
    if init_cells is None or init_cells <= 0:
        k_fold = float(np.max(exp_pop) / (exp_pop[0] + EPSILON))
        target_plateau_density = 0.036
        init_cells = round(target_plateau_density / max(k_fold, 1.0), 4)
        if verbose:
            print(f"init_cells auto-calculado: {init_cells} "
                  f"(Fold-change experimental K={k_fold:.2f}, densidad objetivo={target_plateau_density})")

    if N0_values is None:
        N0_values = [2, 3, 4, 5, 6]
    if P_grow_values is None:
        P_grow_values = np.arange(0.05, 0.51, 0.05).round(2).tolist()
    if P_div_values is None:
        P_div_values = np.arange(0.05, 0.51, 0.05).round(2).tolist()
    if init_substrate_values is None:
        init_substrate_values = [init_substrate]
    if P_maint_values is None:
        P_maint_values = [P_maint]
    if starve_threshold_values is None:
        starve_threshold_values = [starve_threshold]

    if n_jobs is None:
        n_jobs = cpu_count()

    # Crear el config_template ANTES del Pool
    config_template = SimulationConfig(
        L=L,
        steps=steps,
        init_cells=init_cells,
        init_substrate=init_substrate_values[0],
        P_maint=P_maint_values[0],
        starve_threshold=starve_threshold_values[0],
        seed=seed,
    )

    # Preparar la curva experimental interpolada (usar steps, no 481 hardcodeado)
    exp_pop_norm = normalize_curve(exp_pop)
    exp_interp = np.interp(
        np.linspace(0, 1, steps + 1),          # steps + 1
        np.linspace(0, 1, len(exp_pop_norm)),
        exp_pop_norm,
    )

    # Generar todas las combinaciones
    configs = list(product(
        N0_values,
        P_grow_values,
        P_div_values,
        init_substrate_values,
        P_maint_values,
        starve_threshold_values,
    ))

    if verbose:
        print(f"Calibración: {len(configs)} combinaciones, {n_jobs} núcleos")
        print(f"Parámetros: L={L}, steps={steps}, init_cells={init_cells}, "
              f"sustratos={init_substrate_values}, P_maint={P_maint_values}, "
              f"starve_threshold={starve_threshold_values}, seed={seed}")

    # Pasar AMBAS variables al initializer
    with Pool(
        processes=n_jobs,
        initializer=_init_worker,
        initargs=(exp_interp, config_template),   # AMBOS argumentos
    ) as pool:
        #barra de progreso con tqdm
        results = list(tqdm(
            pool.imap(_run_one_config, configs),
            total=len(configs),
            desc="Calibrando",
            unit="sim",
        ))

    # Crear DataFrame
    df = pd.DataFrame(
        results,
        columns=["N0", "P_grow", "P_div", "init_substrate", "P_maint", "starve_threshold", "MSE"],
    )
    df = df.sort_values("MSE", ascending=True).reset_index(drop=True)

    if output_csv is not None:
        df.to_csv(output_csv, index=False)
        if verbose:
            print(f"Resultados guardados en {output_csv}")

    # Guardar curvas y snapshots
    if output_dir is not None and (save_top_n > 0 or save_worst_n > 0):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_top_and_worst(
            df, experimental_data, output_dir,
            save_top_n=save_top_n,
            save_worst_n=save_worst_n,
            snapshot_interval=snapshot_interval,
            config_template=config_template,
            plot_substrate=plot_substrate,
            verbose=verbose,
        )

    if verbose:
        best = df.iloc[0]
        print(f"\nMejor combinación: N0={best['N0']}, "
              f"P_grow={best['P_grow']}, P_div={best['P_div']}, "
              f"init_substrate={best['init_substrate']}, "
              f"P_maint={best['P_maint']}, "
              f"starve_threshold={int(best['starve_threshold'])}, "
              f"MSE={best['MSE']:.6f}")

    return df


def validate(
    config: SimulationConfig,
    experimental_data: Tuple[np.ndarray, np.ndarray],
    show_progress: bool = False, 
) -> dict:
    """Valida el modelo con los parámetros dados contra datos experimentales."""
    exp_time, exp_pop = experimental_data
    exp_pop_norm = normalize_curve(exp_pop)
    exp_interp = np.interp(
        np.linspace(0, 1, config.steps + 1),
        np.linspace(0, 1, len(exp_pop_norm)),
        exp_pop_norm,
    )

    sim_curve = _run_simulation_no_viz(config)
    sim_norm = normalize_curve(sim_curve)

    mse = compute_mse(sim_curve, exp_interp)

    ss_res = np.sum((exp_interp - sim_norm) ** 2)
    ss_tot = np.sum((exp_interp - np.mean(exp_interp)) ** 2)
    r2 = 1 - ss_res / (ss_tot + EPSILON)

    mae = np.mean(np.abs(sim_norm - exp_interp))

    return {"MSE": float(mse), "R2": float(r2), "MAE": float(mae)}


def _save_growth_curve(
    sim_curve: np.ndarray,
    exp_curve: np.ndarray,
    N0: int,
    P_grow: float,
    P_div: float,
    MSE: float,
    output_path: Path,
    init_substrate: Optional[float] = None,
    P_maint: Optional[float] = None,
    starve_threshold: Optional[int] = None,
    time_sim: Optional[np.ndarray] = None,
    time_exp: Optional[np.ndarray] = None,
    raw_exp_pop: Optional[np.ndarray] = None,
):
    """Guarda un gráfico de la curva simulada vs experimental."""
    sim_norm = normalize_curve(sim_curve)
    exp_norm = normalize_curve(exp_curve)

    plt.figure(figsize=(8, 5))

    if time_sim is not None:
        x_sim = time_sim
        x_label = "Tiempo (minutos)"
    else:
        x_sim = np.arange(len(sim_norm))
        x_label = "Paso"

    if time_exp is not None and raw_exp_pop is not None:
        plt.plot(time_exp, normalize_curve(raw_exp_pop), "o-", label="Experimental", color="black", markersize=4)
    else:
        plt.plot(x_sim, exp_norm, "o-", label="Experimental", color="black", markersize=4)

    plt.plot(x_sim, sim_norm, "-", label="Simulación", color="steelblue", linewidth=2)
    plt.xlabel(x_label)
    plt.ylabel("Población normalizada")
    title_str = f"N0={N0}, P_grow={P_grow}, P_div={P_div}"
    if init_substrate is not None:
        title_str += f", S0={init_substrate}"
    if P_maint is not None:
        title_str += f", Pm={P_maint}"
    if starve_threshold is not None:
        title_str += f", St={starve_threshold}"
    title_str += f"\nMSE = {MSE:.6f}"
    plt.title(title_str)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def _save_growth_and_substrate_curve(
    sim_curve: np.ndarray,
    sub_curve: np.ndarray,
    exp_curve: np.ndarray,
    N0: int,
    P_grow: float,
    P_div: float,
    MSE: float,
    output_path: Path,
    init_substrate: Optional[float] = None,
    P_maint: Optional[float] = None,
    starve_threshold: Optional[int] = None,
    time_sim: Optional[np.ndarray] = None,
    time_exp: Optional[np.ndarray] = None,
    raw_exp_pop: Optional[np.ndarray] = None,
):
    """Guarda un gráfico con la curva de crecimiento y la curva de sustrato superpuesta en verde."""
    sim_norm = normalize_curve(sim_curve)
    exp_norm = normalize_curve(exp_curve)

    fig, ax1 = plt.subplots(figsize=(8.5, 5))

    if time_sim is not None:
        x_sim = time_sim
        x_label = "Tiempo (minutos)"
    else:
        x_sim = np.arange(len(sim_norm))
        x_label = "Paso"

    # Eje 1 (Izquierdo): Células / Población
    if time_exp is not None and raw_exp_pop is not None:
        line1 = ax1.plot(time_exp, normalize_curve(raw_exp_pop), "o-", label="Experimental", color="black", markersize=4)
    else:
        line1 = ax1.plot(x_sim, exp_norm, "o-", label="Experimental", color="black", markersize=4)

    line2 = ax1.plot(x_sim, sim_norm, "-", label="Simulación (células)", color="steelblue", linewidth=2)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Población normalizada", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Eje 2 (Derecho): Sustrato en verde
    ax2 = ax1.twinx()
    line3 = ax2.plot(x_sim, sub_curve, "--", label="Sustrato disponible", color="forestgreen", linewidth=2)
    ax2.set_ylabel("Partículas de sustrato", color="forestgreen")
    ax2.tick_params(axis="y", labelcolor="forestgreen")
    ax2.grid(False)

    title_str = f"N0={N0}, P_grow={P_grow}, P_div={P_div}"
    if init_substrate is not None:
        title_str += f", S0={init_substrate}"
    if P_maint is not None:
        title_str += f", Pm={P_maint}"
    if starve_threshold is not None:
        title_str += f", St={starve_threshold}"
    title_str += f"\nMSE = {MSE:.6f}"
    plt.title(title_str)

    # Leyenda combinada
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _save_snapshots(
    config: SimulationConfig,
    output_dir: Path,
    snapshot_interval: int = 50,
):
    """Guarda imágenes del estado del autómata cada N pasos."""
    if config.seed is not None:
        np.random.seed(config.seed)

    model = BacteriaCellularAutomaton(
        L=config.L, init_cells=config.init_cells,
        init_substrate=config.init_substrate,
        N0=config.N0, P_grow=config.P_grow, P_div=config.P_div,
        P_maint=config.P_maint,
        starve_threshold=config.starve_threshold,
    )

    cmap_cells = ListedColormap(["white", "blue", "red", "forestgreen"])
    bounds_cells = [0, 1, 2, 3, 4]
    norm_cells = BoundaryNorm(bounds_cells, cmap_cells.N)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Estado inicial
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(model.G, cmap=cmap_cells, norm=norm_cells, interpolation="nearest")
    ax1.set_title("Células (paso 0)")
    ax2.imshow(model.S, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    ax2.set_title("Sustrato")
    plt.savefig(output_dir / "step_0000.png", dpi=80, bbox_inches="tight")
    plt.close(fig)

    for step in range(1, config.steps + 1):
        model.step()
        if step % snapshot_interval == 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(model.G, cmap=cmap_cells, norm=norm_cells, interpolation="nearest")
            ax1.set_title(f"Células (paso {step})")
            ax2.imshow(model.S, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
            ax2.set_title("Sustrato")
            plt.savefig(output_dir / f"step_{step:04d}.png", dpi=80, bbox_inches="tight")
            plt.close(fig)


def _save_top_and_worst(
    df: pd.DataFrame,
    experimental_data: Tuple[np.ndarray, np.ndarray],
    output_dir: Path,
    save_top_n: int,
    save_worst_n: int,
    snapshot_interval: int,
    config_template: SimulationConfig,
    plot_substrate: bool = True,
    verbose: bool = True,
):
    """Guarda curvas y snapshots de las mejores y peores simulaciones."""
    exp_time, exp_pop = experimental_data
    exp_pop_norm = normalize_curve(exp_pop)
    exp_interp = np.interp(
        np.linspace(0, 1, config_template.steps + 1),
        np.linspace(0, 1, len(exp_pop_norm)),
        exp_pop_norm,
    )

    # Conversión temporal a minutos relativos (t' = t - t_0)
    # Permite que el inicio de las mediciones experimentales coincida con t = 0 min
    is_hours = np.max(exp_time) <= 48
    time_scale = 60.0 if is_hours else 1.0
    exp_time_rel = (exp_time - exp_time[0]) * time_scale
    total_minutes = float(exp_time_rel[-1])
    time_sim_min = np.linspace(0, total_minutes, config_template.steps + 1)
    time_exp_min = exp_time_rel

    # Guardar top N
    if save_top_n > 0:
        top_dir = output_dir / f"top_{save_top_n}"
        top_dir.mkdir(parents=True, exist_ok=True)
        for rank, (_, row) in enumerate(df.head(save_top_n).iterrows(), start=1):
            sub_val = float(row.get("init_substrate", config_template.init_substrate))
            pm_val = float(row.get("P_maint", config_template.P_maint))
            st_val = int(row.get("starve_threshold", config_template.starve_threshold))
            sim_dir = top_dir / f"rank_{rank:02d}_N0-{row['N0']}_Pg-{row['P_grow']}_Pd-{row['P_div']}_S0-{sub_val}_Pm-{pm_val}_St-{st_val}"
            sim_dir.mkdir(exist_ok=True)

            config = SimulationConfig(
                L=config_template.L, steps=config_template.steps,
                init_cells=config_template.init_cells,
                init_substrate=sub_val,
                N0=int(row["N0"]), P_grow=row["P_grow"], P_div=row["P_div"],
                P_maint=pm_val,
                starve_threshold=st_val,
                seed=config_template.seed,
            )

            if plot_substrate:
                sim_curve, sub_curve = _run_simulation_no_viz(config, return_substrate=True)
            else:
                sim_curve = _run_simulation_no_viz(config, return_substrate=False)
                sub_curve = None

            _save_growth_curve(
                sim_curve, exp_interp,
                int(row["N0"]), row["P_grow"], row["P_div"],
                row["MSE"], sim_dir / "curva_crecimiento.png",
                init_substrate=sub_val,
                P_maint=pm_val,
                starve_threshold=st_val,
                time_sim=time_sim_min,
                time_exp=time_exp_min,
                raw_exp_pop=exp_pop,
            )

            if plot_substrate and sub_curve is not None:
                _save_growth_and_substrate_curve(
                    sim_curve, sub_curve, exp_interp,
                    int(row["N0"]), row["P_grow"], row["P_div"],
                    row["MSE"], sim_dir / "curva_crecimiento_sustrato.png",
                    init_substrate=sub_val,
                    P_maint=pm_val,
                    starve_threshold=st_val,
                    time_sim=time_sim_min,
                    time_exp=time_exp_min,
                    raw_exp_pop=exp_pop,
                )

            if snapshot_interval > 0:
                _save_snapshots(config, sim_dir / "snapshots", snapshot_interval)

            with open(sim_dir / "metadata.json", "w") as f:
                json.dump({
                    "rank": rank,
                    "N0": int(row["N0"]),
                    "P_grow": float(row["P_grow"]),
                    "P_div": float(row["P_div"]),
                    "init_substrate": sub_val,
                    "P_maint": pm_val,
                    "starve_threshold": st_val,
                    "MSE": float(row["MSE"]),
                }, f, indent=2)

    # Guardar peores N
    if save_worst_n > 0:
        worst_dir = output_dir / f"worst_{save_worst_n}"
        worst_dir.mkdir(parents=True, exist_ok=True)
        for rank, (_, row) in enumerate(df.tail(save_worst_n).iterrows(), start=1):
            sub_val = float(row.get("init_substrate", config_template.init_substrate))
            pm_val = float(row.get("P_maint", config_template.P_maint))
            st_val = int(row.get("starve_threshold", config_template.starve_threshold))
            sim_dir = worst_dir / f"rank_{rank:02d}_N0-{row['N0']}_Pg-{row['P_grow']}_Pd-{row['P_div']}_S0-{sub_val}_Pm-{pm_val}_St-{st_val}"
            sim_dir.mkdir(exist_ok=True)

            config = SimulationConfig(
                L=config_template.L, steps=config_template.steps,
                init_cells=config_template.init_cells,
                init_substrate=sub_val,
                N0=int(row["N0"]), P_grow=row["P_grow"], P_div=row["P_div"],
                P_maint=pm_val,
                starve_threshold=st_val,
                seed=config_template.seed,
            )

            if plot_substrate:
                sim_curve, sub_curve = _run_simulation_no_viz(config, return_substrate=True)
            else:
                sim_curve = _run_simulation_no_viz(config, return_substrate=False)
                sub_curve = None

            _save_growth_curve(
                sim_curve, exp_interp,
                int(row["N0"]), row["P_grow"], row["P_div"],
                row["MSE"], sim_dir / "curva_crecimiento.png",
                init_substrate=sub_val,
                P_maint=pm_val,
                starve_threshold=st_val,
                time_sim=time_sim_min,
                time_exp=time_exp_min,
                raw_exp_pop=exp_pop,
            )

            if plot_substrate and sub_curve is not None:
                _save_growth_and_substrate_curve(
                    sim_curve, sub_curve, exp_interp,
                    int(row["N0"]), row["P_grow"], row["P_div"],
                    row["MSE"], sim_dir / "curva_crecimiento_sustrato.png",
                    init_substrate=sub_val,
                    P_maint=pm_val,
                    starve_threshold=st_val,
                    time_sim=time_sim_min,
                    time_exp=time_exp_min,
                    raw_exp_pop=exp_pop,
                )

    if verbose:
        print(f"Curvas y snapshots guardados en {output_dir}")