"""
Interfaz de línea de comandos (CLI) para la simulación.

Este módulo se encarga ÚNICAMENTE de:
1. Parsear los argumentos de la línea de comandos.
2. Crear una SimulationConfig.
3. Llamar a run_simulation().

No contiene lógica de simulación.
"""

import cyclopts
from typing import Optional

from bacteria_sim.config import SimulationConfig
from bacteria_sim.simulation import run_simulation

app = cyclopts.App(
    help="Simulación de crecimiento de E. coli mediante autómatas celulares."
)


@app.default
def simulate(
    L: int = 200,
    steps: int = 300,
    seed: Optional[int] = 42,
    init_cells: float = 0.01,
    init_substrate: float = 0.5,
    N0: int = 3,
    P_grow: float = 0.25,
    P_div: float = 0.20,
    save_interval: int = 0,
    output_dir: str = "output_simulacion",
    show_metrics: bool = True,
    update_interval: int = 50,
):
    """Ejecuta una simulación individual del autómata celular."""
    config = SimulationConfig(
        L=L,
        steps=steps,
        seed=seed,
        init_cells=init_cells,
        init_substrate=init_substrate,
        N0=N0,
        P_grow=P_grow,
        P_div=P_div,
        save_interval=save_interval,
        output_dir=output_dir,
        show_metrics=show_metrics,
        update_interval=update_interval,
    )
    run_simulation(config)


if __name__ == "__main__":
    app()