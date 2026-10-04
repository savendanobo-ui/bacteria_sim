"""
Lógica de simulación del autómata celular.

Este módulo ejecuta una simulación a partir de una SimulationConfig.
No sabe nada de argparse, cyclopts ni de cómo se parsean los argumentos.
"""

from pathlib import Path
import numpy as np

from bacteria_sim.config import SimulationConfig
from bacteria_sim.model import BacteriaCellularAutomaton
from bacteria_sim.visualization import BacteriaVisualization


def run_simulation(config: SimulationConfig) -> BacteriaCellularAutomaton:
    """
    Ejecuta una simulación completa con visualización.

    Parameters
    ----------
    config : SimulationConfig
        Configuración de la simulación.

    Returns
    -------
    BacteriaCellularAutomaton
        El modelo después de la simulación.
    """
    # Semilla
    if config.seed is not None:
        np.random.seed(config.seed)

    # Preparar directorio de salida
    output_dir = None
    save_interval = None
    if config.save_interval > 0:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(exist_ok=True)
        save_interval = config.save_interval
        print(f"Guardando imágenes cada {save_interval} pasos en '{output_dir}/'")
    else:
        print("No se guardarán imágenes (use --save-interval para activar)")

    # Crear modelo
    model = BacteriaCellularAutomaton(
        L=config.L,
        init_cells=config.init_cells,
        init_substrate=config.init_substrate,
        N0=config.N0,
        P_grow=config.P_grow,
        P_div=config.P_div,
    )

    # Crear visualización
    viz = BacteriaVisualization(
        model,
        update_interval=config.update_interval,
        show_metrics=config.show_metrics,
        output_dir=output_dir,
        save_interval=save_interval,
    )

    print(f"Iniciando simulación con L={config.L}, steps={config.steps}")

    # Ejecutar
    viz.animate_simulation(steps=config.steps)

    print("\n--- Resultados finales ---")
    print(f"Total células: {model.get_cell_count()}")
    print(f"Sustrato remanente: {model.get_substrate_amount()}")

    return model