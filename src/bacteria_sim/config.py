"""
Configuración estructurada para una simulación del autómata celular.

Este módulo no sabe nada de argparse, cyclopts ni de cómo se ejecuta
la simulación. Solo representa los parámetros como datos.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SimulationConfig:
    """Configuración completa para una simulación del autómata celular."""

    # Parámetros del grid
    L: int = 200

    # Parámetros de simulación
    steps: int = 300
    seed: Optional[int] = 42

    # Parámetros de inicialización
    init_cells: float = 0.01
    init_substrate: float = 0.5

    # Parámetros del modelo
    N0: int = 3
    P_grow: float = 0.25
    P_div: float = 0.20

    # Parámetros de visualización / guardado
    save_interval: int = 0
    output_dir: str = "output_simulacion"
    show_metrics: bool = True
    update_interval: int = 50

    #parametro para ver mensajes de depuración
    verbose: bool = True

    def to_dict(self) -> dict:
        """Convierte la configuración en un diccionario."""
        return {k: v for k, v in self.__dict__.items()}