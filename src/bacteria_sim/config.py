"""
Configuración estructurada para una simulación del autómata celular.

Este módulo no sabe nada de cyclopts ni de cómo se ejecuta
la simulación. Solo representa los parámetros como datos y sirve para ejecutar pruebas aisladas en run.py.
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

    # Límites razonables: evitan consumos absurdos de memoria/CPU y valores
    # que harían fallar el modelo con errores poco claros.
    MAX_L = 2000

    def __post_init__(self):
        """Valida los parámetros y falla pronto con un mensaje claro."""
        if not 3 <= self.L <= self.MAX_L:
            raise ValueError(f"L debe estar entre 3 y {self.MAX_L} (recibido: {self.L})")
        if self.steps < 1:
            raise ValueError(f"steps debe ser >= 1 (recibido: {self.steps})")
        if not 0 <= self.N0 <= 8:
            raise ValueError(f"N0 debe estar entre 0 y 8 (recibido: {self.N0})")
        for name in ("init_cells", "init_substrate", "P_grow", "P_div"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} debe estar entre 0 y 1 (recibido: {value})")
        if self.save_interval < 0:
            raise ValueError(f"save_interval debe ser >= 0 (recibido: {self.save_interval})")
        if self.update_interval < 0:
            raise ValueError(f"update_interval debe ser >= 0 (recibido: {self.update_interval})")

    def to_dict(self) -> dict:
        """Convierte la configuración en un diccionario."""
        return {k: v for k, v in self.__dict__.items()}