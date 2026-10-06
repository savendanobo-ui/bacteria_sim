"""Tests de validación de SimulationConfig y utilidades de la CLI."""

import pytest

from bacteria_sim.cli_calibration import _parse_range
from bacteria_sim.config import SimulationConfig


def test_config_por_defecto_es_valida():
    config = SimulationConfig()
    assert config.L == 200
    assert "MAX_L" not in config.to_dict()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"L": 2},
        {"L": SimulationConfig.MAX_L + 1},
        {"steps": 0},
        {"N0": -1},
        {"N0": 9},
        {"init_cells": -0.1},
        {"init_cells": 1.5},
        {"init_substrate": 2.0},
        {"P_grow": -0.01},
        {"P_div": 1.01},
        {"save_interval": -1},
        {"update_interval": -5},
    ],
)
def test_config_rechaza_valores_invalidos(kwargs):
    with pytest.raises(ValueError):
        SimulationConfig(**kwargs)


def test_config_acepta_limites():
    SimulationConfig(L=3, steps=1, N0=0, P_grow=0.0, P_div=1.0,
                     init_cells=0.0, init_substrate=1.0)
    SimulationConfig(L=SimulationConfig.MAX_L, N0=8)


def test_parse_range_con_tres_valores():
    assert _parse_range("0.1,0.3,0.1") == [0.1, 0.2, 0.3]


def test_parse_range_lista_explicita():
    assert _parse_range("0.1,0.5") == [0.1, 0.5]


@pytest.mark.parametrize("texto", ["0.1,0.5,0", "0.1,0.5,-0.1"])
def test_parse_range_rechaza_paso_no_positivo(texto):
    with pytest.raises(ValueError):
        _parse_range(texto)
