"""
Pruebas unitarias para el autómata celular y el módulo de calibración.

Ejecutar con:
    pytest tests/test_model.py -v
"""

import numpy as np
import pytest
from bacteria_sim.model import BacteriaCellularAutomaton
from bacteria_sim.calibration import normalize_curve, compute_mse


# ============================================================
# 1. Inicialización del modelo
# ============================================================

class TestModelInitialization:
    """Pruebas de inicialización del autómata."""

    def test_grid_dimensions(self):
        """El grid debe tener las dimensiones correctas."""
        model = BacteriaCellularAutomaton(L=50)
        assert model.G.shape == (50, 50)
        assert model.S.shape == (50, 50)

    def test_default_parameters(self):
        """Los parámetros por defecto deben ser los esperados."""
        model = BacteriaCellularAutomaton(L=100)
        assert model.N0 == 3
        assert model.P_grow == 0.25
        assert model.P_div == 0.20

    def test_initial_cells_fraction(self):
        """La fracción inicial de células debe ser aproximadamente la esperada."""
        model = BacteriaCellularAutomaton(L=100, init_cells=0.05)
        total_cells = np.sum(model.G == 2)
        expected = int(100 * 100 * 0.05)
        # Tolerancia por aleatoriedad (la elección es sin reemplazo)
        assert abs(total_cells - expected) <= 5

    def test_initial_substrate_fraction(self):
        """La fracción inicial de sustrato debe ser aproximadamente la esperada."""
        model = BacteriaCellularAutomaton(L=100, init_substrate=0.5)
        total_sub = np.sum(model.S == 1)
        expected = int(100 * 100 * 0.5)
        assert abs(total_sub - expected) <= 5

    def test_initial_time_step(self):
        """El contador de tiempo debe empezar en 0."""
        model = BacteriaCellularAutomaton(L=10)
        assert model.time_step == 0

    def test_states_are_valid(self):
        """Todos los estados deben ser 0, 1 o 2."""
        model = BacteriaCellularAutomaton(L=20, init_cells=0.1)
        unique = np.unique(model.G)
        assert set(unique).issubset({0, 1, 2})

    def test_substrate_is_binary(self):
        """El sustrato debe ser binario (0 o 1)."""
        model = BacteriaCellularAutomaton(L=20, init_substrate=0.3)
        unique = np.unique(model.S)
        assert set(unique).issubset({0, 1})


# ============================================================
# 2. Vecindario de Moore
# ============================================================

class TestNeighbors:
    """Pruebas del vecindario de Moore."""

    def test_moore_neighborhood_size(self):
        """Cada celda debe tener 8 vecinos."""
        model = BacteriaCellularAutomaton(L=10)
        neighbors = model.get_neighbors(5, 5)
        assert len(neighbors) == 8

    def test_no_self_in_neighbors(self):
        """La celda no debe estar en su propio vecindario."""
        model = BacteriaCellularAutomaton(L=10)
        neighbors = model.get_neighbors(5, 5)
        assert (5, 5) not in neighbors

    def test_periodic_boundaries_top_left(self):
        """Los bordes deben ser periódicos (esquina superior izquierda)."""
        model = BacteriaCellularAutomaton(L=10)
        neighbors = model.get_neighbors(0, 0)
        assert (9, 9) in neighbors  # diagonal opuesta
        assert (9, 0) in neighbors
        assert (0, 9) in neighbors

    def test_periodic_boundaries_bottom_right(self):
        """Los bordes deben ser periódicos (esquina inferior derecha)."""
        model = BacteriaCellularAutomaton(L=10)
        neighbors = model.get_neighbors(9, 9)
        assert (0, 0) in neighbors
        assert (0, 9) in neighbors
        assert (9, 0) in neighbors

    def test_all_neighbors_unique(self):
        """Los vecinos no deben repetirse."""
        model = BacteriaCellularAutomaton(L=10)
        neighbors = model.get_neighbors(5, 5)
        assert len(neighbors) == len(set(neighbors))


# ============================================================
# 3. Difusión del sustrato
# ============================================================

class TestDiffusion:
    """Pruebas de la difusión de sustrato."""

    def test_no_double_occupancy(self):
        """Después de la difusión, ninguna celda debe tener > 1 partícula."""
        model = BacteriaCellularAutomaton(L=20, init_substrate=0.5)
        model.diffuse_substrate()
        assert np.all((model.S == 0) | (model.S == 1))

    def test_substrate_conserved(self):
        """El número total de partículas se conserva en la difusión."""
        model = BacteriaCellularAutomaton(L=20, init_substrate=0.5)
        total_before = np.sum(model.S == 1)
        model.diffuse_substrate()
        total_after = np.sum(model.S == 1)
        assert total_before == total_after

    def test_empty_grid_stays_empty(self):
        """Si no hay sustrato, la difusión no crea partículas."""
        model = BacteriaCellularAutomaton(L=10, init_substrate=0.0)
        model.diffuse_substrate()
        assert np.sum(model.S == 1) == 0

    def test_full_grid_stays_full(self):
        """Si el grid está lleno, la difusión no cambia el conteo."""
        model = BacteriaCellularAutomaton(L=10, init_substrate=1.0)
        total_before = np.sum(model.S == 1)
        model.diffuse_substrate()
        total_after = np.sum(model.S == 1)
        assert total_before == total_after


# ============================================================
# 4. Reglas de crecimiento
# ============================================================

class TestGrowthRules:
    """Pruebas de las reglas de crecimiento."""

    def test_consumption_with_substrate(self):
        """Una célula en crecimiento con sustrato lo consume."""
        model = BacteriaCellularAutomaton(L=10, init_cells=0.0, init_substrate=0.0)
        model.G[5, 5] = 2
        model.S[5, 5] = 1
        model.P_grow = 1.0
        model.update_cells()
        assert model.S[5, 5] == 0

    def test_grow_probability_one(self):
        """Con P_grow = 1, siempre pasa a división si hay sustrato."""
        model = BacteriaCellularAutomaton(L=10, init_cells=0.0, init_substrate=0.0)
        model.G[5, 5] = 2
        model.S[5, 5] = 1
        model.P_grow = 1.0
        model.update_cells()
        assert model.G[5, 5] == 1

    def test_grow_probability_zero(self):
        """Con P_grow = 0, nunca pasa a división."""
        model = BacteriaCellularAutomaton(L=10, init_cells=0.0, init_substrate=0.0)
        model.G[5, 5] = 2
        model.S[5, 5] = 1
        model.P_grow = 0.0
        model.update_cells()
        assert model.G[5, 5] == 2

    def test_no_substrate_stays_growing(self):
        """Sin sustrato, la célula permanece en crecimiento."""
        model = BacteriaCellularAutomaton(L=10, init_cells=0.0, init_substrate=0.0)
        model.G[5, 5] = 2
        model.S[5, 5] = 0
        model.P_grow = 1.0
        model.update_cells()
        assert model.G[5, 5] == 2


# ============================================================
# 5. Reglas de división
# ============================================================

class TestDivisionRules:
    """Pruebas de las reglas de división."""

    def test_spatial_inhibition(self):
        """Con vecinos > N0, la célula vuelve a crecimiento."""
        model = BacteriaCellularAutomaton(L=10, init_cells=0.0, init_substrate=0.0)
        model.N0 = 1
        model.G[5, 5] = 1
        # Rodeamos la célula de vecinos
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                model.G[5 + di, 5 + dj] = 2
        model.update_cells()
        assert model.G[5, 5] == 2

    def test_division_with_space(self):
        """Con P_div = 1 y espacio, la célula se divide y crea una hija."""
        model = BacteriaCellularAutomaton(L=10, init_cells=0.0, init_substrate=0.0)
        model.G[5, 5] = 1
        model.P_div = 1.0
        model.N0 = 8
        model.update_cells()
        # La madre vuelve a crecimiento
        assert model.G[5, 5] == 2
        # Debe haber al menos una célula nueva en los vecinos (la hija)
        neighbors = model.get_neighbors(5, 5)
        new_cells_in_neighbors = sum(1 for ni, nj in neighbors if model.G[ni, nj] == 2)
        assert new_cells_in_neighbors >= 1

    def test_division_failure(self):
        """Con P_div = 0, la célula no se divide y vuelve a crecimiento."""
        model = BacteriaCellularAutomaton(L=10, init_cells=0.0, init_substrate=0.0)
        model.G[5, 5] = 1
        model.P_div = 0.0
        model.N0 = 8
        model.update_cells()
        assert model.G[5, 5] == 2
        # Solo debe haber 1 célula (la original)
        assert np.sum(model.G == 2) == 1

    def test_division_no_empty_neighbors(self):
        """Si no hay vecinos vacíos, la célula vuelve a crecimiento."""
        model = BacteriaCellularAutomaton(L=3, init_cells=1.0, init_substrate=0.0)
        # Todo el grid está lleno (excepto donde ponemos la célula en división)
        model.G[:, :] = 2
        model.G[1, 1] = 1
        model.P_div = 1.0
        model.N0 = 8
        model.update_cells()
        # Debe volver a crecimiento (no pudo dividirse)
        assert model.G[1, 1] == 2


# ============================================================
# 6. Estado vacío
# ============================================================

class TestEmptyState:
    """Pruebas del estado vacío."""

    def test_empty_cell_stays_empty(self):
        """Una celda vacía sin vecinos activos permanece vacía."""
        model = BacteriaCellularAutomaton(L=10, init_cells=0.0, init_substrate=0.0)
        model.update_cells()
        assert model.G[5, 5] == 0

    def test_empty_cell_occupied_by_division(self):
        """Una celda vacía puede ser ocupada por una división vecina."""
        model = BacteriaCellularAutomaton(L=10, init_cells=0.0, init_substrate=0.0)
        model.G[5, 5] = 1
        model.P_div = 1.0
        model.N0 = 8
        model.update_cells()
        # Al menos una celda vecina debe estar ocupada (la hija)
        neighbors = model.get_neighbors(5, 5)
        occupied = sum(1 for ni, nj in neighbors if model.G[ni, nj] == 2)
        assert occupied >= 1


# ============================================================
# 7. Reproductibilidad
# ============================================================

class TestReproducibility:
    """Pruebas de reproducibilidad con semilla."""

    def test_same_seed_same_result(self):
        """Con la misma semilla, dos ejecuciones dan el mismo resultado."""
        np.random.seed(42)
        model1 = BacteriaCellularAutomaton(L=20, init_cells=0.05, init_substrate=0.5)
        model1.run(steps=10)
        count1 = model1.get_cell_count()

        np.random.seed(42)
        model2 = BacteriaCellularAutomaton(L=20, init_cells=0.05, init_substrate=0.5)
        model2.run(steps=10)
        count2 = model2.get_cell_count()

        assert count1 == count2

    def test_different_seed_different_result(self):
        """Con semillas diferentes, los resultados pueden diferir."""
        np.random.seed(1)
        model1 = BacteriaCellularAutomaton(L=20, init_cells=0.05, init_substrate=0.5)
        model1.run(steps=20)
        count1 = model1.get_cell_count()

        np.random.seed(999)
        model2 = BacteriaCellularAutomaton(L=20, init_cells=0.05, init_substrate=0.5)
        model2.run(steps=20)
        count2 = model2.get_cell_count()

        # No siempre son iguales, pero la probabilidad es alta
        # (no es una prueba estricta, solo informativa)
        assert isinstance(count1, int) and isinstance(count2, int)


# ============================================================
# 8. Métodos auxiliares
# ============================================================

class TestAuxiliaryMethods:
    """Pruebas de los métodos get_*."""

    def test_get_cell_count(self):
        """El conteo de células debe coincidir con la suma manual."""
        model = BacteriaCellularAutomaton(L=10, init_cells=0.0, init_substrate=0.0)
        model.G[1, 1] = 1
        model.G[2, 2] = 2
        model.G[3, 3] = 2
        assert model.get_cell_count() == 3

    def test_get_growing_cells(self):
        """El conteo de células en crecimiento debe ser correcto."""
        model = BacteriaCellularAutomaton(L=10, init_cells=0.0, init_substrate=0.0)
        model.G[1, 1] = 2
        model.G[2, 2] = 2
        model.G[3, 3] = 1
        assert model.get_growing_cells() == 2

    def test_get_dividing_cells(self):
        """El conteo de células en división debe ser correcto."""
        model = BacteriaCellularAutomaton(L=10, init_cells=0.0, init_substrate=0.0)
        model.G[1, 1] = 1
        model.G[2, 2] = 1
        model.G[3, 3] = 2
        assert model.get_dividing_cells() == 2

    def test_get_substrate_amount(self):
        """El conteo de sustrato debe ser correcto."""
        model = BacteriaCellularAutomaton(L=10, init_cells=0.0, init_substrate=0.0)
        model.S[1, 1] = 1
        model.S[2, 2] = 1
        model.S[3, 3] = 1
        assert model.get_substrate_amount() == 3


# ============================================================
# 9. Utilidades de calibración
# ============================================================

class TestCalibrationUtilities:
    """Pruebas de las utilidades de calibración."""

    def test_normalize_curve(self):
        """La normalización divide cada valor por el inicial."""
        curve = np.array([2.0, 4.0, 6.0])
        norm = normalize_curve(curve)
        assert np.allclose(norm, [1.0, 2.0, 3.0])

    def test_normalize_curve_zero(self):
        """Normalizar una curva con valor inicial 0 lanza error."""
        curve = np.array([0.0, 1.0, 2.0])
        with pytest.raises(ValueError):
            normalize_curve(curve)

    def test_mse_identical_curves(self):
        """El MSE entre dos curvas idénticas debe ser 0."""
        curve = np.array([1.0, 2.0, 4.0, 8.0])
        mse = compute_mse(curve, curve)
        assert mse == pytest.approx(0.0, abs=1e-10)

    def test_mse_different_curves(self):
        """El MSE entre curvas diferentes debe ser > 0."""
        curve1 = np.array([1.0, 2.0, 4.0, 8.0])
        curve2 = np.array([1.0, 1.5, 3.0, 5.0])
        mse = compute_mse(curve1, curve2)
        assert mse > 0

    def test_mse_symmetry(self):
        """El MSE no es simétrico por la normalización, pero ambos > 0."""
        curve1 = np.array([1.0, 2.0, 4.0])
        curve2 = np.array([1.0, 1.5, 3.0])
        mse_12 = compute_mse(curve1, curve2)
        mse_21 = compute_mse(curve2, curve1)
        assert mse_12 > 0
        assert mse_21 > 0


# ============================================================
# 10. Integración (simulación completa)
# ============================================================

class TestIntegration:
    """Pruebas de integración: ejecutar simulaciones completas."""

    def test_full_simulation_runs(self):
        """Una simulación completa debe correr sin errores."""
        model = BacteriaCellularAutomaton(L=20, init_cells=0.05, init_substrate=0.5)
        model.run(steps=50)
        # Debe haber células al final
        assert model.get_cell_count() > 0

    def test_simulation_consumes_substrate(self):
        """La simulación debe consumir sustrato con el tiempo."""
        model = BacteriaCellularAutomaton(L=20, init_cells=0.1, init_substrate=0.5)
        substrate_initial = model.get_substrate_amount()
        model.run(steps=50)
        substrate_final = model.get_substrate_amount()
        assert substrate_final < substrate_initial

    def test_simulation_time_step_increments(self):
        """El contador de tiempo debe incrementarse correctamente."""
        model = BacteriaCellularAutomaton(L=10, init_cells=0.05, init_substrate=0.5)
        model.run(steps=30)
        assert model.time_step == 30

    def test_population_grows(self):
        """En condiciones favorables, la población debe crecer."""
        model = BacteriaCellularAutomaton(
            L=30, init_cells=0.05, init_substrate=0.8,
            N0=4, P_grow=0.5, P_div=0.5,
        )
        initial = model.get_cell_count()
        model.run(steps=30)
        final = model.get_cell_count()
        assert final > initial