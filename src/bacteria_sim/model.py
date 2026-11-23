"""
Módulo de simulación de crecimiento bacteriano mediante autómatas celulares.

Modelo basado en la propuesta:
- Grid bidimensional L x L con condiciones de borde periódicas.
- Dos matrices independientes: G (estados 0,1,2) y S (sustrato 0,1).
- Difusión de sustrato por movimiento browniano con exclusión.
- Parámetros: N0 (umbral de inhibición), P_grow (crecimiento → división),
  P_div (división exitosa).
"""

import numpy as np
from typing import Optional


class BacteriaCellularAutomaton:
    """
    Autómata celular para simular el crecimiento de E. coli K-12 MG1655
    en condiciones estándar (37 °C, medio LB, pH 7.0).

    Reglas:
        - G[i,j] = 0: celda vacía
        - G[i,j] = 1: celda en división
        - G[i,j] = 2: celda en crecimiento
        - G[i,j] = 3: celda en fase estacionaria (arresto por inanición)
        - S[i,j] = 0: sin sustrato
        - S[i,j] = 1: con una partícula de sustrato
    """

    def __init__(
        self,
        L: int = 200,
        init_cells: float = 0.01,
        init_substrate: float = 0.5,
        N0: int = 3,
        P_grow: float = 0.25,
        P_div: float = 0.20,
        P_maint: float = 0.0,
        starve_threshold: int = 3,
        verbose: bool = True,
    ):
        """
        Parámetros:
            L (int): Tamaño del grid (L x L).
            init_cells (float): Fracción inicial de celdas con bacterias.
            init_substrate (float): Fracción inicial de celdas con sustrato.
            N0 (int): Umbral de inhibición espacial.
            P_grow (float): Probabilidad de transición crecimiento → división.
            P_div (float): Probabilidad de división exitosa.
            P_maint (float): Probabilidad de consumo de mantenimiento celular basal.
            starve_threshold (int): Pasos consecutivos sin sustrato antes del arresto de crecimiento (G=3).
            verbose (bool): Mostrar mensajes informativos.
        """
        self.L = L
        self.N0 = N0
        self.P_grow = P_grow
        self.P_div = P_div
        self.P_maint = P_maint
        self.starve_threshold = starve_threshold
        self.verbose = verbose

        # Inicializar matrices
        self.G = np.zeros((L, L), dtype=np.int8)  # Estados celulares
        self.S = np.zeros((L, L), dtype=np.int8)  # Sustrato
        self.starve_time = np.zeros((L, L), dtype=np.int16)  # Tiempo de inanición acumulado

        # Población inicial: bacterias en estado 2 (crecimiento)
        num_cells = int(L * L * init_cells)
        cell_indices = np.random.choice(L * L, num_cells, replace=False)
        self.G.flat[cell_indices] = 2

        # Poblar sustrato
        num_sub = int(L * L * init_substrate)
        sub_indices = np.random.choice(L * L, num_sub, replace=False)
        self.S.flat[sub_indices] = 1

        # Contador de tiempo
        self.time_step = 0

    def get_neighbors(self, i: int, j: int) -> list:
        """Retorna los índices de los 8 vecinos de Moore (borde periódico)."""
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni = (i + di) % self.L
                nj = (j + dj) % self.L
                neighbors.append((ni, nj))
        return neighbors

    def diffuse_substrate(self):
        """
        Fase 1: Difusión del sustrato por movimiento browniano con exclusión.

        Cada partícula intenta moverse a una celda vecina vacía. Si el destino
        está ocupado o ya recibió otra partícula en el mismo paso, permanece.
        """
        pos_substrate = np.argwhere(self.S == 1)
        np.random.shuffle(pos_substrate)

        moved_to = np.zeros_like(self.S, dtype=bool)

        for i, j in pos_substrate:
            neighbors = self.get_neighbors(i, j)
            np.random.shuffle(neighbors)
            for ni, nj in neighbors:
                if self.S[ni, nj] == 0 and not moved_to[ni, nj]:
                    self.S[i, j] = 0
                    self.S[ni, nj] = 1
                    moved_to[ni, nj] = True
                    break

    def count_neighbor_cells(self, i: int, j: int) -> int:
        """Cuenta vecinos ocupados (estados 1, 2 o 3) en el vecindario de Moore."""
        total = 0
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni = (i + di) % self.L
                nj = (j + dj) % self.L
                if self.G[ni, nj] in (1, 2, 3):
                    total += 1
        return total

    def update_cells(self):
        """
        Fase 2: Actualización síncrona de estados celulares según las reglas.

        Reglas:
            - Mantenimiento (Opción A): si P_maint > 0, cada célula viva (G in 1, 2, 3)
              con prob. P_maint consume 1 partícula de sustrato local (en su celda o vecina).
            - Crecimiento (G=2): si hay sustrato en su celda, lo consume y resetea su
              contador de inanición. Con prob. P_grow pasa a división (G=1).
              Si no hay sustrato, incrementa su contador de inanición. Si alcanza
              starve_threshold pasos consecutivos sin alimento, entra en estado
              estacionario / arresto de crecimiento (G=3).
            - División (G=1): si vecinos > N0 → inhibición, vuelve a crecimiento.
              Si vecinos <= N0, con prob. P_div se divide: madre → crecimiento,
              hija → crecimiento en vecino vacío. Si falla, madre → crecimiento.
            - Estacionaria (G=3): célula metabólicamente en reposo; no crece ni se
              divide, manteniendo la morfología de la colonia en meseta estacionaria.
            - Vacía (G=0): permanece vacía (puede ser ocupada por una división).
        """
        new_G = self.G.copy()
        new_S = self.S.copy()
        new_starve = self.starve_time.copy()
        divisions = 0

        # Mantenimiento celular basal (Opción A)
        if self.P_maint > 0:
            for i in range(self.L):
                for j in range(self.L):
                    if self.G[i, j] in (1, 2, 3):
                        if np.random.random() < self.P_maint:
                            if new_S[i, j] == 1:
                                new_S[i, j] = 0
                            else:
                                neighbors = self.get_neighbors(i, j)
                                for ni, nj in neighbors:
                                    if new_S[ni, nj] == 1:
                                        new_S[ni, nj] = 0
                                        break

        for i in range(self.L):
            for j in range(self.L):
                if self.G[i, j] == 2:  # Crecimiento
                    if new_S[i, j] == 1:
                        new_S[i, j] = 0  # Consume sustrato
                        new_starve[i, j] = 0  # Resetea contador de inanición
                        if np.random.random() < self.P_grow:
                            new_G[i, j] = 1  # Pasa a división
                        else:
                            new_G[i, j] = 2  # Sigue en crecimiento
                    else:
                        new_starve[i, j] += 1
                        if new_starve[i, j] >= self.starve_threshold:
                            new_G[i, j] = 3  # Entra en fase estacionaria (arresto)
                        else:
                            new_G[i, j] = 2  # Sigue en crecimiento esperando sustrato

                elif self.G[i, j] == 1:  # División
                    n_vis = self.count_neighbor_cells(i, j)
                    if n_vis > self.N0:
                        new_G[i, j] = 2  # Inhibición espacial
                        new_starve[i, j] = 0
                    else:
                        if np.random.random() < self.P_div:
                            neighbors = self.get_neighbors(i, j)
                            empty_neighbors = [
                                (ni, nj) for ni, nj in neighbors
                                if self.G[ni, nj] == 0
                            ]
                            if empty_neighbors:
                                ni, nj = empty_neighbors[
                                    np.random.randint(len(empty_neighbors))
                                ]
                                new_G[ni, nj] = 2  # Célula hija en crecimiento
                                new_starve[ni, nj] = 0
                                divisions += 1
                            new_G[i, j] = 2  # Madre vuelve a crecimiento
                            new_starve[i, j] = 0
                        else:
                            new_G[i, j] = 2  # División fallida → crecimiento
                            new_starve[i, j] = 0

                elif self.G[i, j] == 3:  # Fase estacionaria (arresto de crecimiento)
                    new_G[i, j] = 3

        self.G = new_G
        self.S = new_S
        self.starve_time = new_starve

        if self.verbose and divisions > 0 and self.time_step % 50 == 0:
            print(f"Paso {self.time_step}: {divisions} nuevas células creadas")

    def step(self):
        """Ejecuta un paso completo: difusión + actualización celular."""
        self.diffuse_substrate()
        self.update_cells()
        self.time_step += 1

    def run(self, steps: int, callback=None):
        """Ejecuta la simulación durante `steps` pasos."""
        for _ in range(steps):
            self.step()
            if callback:
                callback(self)

    # Métodos auxiliares para métricas
    def get_cell_count(self) -> int:
        return int(np.sum((self.G >= 1) & (self.G <= 3)))

    def get_growing_cells(self) -> int:
        return int(np.sum(self.G == 2))

    def get_dividing_cells(self) -> int:
        return int(np.sum(self.G == 1))

    def get_stationary_cells(self) -> int:
        return int(np.sum(self.G == 3))

    def get_substrate_amount(self) -> int:
        return int(np.sum(self.S == 1))