"""
Modulo de simulación de crecimiento bacteriano mediante automatas celulares.

Reglas:
- Dos matrices: G (estados 0,1,2) y S (sustrato 0,1)
- Difusion de sustrato (movimiento browniano simple)
- Crecimiento y division para inhibicion espacial (umbral N0)
- Parametros calibrados para gravedad normal y microgravedad (se deben añadir metodos para estos parametros)
"""

import numpy as np
from typing import Tuple, Optional

class BacteriaCellularAutomaton:
    """
    Automata celular para simular crecimiento bacteriano en condiciones
    de gravedad normal o microgravedad
    """

    def __init__(self,
                    L: int = 200, #tamaño del grid
                    microgravity: bool = False, #true para micg, false g normal
                    init_cells: float = 0.01, #fracción inicial de celdas en estado 2
                    init_substrate: float = 0.5): #fracción inicial de celdas con sustrato

                    self.L = L
                    self.microgravity = microgravity

                    #parametros segun condicion
                    if microgravity:
                        self.N0 = 5 #umbral de inhibicion
                        self.P_div = 0.6 #probabilidad de division
                        self.P_cluster = 0.45 #preferencia por clusters
                    else: #condiciones de gravedad normal
                        self.N0 = 3
                        self.P_div = 0.2 #ESTOS PARAMETROS NECESITAN FUNCIONES MATEMATICAS QUE LOS JUSTIFIQUEN
                        self.P_cluster = 0.15 

                    #inicializar matrices
                    #G: 0=vacia, 1=division, 2=crecimiento
                    self.G = np.zeros((L,L), dtype=np.int8)
                    #S: 0=sin sustrato, 1=con sustrato
                    self.S = np.zeros((L,L), dtype=np.int8)

                    #Poblacion inicial (celulas en estado 2: crecimiento)
                    num_cells = int(L*L*init_cells)
                    cell_indices = np.random.choice(L*L, num_cells, replace=False)
                    self.G.flat[cell_indices] = 2

                    #poblar sustrato
                    num_sub = int(L*L*init_substrate)
                    sub_indices = np.random.choice(L*L, num_sub, replace=False)
                    self.S.flat[sub_indices] = 1

                    #contador de tiempo 
                    self.time_step = 0 #variable inicializada para contar los pasos

    def get_neighbors(self, i: int, j: int) -> np.ndarray:
        """
        Retornar los indices de los 8 vecinos de Moore (condiciones de borde periodicas)
        """
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj ==0:
                    continue
                ni = (i+di) % self.L
                nj = (j+dj) % self.L
                neighbors.append((ni, nj))
        return np.array(neighbors) 

    def diffuse_substrate(self):
        """
        Fase 1: Difusion de particulas de sustrato.
        """

        #encontrar posiciones con sustrato
        pos_substrate = np.argwhere(self.S == 1)
        #mezclar para orden aleatorio
        np.random.shuffle(pos_substrate)

        #Matriz temporal para marcar celdas que ya recibieron una particula
        #(evita que dos particulas caigan en la misma celda en un mismo paso)
        moved_to = np.zeros_like(self.S, dtype=bool)

        for i, j in pos_substrate:
            #elegir direccion aleatoria entre los 8 vecinos
            neighbors = self.get_neighbors(i,j)
            np.random.shuffle(neighbors)
            for ni, nj in neighbors:
                if self.S[ni, nj] == 0 and not moved_to[ni, nj]:
                    #mover particula
                    self.S[i,j] = 0
                    self.S[ni, nj] = 1
                    moved_to[ni, nj] = True
                    break #solo se mueve una vez por particula

    def count_neighbor_cells(self, i: int, j: int) -> int:
        """
        Cuenta el numero total de celulas vecinas (estado 1 o 2) en el vecindario de Moore
        """
        total= 0
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni = (i + di) % self.L
                nj = (j + dj) % self.L
                if  self.G[ni, nj] in (1, 2):
                    total += 1
        return total

    def update_cells(self):
        """
        Fase 2: Actualización de estados celulares según reglas.
        Versión corregida con división asegurada.
        """
        new_G = self.G.copy()
        new_S = self.S.copy()
        
        # Contador de divisiones (para depuración)
        divisions = 0
        
        for i in range(self.L):
            for j in range(self.L):
                if self.G[i, j] == 2:  # Célula en crecimiento
                    if self.S[i, j] == 1:
                        # Consume sustrato y pasa a división
                        new_G[i, j] = 1
                        new_S[i, j] = 0
                    else:
                        new_G[i, j] = 2
                
                elif self.G[i, j] == 1:  # Célula en división
                    n_vis = self.count_neighbor_cells(i, j)
                    if n_vis > self.N0:
                        # Inhibición espacial: pasa a crecimiento
                        new_G[i, j] = 2
                    else:
                        # Intenta dividirse
                        if np.random.random() < self.P_div:
                            # Buscar vecinos vacíos (en el estado actual, no en new_G)
                            neighbors = self.get_neighbors(i, j)
                            empty_neighbors = []
                            for ni, nj in neighbors:
                                if self.G[ni, nj] == 0:  # vacía en el estado actual
                                    empty_neighbors.append((ni, nj))
                            
                            if empty_neighbors:
                                # Elegir un vecino vacío al azar (sin preferencia de clusters por simplicidad)
                                ni, nj = empty_neighbors[np.random.randint(len(empty_neighbors))]
                                new_G[ni, nj] = 2  # Nueva célula en crecimiento
                                divisions += 1
                                # La célula madre permanece en división (new_G[i,j] ya es 1)
                            # Si no hay vecinos vacíos, no se divide y permanece en 1
                # Si G[i,j]==0, no cambia
        
        # Actualizar matrices
        self.G = new_G
        self.S = new_S
        
        # Opcional: imprimir divisiones cada cierto paso
        if divisions > 0 and self.time_step % 50 == 0:
            print(f"Paso {self.time_step}: {divisions} nuevas células creadas")

    def step(self):
          """
          ejecuta un paso de tiempo completo: Difusión de sustrato + actualización celular
          """

          self.diffuse_substrate()
          self.update_cells()
          self.time_step += 1

    def run(self, steps: int, callback=None):
        """
        Ejecutar simulacion durante un numero dado de pasos

        Parametros:
        steps : int, numero de pasos de tiempo
        callback: callable, puede servir para guardar snapshots
        """
        for _ in range(steps):
            self.step()
            if callback:
                callback(self)

    def get_cell_count(self) -> int:
        #devuelve el numero total de celulas (estado 1 y 2)
        return np.sum((self.G == 1) | (self.G == 2))

    def get_growing_cells(self) -> int:
        #numero de células en crecimiento (estado 2)
        return np.sum(self.G == 2)

    def get_dividing_cells(self) -> int:
        #numero de celulas en división (estado 1)
        return np.sum(self.G == 1)

    def get_substrate_amount(self) -> int:
        #cantidad total de sustrato disponible
        return np.sum(self.S == 1)                                      



            