# Simulación de crecimiento bacteriano de E. coli K-12 MG1655 en condiciones estándar 

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Descripción

Este proyecto implementa un **autómata celular bidimensional** para simular el crecimiento de E. coli en condiciones estándar (37°C, medio LB, pH = 7,0).

## Base científica

- **Modelo estocástico basado en la literatura científica** → La probabilidad de crecimiento (`P_grow`) y probabilidad de división (`P_div`) controlan la fase de lag y de crecimiento; para controlar la fase estacionaria se usa la variable (`N0`) que representa la inhibición espacial.
- **Sustrato y gradientes de concentración** → difusión browniana y consumo local por células en crecimiento. Gracias al consumo local de nutrientes de las bacterias se genera un gradiente de concentración natural.
- **Reglas celulares**:
  - `0` = vacío
  - `1` = célula en división
  - `2` = célula en crecimiento
- **Inhibición espacial**: una célula en división pasa a crecimiento si el número de vecinos supera el umbral `N0`.

## Instalación

### Requisitos
- Python 3.12 o superior.
- [`uv`](https://docs.astral.sh/uv/) para la gestión del entorno y dependencias.

### Pasos

1. Clonar el repositorio:
  ```bash
  git clone https://github.com/savendanobo-ui/bacteria_sim.git
  cd bacteria_sim
  ```

2. Sincronizar entorno con uv:
  ```bash
  uv sync
  ```

3. Activar el entorno:
  ```bash
  source .venv/bin/activate # En linux/MAC # .venv\Scripts\activate en Windows
  ```

4. Uso:
  ```bash
  python run.py 
  ```

Puede alterar los diferentes argumentos, use 
  ```bash
  python run.py --help #para ver los argumentos disponibles

  #Argumentos disponibles
  --steps #STEPS	Número de pasos de simulación	
  --size #SIZE	Tamaño del grid (SIZE x SIZE)	
  --save #SAVE	Guardar último frame como PNG	
  --interval #INTERVAL	Milisegundos entre pasos de animación	
  --no-metrics	#Oculta la gráfica de evolución	
  --seed #SEED	Semilla para reproducibilidad	
  --save-interval #N	Guardar imagen cada N pasos (0 = no guardar)	
  --output-dir #DIR	Carpeta destino para imágenes guardadas	
  ```

Limitaciones conocidas:
- El sustrato se difunde aleatoriamente (sin gradientes de concenración macroscópicos ni quimiotaxis)
- El modulo de calibración de parametros aun esta en desarrollo
- La visualización puede volverse lenta para grids de mas de 200x200

Distribuido bajo licencia MIT. Consulta el archivo LICENSE para mas información.