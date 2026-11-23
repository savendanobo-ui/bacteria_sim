# Simulación de crecimiento bacteriano de E. coli K-12 MG1655 en condiciones estándar 

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Descripción

Este proyecto implementa un **autómata celular bidimensional** para simular el crecimiento de E. coli en condiciones estándar (37°C, medio LB, pH = 7,0).

## Base científica

- **Modelo estocástico basado en la literatura científica:** La probabilidad de crecimiento (`P_grow`), la probabilidad de división (`P_div`) y (`P_maint`) controlan la fase de lag y de crecimiento; para controlar la fase estacionaria se usa la variable (`N0`) que representa la inhibición espacial y la variable (`starve_threshold`) que dispara el arresto replicativo irreversible de la célula hacia el estado quiescente.
- **Sustrato y gradientes de concentración:** difusión browniana y consumo local por células en crecimiento. Gracias al consumo local de nutrientes de las bacterias se genera un gradiente de concentración natural.
- **Reglas celulares**:
  - `0` = vacío
  - `1` = célula en crecimiento
  - `2` = célula en división
  - `3` = célula en fase estacionaria
- **Inhibición espacial**: una célula en división pasa a crecimiento si el número de vecinos supera el umbral `N0`.
- **Contador de inanición**: Variable interna caracteristico de cada bacteria que guarda la cantidad de pasos en los que no consume sustrato.

## Instalación

### Requisitos
- Python 3.13 o superior.
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
  uv run python run.py # para correr una simulación
  uv run python run_calibration.py # para realizar la calibración del modelo a partir de datos de /data/experimental; por ahora solo admite .csv
  ```

Puede alterar los diferentes argumentos, use 
  ```bash
  uv run python run.py --help #para ver los argumentos disponibles
	uv run python run_calibration.py --help #puede ver los comandos y argumentos disponibles
  uv run python run_calibration.py --comando --help #puede ver los argumentos disponibles para el comando (validate o calibrate)
  ```

Limitaciones conocidas:
- El sustrato se difunde aleatoriamente (sin gradientes de concenración macroscópicos ni quimiotaxis).
- La visualización puede volverse lenta para grids de mas de 200x200.
- Fase de construcción del modelo analitico predictivo pendiente.

Distribuido bajo licencia MIT. Consulta el archivo LICENSE para mas información.