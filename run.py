#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
from bacteria_sim.model import BacteriaCellularAutomaton
from bacteria_sim.visualization import BacteriaVisualization


def main():
    parser = argparse.ArgumentParser(
        description="Simulación de crecimiento de E. coli K-12 MG1655"
    )
    parser.add_argument("--steps", type=int, default=300,
                        help="Número de pasos de simulación")
    parser.add_argument("--size", type=int, default=200,
                        help="Tamaño del grid (SIZE x SIZE)")
    parser.add_argument("--save", type=str, default=None,
                        help="Guardar último frame como PNG")
    parser.add_argument("--interval", type=int, default=50,
                        help="ms entre pasos de animación")
    parser.add_argument("--no-metrics", action="store_true",
                        help="Ocultar gráfica de métricas")
    parser.add_argument("--seed", type=int, default=None,
                        help="Semilla para reproducibilidad")
    parser.add_argument("--save-interval", type=int, default=0,
                        help="Guardar imagen cada N pasos (0 = no guardar)")
    parser.add_argument("--output-dir", type=str, default="output_simulacion",
                        help="Carpeta para imágenes guardadas")
    parser.add_argument("--N0", type=int, default=3,
                        help="Umbral de inhibición espacial")
    parser.add_argument("--P-grow", type=float, default=0.25,
                        help="Probabilidad de transición crecimiento → división")
    parser.add_argument("--P-div", type=float, default=0.20,
                        help="Probabilidad de división exitosa")

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    output_dir = None
    save_interval = None
    if args.save_interval > 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        save_interval = args.save_interval
        print(f"Guardando imágenes cada {save_interval} pasos en '{output_dir}/'")
    else:
        print("No se guardarán imágenes (use --save-interval para activar)")

    model = BacteriaCellularAutomaton(
        L=args.size,
        N0=args.N0,
        P_grow=args.P_grow,
        P_div=args.P_div,
    )

    viz = BacteriaVisualization(
        model,
        update_interval=args.interval,
        show_metrics=not args.no_metrics,
        output_dir=output_dir,
        save_interval=save_interval,
    )

    print("Iniciando simulación en condiciones estándar (37 °C, LB, pH 7.0)")
    viz.animate_simulation(
        steps=args.steps,
        save=(args.save is not None),
        filename=args.save if args.save else "simulation.png",
    )

    print("\n--- Resultados finales ---")
    print(f"Total células: {model.get_cell_count()}")
    print(f"Sustrato remanente: {model.get_substrate_amount()}")


if __name__ == "__main__":
    main()