#!/usr/bin/env python3
import argparse
from pathlib import Path
from bacteria_sim.model import BacteriaCellularAutomaton
from bacteria_sim.visualization import BacteriaVisualization

def main():
    parser = argparse.ArgumentParser(description="Simulación de crecimiento bacteriano en microgravedad")
    parser.add_argument("--microgravity", action="store_true", help="Activar microgravedad")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--save", type=str, default=None, help="Guardar último frame como PNG")
    parser.add_argument("--interval", type=int, default=50, help="ms entre pasos")
    parser.add_argument("--no-metrics", action="store_true", help="Ocultar gráfica de métricas")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=0, help="Guardar imagen cada N pasos (0 = no guardar)")
    parser.add_argument("--output-dir", type=str, default="output_simulacion", help="Carpeta para imágenes guardadas")
    
    args = parser.parse_args()
    
    if args.seed is not None:
        import numpy as np
        np.random.seed(args.seed)
    
    # Crear carpeta de salida si es necesario
    output_dir = None
    save_interval = None
    if args.save_interval > 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        save_interval = args.save_interval
        print(f"Guardando imágenes cada {save_interval} pasos en '{output_dir}/'")
    else:
        print("No se guardarán imagenes (puede usar --save-interval para activar)")

    model = BacteriaCellularAutomaton(L=args.size, microgravity=args.microgravity)
    
    viz = BacteriaVisualization(
        model,
        update_interval=args.interval,
        show_metrics=not args.no_metrics,
        output_dir=output_dir,
        save_interval=save_interval
    )
    
    print(f"Iniciando simulación en {'MICROGRAVEDAD' if args.microgravity else 'GRAVEDAD NORMAL'}")
    viz.animate_simulation(
        steps=args.steps,
        save=(args.save is not None),
        filename=args.save if args.save else "simulation.png"
    )
    
    print(f"\n--- Resultados finales ---")
    print(f"Total células: {model.get_cell_count()}")
    print(f"Sustrato remanente: {model.get_substrate_amount()}")

if __name__ == "__main__":
    main()