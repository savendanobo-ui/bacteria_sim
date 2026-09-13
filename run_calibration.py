#!/usr/bin/env python3
"""
Script para ejecutar la calibración con guardado de curvas y snapshots.
"""

import argparse
from pathlib import Path
from bacteria_sim.calibration import load_experimental_data, calibrate, validate


def main():
    parser = argparse.ArgumentParser(description="Calibración del autómata celular")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, default="resultados_calibracion.csv")
    parser.add_argument("--output-dir", type=str, default="resultados_calibracion")
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-top", type=int, default=10,
                        help="Guardar curvas y snapshots de las N mejores")
    parser.add_argument("--save-worst", type=int, default=5,
                        help="Guardar curvas de las N peores")
    parser.add_argument("--snapshot-interval", type=int, default=50,
                        help="Guardar snapshot cada N pasos")
    args = parser.parse_args()

    print(f"Cargando datos desde {args.data}")
    exp_time, exp_pop = load_experimental_data(Path(args.data))

    df = calibrate(
        experimental_data=(exp_time, exp_pop),
        L=args.L,
        steps=args.steps,
        seed=args.seed,
        output_csv=Path(args.output),
        output_dir=Path(args.output_dir),
        save_top_n=args.save_top,
        save_worst_n=args.save_worst,
        snapshot_interval=args.snapshot_interval,
    )

    best = df.iloc[0].to_dict()
    print("\n--- Validación con los mejores parámetros ---")
    metrics = validate(
        best_params=best,
        experimental_data=(exp_time, exp_pop),
        L=args.L,
        steps=args.steps,
        seed=args.seed,
    )
    print(f"MSE  = {metrics['MSE']:.6f}")
    print(f"R²   = {metrics['R2']:.6f}")
    print(f"MAE  = {metrics['MAE']:.6f}")


if __name__ == "__main__":
    main()