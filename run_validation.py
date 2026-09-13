#!/usr/bin/env python3
"""
Script para validar el modelo calibrado con datos experimentales independientes.
"""

import argparse
from pathlib import Path
from bacteria_sim.calibration import load_experimental_data, validate


def main():
    parser = argparse.ArgumentParser(description="Validación del modelo")
    parser.add_argument("--data", type=str, required=True,
                        help="CSV con datos de validación")
    parser.add_argument("--N0", type=int, required=True)
    parser.add_argument("--P-grow", type=float, required=True)
    parser.add_argument("--P-div", type=float, required=True)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    exp_time, exp_pop = load_experimental_data(Path(args.data))

    metrics = validate(
        best_params={"N0": args.N0, "P_grow": args.P_grow, "P_div": args.P_div},
        experimental_data=(exp_time, exp_pop),
        L=args.L, steps=args.steps, seed=args.seed,
    )

    print(f"\n--- Resultados de validación ---")
    print(f"N0 = {args.N0}, P_grow = {args.P_grow}, P_div = {args.P_div}")
    print(f"MSE = {metrics['MSE']:.6f}")
    print(f"R²  = {metrics['R2']:.6f}")
    print(f"MAE = {metrics['MAE']:.6f}")


if __name__ == "__main__":
    main()