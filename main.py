"""
Entry point for running the full 3D → 2D embedding pipeline from the command line.

This script calls `run_embedding_algorithm` from the `embedding_algorithm` package
with a few configurable parameters, and prints a short summary of the result.
"""

import argparse
import sys
from embedding_algorithm import run_embedding_algorithm

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the embedding run.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the full embedding pipeline for a random 3D unit-disk graph (UDG) "
            "instance and construct a 2D layout that preserves its topology."
        )
    )

    parser.add_argument(
        "--num-nodes",
        type=int,
        default=200,
        help="Number of base nodes to generate with the Sobol sequence (default: 200).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None).",
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=1.0,
        help="Disk radius used to build the 2D UDG (default: 1.0).",
    )

    parser.add_argument(
        "--height",
        type=float,
        default=1.0,
        help="z-coordinate for apex nodes in the 3D UDG (default: 1.0).",
    )

    parser.add_argument(
        "--plot-2d",
        action="store_true",
        help="If set, plot the initial 2D UDG instance.",
    )

    parser.add_argument(
        "--plot-3d",
        action="store_true",
        help="If set, plot the 3D UDG with apex nodes.",
    )

    parser.add_argument(
        "--time-check",
        action="store_true",
        help="If set, print timing information for the embedding process.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, print detailed status messages during component attachment.",
    )

    # Per ora non esponiamo `border` a linea di comando: lasciamo che la funzione
    # usi il suo esagono di default (border=None).
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    try:
        (
            sobol_points,
            G_2d,
            G_3d,
            coords_3d,
            final_pos,
            all_nodes,
            apex_list,
            total_time,
        ) = run_embedding_algorithm(
            num_nodes=args.num_nodes,
            seed=args.seed,
            border=None,
            radius=args.radius,
            height=args.height,
            plot_2d=args.plot_2d,
            plot_3d=args.plot_3d,
            time_check=args.time_check,
            verbose=args.verbose,
        )
    except ValueError as exc:
        print(f"[ERROR] Embedding failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # --- Piccolo riepilogo testuale ---
    print("Embedding completed successfully.")
    print(f"- Number of base nodes (Sobol): {len(sobol_points)}")
    print(
        f"- 2D graph: {G_2d.number_of_nodes()} nodes, "
        f"{G_2d.number_of_edges()} edges"
    )
    print(
        f"- 3D graph: {G_3d.number_of_nodes()} nodes, "
        f"{G_3d.number_of_edges()} edges"
    )
    print(f"- Apex nodes: {len(apex_list)}")
    print(f"- Final embedded nodes: {len(all_nodes)}")

    if args.time_check and total_time is not None:
        print(f"- Total time: {total_time:.3f} s")

if __name__ == "__main__":
    main()

