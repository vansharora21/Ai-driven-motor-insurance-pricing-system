from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_loader import DataValidationError
from src.scenario_simulation import simulate_scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scenario analysis for a single policy row.")
    parser.add_argument("--input", required=True, help="Path to a CSV file containing exactly one policy row.")
    parser.add_argument(
        "--output",
        default="results/premium_reports/scenario_analysis.csv",
        help="Where to save the scenario analysis report CSV.",
    )
    parser.add_argument(
        "--plot-output",
        default="results/plots/scenario_premium_curves.png",
        help="Where to save the optional scenario premium curve plot.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting and only save the CSV report.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    plot_output_path = Path(args.plot_output)

    try:
        input_df = pd.read_csv(input_path)
        report = simulate_scenarios(
            input_df,
            output_path=output_path,
            save_plot=not args.no_plot,
            plot_path=plot_output_path,
        )
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"Invalid CSV format in {input_path}: {exc}") from exc
    except DataValidationError as exc:
        raise ValueError(f"Input validation failed for {input_path}: {exc}") from exc

    print(f"Saved scenario analysis to {output_path.resolve()}")
    if not args.no_plot:
        print(f"Saved scenario premium curves to {plot_output_path.resolve()}")
    print(report[["scenario_name", "final_premium", "premium_delta_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
