"""Rebuild plots from an existing run folder."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, help='Path to runs/<name>')
    args = parser.parse_args()

    run = Path(args.run)
    plots_dir = run / 'plots'
    utils.ensure_dir(plots_dir)

    history = utils.load_json(run / 'history.json', default=None)
    if history:
        utils.plot_training_curves(history, plots_dir / 'training_curves.png')
        print(f'Saved training curves: {plots_dir / "training_curves.png"}')

    for stem in ['test_metrics', 'evaluate_metrics', 'valid_metrics']:
        metrics = utils.load_json(run / f'{stem}.json', default=None)
        if metrics:
            utils.plot_final_metrics(metrics, plots_dir / f'{stem}.png')
            print(f'Saved metrics bar chart: {plots_dir / f"{stem}.png"}')

    summary = utils.load_json(run / 'summary.json', default=None)
    if summary and 'test' in summary:
        utils.plot_final_metrics(summary['test'], plots_dir / 'test_metrics.png')
        print(f'Saved summary test metrics chart: {plots_dir / "test_metrics.png"}')


if __name__ == '__main__':
    main()
