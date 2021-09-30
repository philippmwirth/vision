import os
import sys

from benchmark_utils.plogger import ALBenchmarkPlogger


def main(json_file: str, metric_name: str):
    """Creates plots of the logs stored in the json_file.

    If save_figure is True, the plot will be saved in the directory of the 
    json_file.

    Args:
        json_file:
            Path to the json file containing the benchmarking logs.

    """
    # create a plogger
    al_benchmark_plogger = ALBenchmarkPlogger(filename=json_file)

    # create benchmark plots with confidence intervals
    al_benchmark_plogger.plot_all_content_with_confidence_intervals(
        metric_name=metric_name, save_figure=True,
    )


if __name__ == '__main__':


    metrics = [
        'AP IoU=0.50:0.95',
        'AP IoU=0.50',
        'AP IoU=0.75',
        'AP IoU=0.50:0.95 small',
        'AP IoU=0.50:0.95 medium',
        'AP IoU=0.50:0.95 large',
        'AR IoU=0.50:0.95',
        'AR IoU=0.50:0.95',
        'AR IoU=0.50:0.95',
        'AR IoU=0.50:0.95 small',
        'AR IoU=0.50:0.95 medium',
        'AR IoU=0.50:0.95 large',
    ]

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('log_file', type=str)
    parser.add_argument('--metric_name', choices=metrics, default='AP IoU=0.50')

    args = parser.parse_args()
    
    main(args.log_file, args.metric_name)