# main.py
import os
import argparse
import logging
import numpy as np
import csv
from src.graph_processing.segmentation_processor import SegmentationProcessor
from src.graph_processing.skeleton_processor import SkeletonProcessor
from src.parametrized_curves.curve_class import ParametricCurve, CurveAnalyzer, SplineOptimizer
from src.parametrized_curves.curve_class import Plotter
import time

def parse_arguments():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Convert segmentation to skeleton, fit curves to points, and estimate tortuosity measures.")
    parser.add_argument(
        "-t", "--workdir",
        type=str,
        required=True,
        help="Working directory that contains the input NIfTI file."
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="TOF_eICAB_CW.nii.gz",
        help="Input NIfTI file name (default: TOF_eICAB_CW.nii.gz). This NIFTI is a segmentation with labels 1 and 2 representing the left and right ICA, respectively."
    )
    parser.add_argument(
        "-l", "--labels",
        nargs="*",  # needs multiple arguments as input
        default=["1", "2"],  # default labels 1 and 2 (left and right ICA, respectively)
        help = "Choose the input seg label you care about (In the original segmentations, the values range from 1-18. You may also use your own segmentations and respective labels)."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help = "Enables verbose mode"
    )
    
    # Add other arguments as needed
    return parser.parse_args()



def initialize_csv(output_dir):
    """
    Initialize the CSV file for saving skeleton metrics.

    Parameters
    ----------
    output_dir : str
        Directory where the CSV file will be saved.
    """
    csv_file = os.path.join(output_dir, 'skeleton_metrics.csv')
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=[
            'Label', 'Start Point', 'End Point', 'Skeleton Size', 
            'Total Curvature', 'Mean Squared Curvature', 'RMS Curvature',
            'AOC', 'Parametric Curve Length', 'Final Loss', 'RMSE',
        ])
        writer.writeheader()  # Write the header only once
    logging.info(f"Initialized CSV file at {csv_file}")


def save_metrics_to_csv(output_dir, label, start_point, end_point, skeleton_size, total_curvature, mean_squared_curvature, rms_curvature, aoc, curve_length, final_loss, rmse):
    """
    Save computed metrics to the CSV file.

    Parameters
    ----------
    output_dir : str
        Directory where the CSV file is saved.
    metrics : dict
        Dictionary containing the metrics data.
    """ 
   
   
    csv_file = os.path.join(output_dir, 'skeleton_metrics.csv')
    file_exists = os.path.isfile(csv_file)

    # Prepare data to write
    row_data = {
        'Label': label,
        'Start Point': start_point,
        'End Point': end_point,
        'Skeleton Size': skeleton_size,
        'Total Curvature': total_curvature,
        'Mean Squared Curvature': mean_squared_curvature,
        'RMS Curvature': rms_curvature,
        'AOC': aoc,
        'Parametric Curve Length': curve_length,
        'Final Loss': final_loss,
        'RMSE': rmse,
    }

    # Write the CSV file
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()  # Write header if the file doesn't exist yet
        writer.writerow(row_data)

    logging.info(f"Metrics saved to {csv_file}")


def process_label(label, output_dir):
        """
        Compute values on a specific label by loading a set of ordered points,
        fitting a cubic spline, and calculating tortuosity metrics.
        Note that since the fitting is done on the discrete/grid space of segmentation matrix. 
        the unit of spline depends on the segmentation input image itself. 
        Hence, scale invariant metrics are important. 

        Parameters
        ----------
        label : str
            The label to process.
        output_dir : str
            Directory where results will be saved.
        """
        ordered_points_file = os.path.join(output_dir, f"ordered_edge_label_{label}.0_ordered_points.npy")
        ordered_points = np.load(ordered_points_file)
        
        # Split the ordered points into x, y, z components
        # It is 2D np array, each 1D array represent a np.argwhere of point that belongs to skeleton 
        x = ordered_points[:, 0]
        y = ordered_points[:, 1]
        z = ordered_points[:, 2]

        # Create a Parametric Curve from the points
        curve = ParametricCurve(x, y, z)
        analyzer = CurveAnalyzer(curve)

        points_per_unit_length = 10  
        eval_points = int(curve.total_length * points_per_unit_length)
        optimizer = SplineOptimizer(analyzer)
        final_loss = optimizer.fit_with_multiple_initial_smoothings(
            k=3, max_iterations=200, tolerance=1e-6, eval_points=eval_points
        )

        total_curvature = analyzer.curvature_calculator.total_curvature()
        mean_squared_curvature = analyzer.curvature_calculator.mean_squared_curvature()
        rms_curvature = analyzer.curvature_calculator.rms_curvature()
        aoc = analyzer.calculate_aoc()

        # Log output
        logging.info("Processing completed successfully.")
        logging.info(f"Total Curvature (Spline-Based): {total_curvature:.6f}")
        logging.info(f"Mean Squared Curvature (Spline-Based): {mean_squared_curvature:.6f}")
        logging.info(f"RMS Curvature (Spline-Based): {rms_curvature:.6f}")
        logging.info(f"AOC (Spline-Based): {aoc:.6f}")
        logging.info(f"Final Loss using RMSE+RMS Curvature: {final_loss['final_loss']:.6f}")
        logging.info(f"Final RMSE: {final_loss['rmse']:.6f}")

        # After curvature calculations
        start_point = (x[0], y[0], z[0])
        end_point = (x[-1], y[-1], z[-1])

        # Size of skeleton is the number of points that consitutes it 
        skeleton_size = len(x)  
        curve_length = curve.total_length
        save_metrics_to_csv(output_dir, 
                            label, 
                            start_point, 
                            end_point, 
                            skeleton_size, 
                            total_curvature, 
                            mean_squared_curvature, 
                            rms_curvature, 
                            aoc, 
                            curve_length, 
                            final_loss['final_loss'], 
                            final_loss['rmse'])

        # Plot and save the plots
        curve_plot_path = os.path.join(output_dir, f"curve_plot_label_{label}.png")
        curvature_plot_path = os.path.join(output_dir, f"curvature_plot_label_{label}.png")
        residual_plot_path = os.path.join(output_dir, f"residual_plot_{label}.png")

        analyzer.plot_curve(curve_plot_path)

        analyzer.spline_fitter.plot_residuals_vs_arc_length(residual_plot_path)


def main():
    args = parse_arguments()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
    # record start time
    start_time = time.time()
    try:
        # Construct file paths
        input_path = os.path.join(args.workdir, args.input)
        output_dir = args.workdir
        base_filename = os.path.splitext(os.path.basename(args.input))[0]
        
        initialize_csv(output_dir)

        # Initialize SegmentationProcessor
        seg_processor = SegmentationProcessor.from_nifti(input_path)

        # Process labels
        seg_processor.extract_labels()
        seg_processor.binarize_labels()

        # generate skeleton
        seg_processor.generate_skeleton()
        seg_processor.process_skeletons(output_dir=output_dir, base_filename="ordered_edge")

        for label in args.labels:
            logging.info(f"Processing label {label}...")
            process_label(label, output_dir)

        logging.info("Processing completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
    finally:
        total_time = time.time() - start_time
        logging.info(f"time taken to process this participant is {total_time:.2f} seconds")

if __name__ == "__main__":
    main()