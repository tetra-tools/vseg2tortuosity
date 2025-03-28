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
import nibabel as nib
from scipy import ndimage

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
        "-s", "--sobel",
        action="store_true",
        help = "Apply Sobel filter to the input segmentation."
    )
    parser.add_argument(
        "-m", "--max_final_loss",
        type=int,
        default=10,
        help = "Max Final Loss (combined RMSE with RMS CUrvature) tolerated for spline fit, if Final Loss exceed this, we erode segmentation and auto rerun"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help = "Enables verbose mode"
    )
    # args.max_final_loss
    
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
            'AOC', 'Skeleton Length', 'Spline Length', 'Final Loss', 'RMSE', 'Eroded'
        ])
        writer.writeheader()  # Write the header only once
    logging.info(f"Initialized CSV file at {csv_file}")


def save_metrics_to_csv(output_dir, label, start_point, end_point, skeleton_size,
                        total_curvature, mean_squared_curvature, rms_curvature, aoc,
                        skeleton_length, spline_length, final_loss, rmse, eroded=0):
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
        'Skeleton Length': skeleton_length,
        "Spline Length": spline_length,
        'Final Loss': final_loss,
        'RMSE': rmse,
        'Eroded': eroded
    }

    # Write the CSV file
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()  # Write header if the file doesn't exist yet
        writer.writerow(row_data)

    logging.info(f"Metrics saved to {csv_file}")


def process_label(label, output_dir, base_filename, eroded=0, segmentation_points=None):
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
        ordered_points_file = os.path.join(output_dir, f"{base_filename}_label_{label}_ordered_points.npy")
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
        start_point = f"({start_point[0]}, {start_point[1]}, {start_point[2]})"
        end_point = f"({end_point[0]}, {end_point[1]}, {end_point[2]})"



        # Size of skeleton is the number of points that consitutes it 
        skeleton_size = len(x)  
        skeleton_length = curve.total_length # from .diff in points of skeleton
        spline_length = analyzer.spline_fitter.total_length # from integrating on fitted spline
        #print(f"skeleton_length {skeleton_length}, spline_length{spline_length}")
        save_metrics_to_csv(output_dir, 
                            label, 
                            start_point,
                            end_point, 
                            skeleton_size, 
                            total_curvature, 
                            mean_squared_curvature, 
                            rms_curvature, 
                            aoc, 
                            skeleton_length,
                            spline_length,
                            final_loss['final_loss'], 
                            final_loss['rmse'],
                            eroded=eroded)
        metrics = {
        "Start Point": start_point,
        "End Point": end_point,
        "Skeleton Size": skeleton_size,
        "Skeleton Length": f"{skeleton_length:.2f}",
        "Spline Length": f"{spline_length:.2f}",
        "Total Curvature": f"{total_curvature:.4f}",
        "Mean Squared Curvature": f"{mean_squared_curvature:.4f}",
        "RMS Curvature": f"{rms_curvature:.4f}",
        "AOC": f"{aoc:.4f}",
        "Final Loss": f"{final_loss['final_loss']:.4f}",
        "RMSE": f"{final_loss['rmse']:.4f}",
        "Eroded": "Yes" if eroded else "No"
        }

        # Plot and save the plots
        #curve_plot_path = os.path.join(output_dir, f"curve_plot_label_{label}_eroded_{eroded}.png")
        #curvature_plot_path = os.path.join(output_dir, f"curvature_plot_label_{label}_eroded_{eroded}.png")
        residual_plot_path = os.path.join(output_dir, f"residual_plot_{label}_eroded_{eroded}.png")
        #curvature_plot_path = os.path.join(output_dir, f"curvature_plot_label_{label}_eroded_{eroded}.png")
        #analyzer.plot_curve(curve_plot_path)

        analyzer.spline_fitter.plot_residuals_vs_arc_length(residual_plot_path)
   
        
        prefix = "eroded_" if eroded else ""
        interactive_plot_path = os.path.join(output_dir, f"{prefix}interactive_curve_label_{label}.html")
        analyzer.plot_interactive(save_path=interactive_plot_path, 
                              title=f"3D Vessel Curve - Label {label}" + (" (Eroded)" if eroded else ""),
                              metrics=metrics,
                              segmentation_points=segmentation_points)
        # return the final loss so we can check whether we need to erode and rerun

        return final_loss['final_loss']



def erode_label(workdir, input_filename, label):

    # the input label should be int
    label_value = int(label)

    img = nib.load(os.path.join(workdir, input_filename))
    data = img.get_fdata()

    binary_mask = (data==label_value).astype(np.uint8)
    eroded_mask = ndimage.binary_erosion(binary_mask).astype(binary_mask.dtype)

    new_data = np.zeros_like(data, dtype=np.int32)
    for unique_label in np.unique(data):
        if unique_label != 0 and unique_label != label_value:
            new_data[data == unique_label] = int(unique_label)

    new_data[eroded_mask == 1] = label_value # be mindful here should be label value instead of just 1 
    # because i need to save as segmentation and in new segmentation, each postition need to have value of the label

    eroded_filename = f"eroded_label_{label_value}_{input_filename}"
    eroded_path = os.path.join(workdir, eroded_filename)
    # the erroded_segmentation contain all original labels, it only modify the content of target label to be erroded
    # doing np.unque on erroded should give us same set as doing it on original data
    eroded_img = nib.Nifti1Image(new_data, img.affine, img.header)
    nib.save(eroded_img, eroded_path)
    logging.info(f"Saved eroded segmentation for label {label_value} to {eroded_path}")
    return eroded_path



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

        
        initialize_csv(output_dir)

        # Initialize SegmentationProcessor
        seg_processor = SegmentationProcessor.from_nifti(input_path, custom_labels=args.labels)

        # Process labels
        seg_processor.binarize_labels()
        if args.sobel:
            seg_processor.apply_sobel()

        # Generate skeleton
        seg_processor.generate_skeleton()
        seg_processor.process_skeletons(output_dir=output_dir, base_filename="ordered_edge")



        high_loss_labels = {}
        for label in args.labels:
            logging.info(f"Processing label {label}...")
            segmentation_points = seg_processor.extract_segmentation_points(int(label))
            final_loss = process_label(label,
                                       output_dir,
                                       base_filename="ordered_edge",
                                       eroded=0,
                                       segmentation_points=segmentation_points)
            
            if final_loss > args.max_final_loss:
                # maybe make high_loss_labels a list? either way works
                high_loss_labels[label] = final_loss
                logging.info(f"label {label} has high loss of {final_loss}, will erode and run again")
            
            
        if high_loss_labels:
            # if this is not a empty map
            logging.info(f"applying erosion to {len(high_loss_labels)} labels: {list(high_loss_labels.keys())}")
            for high_loss_label, _ in high_loss_labels.items():
                eroded_path = erode_label(workdir = args.workdir, input_filename = args.input, label=high_loss_label)
                if eroded_path:
                    eroded_seg_processor = SegmentationProcessor.from_nifti(eroded_path, custom_labels=list(high_loss_label))
                    eroded_seg_processor.generate_skeleton()
                    eroded_seg_processor.process_skeletons(output_dir=output_dir, base_filename="eroded_ordered_edge")
                    erroded_segmentation_points = eroded_seg_processor.extract_segmentation_points(int(high_loss_label))
                    process_label(high_loss_label, output_dir, "eroded_ordered_edge", eroded=1, segmentation_points=erroded_segmentation_points)

        logging.info("Processing completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
    finally:
        total_time = time.time() - start_time
        logging.info(f"time taken to process this participant is {total_time:.2f} seconds")

if __name__ == "__main__":
    main()