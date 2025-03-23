# utils/file_operations.py
import pandas as pd
import os
import pickle
import numpy as np
import nibabel as nib
import logging

class FileHandler:

    @staticmethod
    def export_to_csv(map: dict, filename: str, output_dir: str):
        df = pd.DataFrame(list(map.items()), columns=['Edge', filename])
        df['Edge'] = df['Edge'].apply(lambda x: f"{x[0]}, {x[1]}" if isinstance(x, tuple) else x)

        filepath = os.path.join(output_dir, f"{filename}.csv")
        df.to_csv(filepath, index=False)
        logging.info(f"Exported {filename} to {filepath}")


    @staticmethod
    def export_all_metrics_to_csv(curvature_metrics: dict, output_dir: str, output_filename: str):
        """
        Export all curvature metrics and AOC into a single CSV file.
        
        Parameters:
            - curvature_metrics (dict): Dictionary where keys are metric names and values are dicts mapping edges to values.
            - output_dir (str): Directory to save the CSV file.
            - output_filename (str): Name of the output CSV file.
        """
        
        all_edges = set()
        for metrics in curvature_metrics.values():
            all_edges.update(metrics.keys())
        
        sorted_edges = sorted(all_edges, key=lambda x: (x[0], x[1]))
     
        data = {'Edge': [f"{s}, {e}" for (s, e) in sorted_edges]}        

        for metric_name, metrics in curvature_metrics.items():
            data[metric_name] = [metrics.get(edge, np.nan) for edge in sorted_edges]

        df = pd.DataFrame(data)
        
        filepath = os.path.join(output_dir, output_filename)
        df.to_csv(filepath, index=False)
        print(f"Exported all metrics to {filepath}")


    @staticmethod
    def merge_csv(file_path: list, output_path: str, key: str = 'Edge'):
        """
        Merge multiple CSV files based on a common key.
        
        Parameters:
            file_paths (list): List of CSV file paths to merge.
            output_path (str): Path to save the merged CSV.
            key (str): The column name to merge on (default: 'Edge').
        """
        try:
            dfs = [pd.read_csv(file) for file in file_path]
            merged_df = dfs[0]
            for df in dfs[1:]:
                merged_df = pd.merge(merged_df, df, on=key, how='left')
            merged_df.to_csv(output_path, index=False)
            print(f"Merged CSV saved to {output_path}.")
        except Exception as e:
            print(f"Error merging CSV files: {e}")
            raise
    
    @staticmethod
    def load_nifti(file_path: str) -> nib.Nifti1Image:
        """
        Load a NIfTI file.
        
        Parameters:
            - file_path (str): Path to the NIfTI file.
        
        Returns:
            - nib.Nifti1Image: Loaded NIfTI image.
        """
        if not os.path.isfile(file_path):
            logging.error(f"NIfTI file '{file_path}' does not exist.")
            raise FileNotFoundError(f"NIfTI file '{file_path}' does not exist.")
        try:
            img = nib.load(file_path)
            logging.info(f"NIfTI file loaded from {file_path}")
            return img
        except Exception as e:
            logging.error(f"Failed to load NIfTI file '{file_path}': {e}")
            raise
    
    @staticmethod
    def save_nifti(data: np.ndarray, affine: np.ndarray, header: nib.Nifti1Header, file_path: str):
        """
        Save data as a NIfTI file.
        
        Parameters:
            - data (np.ndarray): Image data to save.
            - affine (np.ndarray): Affine transformation matrix.
            - header (nib.Nifti1Header): NIfTI header.
            - file_path (str): Path to save the NIfTI file.
        """
        try:
            img = nib.Nifti1Image(data, affine, header)
            nib.save(img, file_path)
            logging.info(f"NIfTI file saved to {file_path}")
        except Exception as e:
            logging.error(f"Failed to save NIfTI file '{file_path}': {e}")
            raise
    
    @staticmethod
    def save_numpy(array, file_path):
        np.save(file_path, array)
        print(f"Saved numpy file to {file_path}")