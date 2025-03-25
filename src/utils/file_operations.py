# utils/file_operations.py
import pandas as pd
import os
import pickle
import numpy as np
import nibabel as nib
import logging

class FileHandler:

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
    
