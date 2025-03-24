from dataclasses import dataclass, field
import numpy as np
import nibabel as nib
import logging
import os
from src.utils.file_operations import FileHandler
from src.graph_processing.skeleton_processor import SkeletonProcessor
from skimage.morphology import skeletonize

@dataclass
class SegmentationProcessor:
    """
    A class for processing 3D medical image segmentations, including binarization,
    skeletonization, and mesh generation.

    Attributes
    ----------
    data : np.ndarray
        3D array containing the segmentation data.
    affine : np.ndarray
        Affine transformation matrix of the NIfTI image.
    shape : tuple
        Shape of the segmentation data.
    header : nib.Nifti1Header
        Header information of the NIfTI image.
    labels : np.ndarray
        Array of unique labels in the segmentation data.
    binary_data : dict
        Dictionary of binary masks for each label.
    skeleton_data : dict
        Dictionary of skeletonized data for each label.
    point_clouds : dict
        Dictionary of point clouds for each label.
    ordered_points : dict
        Dictionary of ordered points for each label's skeleton.
    """
    mesh_data: dict = field(default_factory=dict)  # To store mesh for each label
    data: np.ndarray = field(default=None)
    affine: np.ndarray = field(default=None)
    shape: tuple = field(default=None)
    header: nib.Nifti1Header = field(default=None)
    labels: np.ndarray = field(default_factory=lambda: np.array([]))
    binary_data: dict = field(default_factory=dict)
    skeleton_data: dict = field(default_factory=dict)
    point_clouds: dict = field(default_factory=dict)
    ordered_points: dict = field(default_factory=dict)
    binary_images: dict = field(default_factory=dict)

    @classmethod
    def from_nifti(cls, file_path: str, custom_labels=None):
        """
        Create an instance from a NIfTI file.

        Parameters
        ----------
        file_path : str
            Path to the NIfTI file.

        Returns
        -------
        SegmentationProcessor
            An instance of SegmentationProcessor with loaded data.
        """
        img = FileHandler.load_nifti(file_path)
        data = img.get_fdata()
        shape = data.shape
        affine = img.affine
        header = img.header
        logging.info(f"Loaded NIfTI file from {file_path}")
        processor = cls(data=data, affine=affine, header=header, shape=shape)
        if custom_labels is not None:
            processor.set_custom_labels(custom_labels)
        return processor
    
    def set_custom_labels(self, custom_labels):
        # make sure custom_labels is numpy array
        if isinstance(custom_labels, list):
            self.labels = np.array(custom_labels, dtype=np.int32)
        else:
            # make sure the np array is int32
            self.labels = np.array(custom_labels).astype(np.int32)
        
        exiting_labels = np.unique(self.data).astype(np.int32)
        for label in self.labels:
            if label not in exiting_labels:
                logging.warning(f"Label {label} specified but not found in the data.")

        logging.info(f"Only use specified labels: {self.labels} instead of all labels")



    def extract_labels(self):
        """
        Extract unique labels from the data (excluding background 0).

        Returns
        -------
        None
        """
        self.labels = np.unique(self.data)
        self.labels = self.labels[self.labels != 0]
        # labels extracted from load_nifti is automatically 
        # numpy.float64 without additional specification
        # hence we need to convert label to integer here
        self.labels = self.labels.astype(np.int32)
        # here labels are int32. maybe they should be unsigned int? make sure i am consistent
        logging.info(f"Found labels: {self.labels}")

    def binarize_labels(self):
        """
        Binarize the data for each label and apply morphological closing.

        Returns
        -------
        None
        """
        #from skimage.morphology import closing, ball

        if self.labels.size == 0:
            self.extract_labels()
        else:
            logging.info(f'Already defined label {self.labels}')
        for label in self.labels:
            binary_mask = (self.data == label).astype(np.uint8)
            self.binary_data[label] = binary_mask
            logging.info(f"Binarized label {label}. Number of voxels: {np.sum(binary_mask)}")

    def sobel_filter(self):
        """
        Apply Sobel filter to the data.

        Returns
        -------
        None
        """
        from scipy.ndimage import sobel

        for label, binary_image in self.binary_data.items():
            # Apply Sobel filter in each direction
            sobel_x = sobel(binary_image, axis=0)
            sobel_y = sobel(binary_image, axis=1)
            sobel_z = sobel(binary_image, axis=2)

            # Combine the gradients to get the overall edge map
            edge_map = sobel_x + sobel_y + sobel_z
            binary_image = binary_image * (edge_map == 0)

            # Save the binary image with Sobel filter applied
            self.binary_data[label] = binary_image
            logging.info(f"Applied Sobel filter to label {label}")

    def generate_skeleton(self):
        """
        Generate the skeleton for each label's binary data.

        Returns
        -------
        None
        """
        if not self.binary_data:
            self.binarize_labels()

        for label, binary_image in self.binary_data.items():
            # Use medial axis skeletonization or skeletonize_3d
            skeleton = skeletonize(binary_image)
            self.skeleton_data[label] = skeleton.astype(np.uint8)
            logging.info(f"Skeletonized label {label}. Number of skeleton voxels: {np.sum(skeleton)}")

    def process_skeletons(self, output_dir, base_filename):
        """
        Traverse the skeleton for each label and save ordered points.

        Parameters
        ----------
        output_dir : str
            Directory to save the ordered points.
        base_filename : str
            Base filename for saving the ordered points.

        Returns
        -------
        None
        """
        if not self.skeleton_data:
            self.generate_skeleton()

        for label, skeleton in self.skeleton_data.items():
            if np.sum(skeleton) == 0:
                logging.warning(f"Skeleton for label {label} is empty. Skipping.")
                continue
            skeleton_processor = SkeletonProcessor(skeleton)
            skeleton_processor.remove_outliers()
            skeleton_processor.find_endpoints()
            skeleton_processor.traverse_skeleton()
            skeleton_processor.save_ordered_points(output_dir, label, base_filename)
