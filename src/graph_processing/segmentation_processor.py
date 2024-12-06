from dataclasses import dataclass, field
import numpy as np
import nibabel as nib
import logging
import os
from src.utils.file_operations import FileHandler
from src.graph_processing.skeleton_processor import SkeletonProcessor
from skimage.morphology import skeletonize_3d, ball, closing
from skimage.measure import marching_cubes
import trimesh

@dataclass
class SegmentationProcessor:
    """
    A class for processing 3D medical image segmentations, including binarization,
    skeletonization, and mesh generation.

    Attributes
    ----------
    mesh_data : dict
        Dictionary storing mesh data (vertices, faces) for each label.
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
    def from_nifti(cls, file_path: str):
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
        return cls(data=data, affine=affine, header=header, shape=shape)

    def extract_labels(self):
        """
        Extract unique labels from the data (excluding background 0).

        Returns
        -------
        None
        """
        self.labels = np.unique(self.data)
        self.labels = self.labels[self.labels != 0]
        logging.info(f"Found labels: {self.labels}")

    def binarize_labels(self):
        """
        Binarize the data for each label and apply morphological closing.

        Returns
        -------
        None
        """
        from skimage.morphology import closing, ball

        if self.labels.size == 0:
            self.extract_labels()
        for label in self.labels:
            binary_mask = (self.data == label).astype(np.uint8)
            self.binary_data[label] = binary_mask
            logging.info(f"Binarized and closed label {label}. Number of voxels: {np.sum(binary_mask)}")

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
            skeleton = skeletonize_3d(binary_image)
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

    def create_mesh(self):
        """
        Create a mesh for each labels' binary data using marching cubes.

        Returns
        -------
        None
        """
        if not self.binary_data:
            self.binarize_labels()

        for label, binary_mask in self.binary_data.items():
            verts, faces, _, _ = marching_cubes(binary_mask, level=0.5)
            # Store the vertices and faces for visualization
            self.mesh_data[label] = {'verts': verts, 'faces': faces}
            logging.info(f"Mesh created for label {label}.")

    def save_mesh(self, output_dir, base_filename):
        """
        Save the mesh as an STL file for each label.

        Parameters
        ----------
        output_dir : str
            Directory to save the STL files.
        base_filename : str
            Base filename for the STL files.

        Returns
        -------
        None
        """
        if not self.mesh_data:
            self.create_mesh()

        for label, mesh in self.mesh_data.items():
            verts, faces = mesh['verts'], mesh['faces']
            output_path = os.path.join(output_dir, f"{base_filename}_label_{label}.stl")
            trimesh.Trimesh(vertices=verts, faces=faces).export(output_path)
            logging.info(f"Mesh for label {label} saved to {output_path}.")