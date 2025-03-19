# src/graph_processing/skeleton_processor.py
from scipy.ndimage import convolve
from itertools import product
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
import logging

class SkeletonProcessor:
    """
    A class for processing and traversing a 3D skeleton extracted from a segmentation.

    Attributes
    ----------
    skeleton : np.ndarray
        A 3D binary array representing the skeleton (1 for skeleton points, 0 for background).
    endpoints : list of tuple
        List of endpoints detected in the skeleton.
    ordered_points : list of tuple
        List of ordered points after skeleton traversal.
    """
    def __init__(self, skeleton):
        """
        Initialize the SkeletonProcessor with a 3D skeleton array.

        Parameters
        ----------
        skeleton : np.ndarray
            A 3D binary array representing the skeleton.
        """
        self.skeleton = skeleton
        self.endpoints = None
        self.ordered_points = None

    def remove_outliers(self):
        """
        Remove outlier points from the skeleton using a nearest neighbors approach.

        Outliers are defined as points whose distances to their nearest neighbors
        exceed a threshold based on the mean and standard deviation of neighbor distances.

        Returns
        -------
        None
        """

        skeleton_points = np.argwhere(self.skeleton == 1)

        if len(skeleton_points) < 3:
            logging.warning(f"Not enough skeleton points to remove outliers (found {len(skeleton_points)} points). Skipping outlier removal.")
            return

        # Fit the nearest neighbors model
        nbrs = NearestNeighbors(n_neighbors=3).fit(skeleton_points)
        distances, _ = nbrs.kneighbors(skeleton_points)

        # Calculate mean and standard deviation of distances to neighbors
        # Threshold is points that are 2 standard deviations beyond mean
        distances = distances[:, 1:3].flatten()
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + 2 * std_dist
    
        # Identify outliers based on the threshold
        outlier_mask = np.any(distances.reshape(-1, 2) > threshold, axis=1)
        removed_outliers = np.sum(outlier_mask)

        # Remove outlier points from the skeleton points
        skeleton_points = skeleton_points[~outlier_mask]

        # Rebuild the skeleton matrix from the cleaned points
        self.skeleton = np.zeros_like(self.skeleton)
        self.skeleton[tuple(skeleton_points.T)] = 1  # Set cleaned points back in the skeleton array

        logging.info(f"Removed {removed_outliers} outlier points from the skeleton.")


    def find_endpoints(self):
        """
        Identify endpoints of the skeleton based on neighbor count.

        An endpoint is defined as a skeleton point with only one neighbor.

        Returns
        -------
        None
        """
        if np.sum(self.skeleton) == 0:
            logging.warning("Skeleton is empty. Skipping.")
            return
        kernel = np.ones((3, 3, 3), dtype=int)
        kernel[1, 1, 1] = 0  
        neighbor_count = convolve(self.skeleton, kernel, mode='constant', cval=0)
        endpoints = np.argwhere((self.skeleton == 1) & (neighbor_count == 1))
        self.endpoints = [tuple(point) for point in endpoints]
        if len(self.endpoints) == 0:
            logging.warning("No endpoints found in the skeleton.")
        else:
            logging.info(f"Found {len(self.endpoints)} endpoints.")


    def traverse_skeleton(self):
        """
        Traverse the skeleton using a depth-first search (DFS) from one endpoint to another.

        Returns
        -------
        None
        """
        if self.endpoints is None or len(self.endpoints) < 2:
            logging.warning("Not enough endpoints to traverse the skeleton.")
            return
        # start, end = self.endpoints[:2]  # Assuming we take the first two endpoints
        # self.ordered_points = self._dfs_traversal(start, end)
        # logging.info(f"Skeleton traversal complete. Ordered {len(self.ordered_points)} points.")

        # evaluate all possible pairs from endpoints 
        # find best path from any pair of endpoints
        best_path = None
        best_path_length = 0

        # nested for loop to try all pair
        for(i, start) in enumerate(self.endpoints):
            for end in self.endpoints[i+1:]:
                path = self._dfs_traversal(start, end)

                if not path or path[-1] != end:
                    continue
                cur_length = self._estimate_path_length(path)
                if cur_length > best_path_length:
                    best_path_length= cur_length
                    best_path = path
        if best_path:
            self.ordered_points = best_path
            logging.info(f"Skeleton traversal complete. Selected best path with {len(self.ordered_points)} points.")
        else:
            logging.warning("No valid path find between any end points pairs")


    def _estimate_path_length(self, path):
        """
        Perform a path length estiamtion on a given path.

        Parameters
        ----------
        path : List
            Ordered list of points forming a path.
        Returns
        -------
        path length
            Estimated length of the input path.
        """

        if len(path) < 3:
            return 0 
        points = np.array(path)
        segments = np.diff(points, axis=0)
        segment_lengths = np.sqrt(np.sum(segments**2, axis=1))
        total_length = np.sum(segment_lengths)
        return total_length

    def _dfs_traversal(self, start, end):
        """
        Perform a depth-first search (DFS) traversal of the skeleton.

        Parameters
        ----------
        start : tuple
            Starting point of the traversal.
        end : tuple
            Ending point of the traversal.

        Returns
        -------
        list of tuple
            List of ordered points from the traversal.
        """

        # explore all 26 adjacent cells
        neighbors_offsets = [offset for offset in product([-1, 0, 1], repeat=3) if offset != (0, 0, 0)]
        visited = set()
        ordered_points = []
        stack = [(start, None)]  # Each element is (current_point, previous_point)

        while stack:
            current, prev = stack.pop()
            if current in visited:
                continue

            ordered_points.append(current)
            visited.add(current)

            if current == end:
                break

            # Explore neighbors
            neighbors = []
            for offset in neighbors_offsets:
                # Add offset to move from current to neighbor
                neighbor = tuple(np.array(current) + np.array(offset))
                # Check bounds
                if (0 <= neighbor[0] < self.skeleton.shape[0] and
                    0 <= neighbor[1] < self.skeleton.shape[1] and
                    0 <= neighbor[2] < self.skeleton.shape[2] and
                    self.skeleton[neighbor] == 1 and
                    neighbor != prev):
                    neighbors.append(neighbor)

            # After for loop, get a list of valid neighbors of current cell 
            # Sort neighbors by Euclidean distance

            neighbor_distances = [] # (key: distance from neightbor to current, val: neighbor)
            for neighbor in neighbors:
                distance = np.linalg.norm(np.array(neighbor) - np.array(current))
                neighbor_distances.append((distance, neighbor))
            
            # sort neighbor by distance, sorted in increasing order
            neighbor_distances.sort(key=lambda x: x[0])

            # get list of sorted neighbor from the sorted index
            sorted_neighbors = [neighbor for _, neighbor in neighbor_distances]

            for neighbor in sorted_neighbors:
                if neighbor not in visited:
                    stack.append((neighbor, current))

        return ordered_points
    
    def save_ordered_points(self, output_dir, label, base_filename):
        """
        Save the ordered points from the skeleton traversal as a NumPy file.

        Parameters
        ----------
        output_dir : str
            Directory to save the ordered points.
        label : int
            Label of the current skeleton being processed.
        base_filename : str
            Base filename for the saved NumPy file.

        Returns
        -------
        None
        """
        if self.ordered_points is None:
            logging.warning(f"No ordered points available for label {label}. Skipping save.")
            return
        # Create the filename for saving the points
        filename = f"{base_filename}_label_{label}_ordered_points.npy"
        output_path = os.path.join(output_dir, filename)

        # Save the ordered points as a numpy file
        np.save(output_path, np.array(self.ordered_points))
        logging.info(f"Ordered points saved to {output_path}")