import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev, interp1d, splrep
from scipy.stats import skew
import logging
import plotly.graph_objects as go

class ParametricCurve:
    """
    A class to represent a discrete parametric curve in 3D space.

    Attributes
    ----------
    x : np.ndarray
        Array of x-coordinates of the curve.
    y : np.ndarray
        Array of y-coordinates of the curve.
    z : np.ndarray
        Array of z-coordinates of the curve.
    num_points : int
        The number of points in the curve.
    s : np.ndarray
        Cumulative arc length of the curve.
    total_length : float
        Approximate total length of the curve.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        """
        Initialize the ParametricCurve with x, y, z coordinates.

        Parameters
        ----------
        x : np.ndarray
            Array of x-coordinates of the curve.
        y : np.ndarray
            Array of y-coordinates of the curve.
        z : np.ndarray
            Array of z-coordinates of the curve.

        Raises
        ------
        ValueError
            If the lengths of x, y, and z are not equal.
        """
        if not (len(x) == len(y) == len(z)):
            raise ValueError("x, y, z are not the same length")
        self.x = x
        self.y = y
        self.z = z
        self.num_points = len(self.x)
        self.s = self.compute_arc_length()
        self.total_length = self.approximate_length()

    def compute_arc_length(self) -> np.ndarray:
        """
        Compute the cumulative arc length of the curve.

        Returns
        -------
        np.ndarray
            Cumulative arc length array, where each element is the length
            from the start of the curve to the corresponding point.
        """
        dx = np.diff(self.x)
        dy = np.diff(self.y)
        dz = np.diff(self.z)

        ds = np.sqrt(dx**2 + dy**2 + dz**2)
        s = np.concatenate(([0], np.cumsum(ds)))
        return s

    def approximate_length(self) -> np.ndarray:
        """
        Approximate the total length of the curve.

        Returns
        -------
        float
            Total arc length of the curve.
        """
        return self.s[-1]


    def add_noise(self, snr_db: float, signal_power_metric: float):
        """
        Add Gaussian noise to the curve based on a specified SNR in dB.

        Parameters
        ----------
        snr_db : float
            Desired Signal-to-Noise Ratio in decibels.
        signal_power_metric : float
            Reference signal power metric (e.g., average radius) for calculating noise.

        Modifies
        --------
        self.x, self.y, self.z : np.ndarray
            The coordinates are modified in place with added Gaussian noise.

        Notes
        -----
        This method updates the cumulative arc length and total length after adding noise.
        """
        # calculate noise needed base on input snr_db
        noise_power = signal_power_metric ** 2 / (10 ** (snr_db / 10))
        sigma = np.sqrt(noise_power)

        # add noise in x, y, z
        self.x += np.random.normal(0, sigma, size=self.x.shape)
        self.y += np.random.normal(0, sigma, size=self.y.shape)
        self.z += np.random.normal(0, sigma, size=self.z.shape)

        # compute arc length after adding noise
        self.s = self.compute_arc_length()
        self.total_length = self.approximate_length()

    def plot_original(self, ax=None):
        """
        Plot the original discrete points of the curve in 3D space.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.Axes3DSubplot, optional
            A 3D matplotlib Axes object for plotting. If None, a new figure and axes are created.

        Returns
        -------
        None
        """
        from mpl_toolkits.mplot3d import Axes3D

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.x, self.y, self.z, color = 'red', label='Data Points')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Original Discrete Points of Space Curve ')
        plt.legend()
        plt.show()
        plt.close()

class SplineFitter:
    """
    A class for fitting a 3D parametric curve with splines and reparametrizing it by arc length.

    Attributes
    ----------
    curve : ParametricCurve
        The input parametric curve to fit.
    smoothing : float
        Smoothing factor for the spline. `s = 0` for interpolation; `s > 0` for approximation.
    k : int
        Degree of the spline. Default is `k = 3` for cubic splines.
    eval_points : int
        Number of points for evaluating the spline. If None, defaults to the number of points in the curve.
    tck : tuple
        Spline parameters (tck_x, tck_y, tck_z) for x, y, and z coordinates.
    total_length : float
        Total arc length of the fitted spline.
    s_uniform : np.ndarray
        Uniformly spaced arc length values for evaluation.
    x_spline, y_spline, z_spline : np.ndarray
        Evaluated x, y, and z coordinates of the spline.
    derivatives : dict
        Dictionary containing first and second derivatives of the spline with respect to arc length.
    """

    def __init__(self, curve: ParametricCurve, smoothing: float = 0.0, k: int = 3, eval_points: int = None):
        """
        Initialize the SplineFitter with a ParametricCurve.

        Parameters
        ----------
        curve : ParametricCurve
            The curve to fit using splines.
        smoothing : float, optional
            Smoothing factor for the spline fitting. Default is 0 (interpolation).
        k : int, optional
            Degree of the spline (1 to 5). Default is 3 (cubic spline).
        eval_points : int, optional
            Number of points for evaluating the spline. If None, uses the number of points in the input curve.
        """
        self.curve = curve
        self.smoothing = smoothing
        self.k = k
        self.eval_points = eval_points 

        self.tck = self.fit_spline()
        self.compute_spline_arc_length()
        self.s_uniform, self.x_spline, self.y_spline, self.z_spline = self.evaluate_spline()
        self.derivatives = self.compute_derivatives()


    def fit_spline(self):
        """
        Fit splines to x, y, and z coordinates of the curve.

        Returns
        -------
        tuple
            Spline parameters (tck_x, tck_y, tck_z) for x, y, and z coordinates.
        """
        num_points = self.curve.num_points
        u = np.linspace(0, 1, num_points)
        tck_x = splrep(u, self.curve.x, s=self.smoothing, k=self.k)
        tck_y = splrep(u, self.curve.y, s=self.smoothing, k=self.k)
        tck_z = splrep(u, self.curve.z, s=self.smoothing, k=self.k)
        return (tck_x, tck_y, tck_z)
    
    def compute_spline_arc_length(self):
        """
        Compute the cumulative arc length along the spline and create mappings from parameter (u) to arc length (s).
        """
        # Use more points for better accuracy
        num_points = 10 * self.curve.num_points
        u_values = np.linspace(0, 1, num_points)

        dx_du = splev(u_values, self.tck[0], der=1)
        dy_du = splev(u_values, self.tck[1], der=1)
        dz_du = splev(u_values, self.tck[2], der=1)

        du = np.diff(u_values)
        ds = np.sqrt(dx_du**2 + dy_du**2 + dz_du**2)[:-1] * du
        s = np.concatenate(([0], np.cumsum(ds)))

        self.u_to_s = interp1d(u_values, s, kind='linear')
        self.s_to_u = interp1d(s, u_values, kind='linear')
        self.total_length = s[-1]

    def evaluate_spline(self):
        """
        Evaluate the fitted spline at uniformly spaced points re-parametrized by arc length.

        Returns
        -------
        tuple
            Uniform arc length values (s_uniform), and x, y, z coordinates of the evaluated spline.
        """
        if self.eval_points is None:
            self.eval_points = self.curve.num_points
        
        s_uniform = np.linspace(0, self.total_length, self.eval_points)
        u_uniform = self.s_to_u(s_uniform)
        
        x_spline = splev(u_uniform, self.tck[0], der=0)
        y_spline = splev(u_uniform, self.tck[1], der=0)
        z_spline = splev(u_uniform, self.tck[2], der=0)
        
        self.s_uniform = s_uniform
        return (s_uniform, x_spline, y_spline, z_spline)

    def compute_rmse(self):
        """
        Compute the Root Mean Square Error (RMSE) between the original data points and the spline fit.

        Returns
        -------
        float
            The RMSE value.
        """

        # Parameter values for the original points
        t_values = np.linspace(0, 1, self.curve.num_points)
        
        # Spline evaluated points
        x_spline = splev(t_values, self.tck[0])
        y_spline = splev(t_values, self.tck[1])
        z_spline = splev(t_values, self.tck[2])
        
        # Calculate residuals for each dimension
        residuals_x = self.curve.x - x_spline
        residuals_y = self.curve.y - y_spline
        residuals_z = self.curve.z - z_spline
        
        # Calculate total residuals (distance between original points and fitted spline points)
        self.residuals = np.sqrt(residuals_x**2 + residuals_y**2 + residuals_z**2)
        
        # Compute RMSE
        self.rmse = np.sqrt(np.mean(self.residuals**2))

        # Skewness
        self.residual_skewness = skew(self.residuals)

        # Max residual
        self.max_residual = np.max(self.residuals)
        
        return self.rmse


    def compute_derivatives(self):
        """
        Compute first and second derivatives of the spline with respect to arc length.

        Returns
        -------
        dict
            Dictionary containing first and second derivatives.
        """
        u_uniform = self.s_to_u(self.s_uniform)

        du_ds = np.gradient(u_uniform, self.s_uniform)
        dx_du = splev(u_uniform, self.tck[0], der=1)
        dy_du = splev(u_uniform, self.tck[1], der=1)
        dz_du = splev(u_uniform, self.tck[2], der=1)

        ddx_du2 = splev(u_uniform, self.tck[0], der=2)
        ddy_du2 = splev(u_uniform, self.tck[1], der=2)
        ddz_du2 = splev(u_uniform, self.tck[2], der=2)

        dx_ds = dx_du * du_ds
        dy_ds = dy_du * du_ds
        dz_ds = dz_du * du_ds

        ddu_ds = np.gradient(du_ds, self.s_uniform)
        ddx_ds2 = ddx_du2 * (du_ds ** 2) + dx_du * ddu_ds
        ddy_ds2 = ddy_du2 * (du_ds ** 2) + dy_du * ddu_ds
        ddz_ds2 = ddz_du2 * (du_ds ** 2) + dz_du * ddu_ds

        return {
            'first_derivative': (dx_ds, dy_ds, dz_ds),
            'second_derivative': (ddx_ds2, ddy_ds2, ddz_ds2),
        }
    
    def plot_residuals_vs_arc_length(self, path = None):
        """
        Plot the residuals against the arc length of the curve.

        Returns
        -------
        None
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.curve.s, self.residuals, label='Residuals', color='purple')
        plt.xlabel('Arc Length')
        plt.ylabel('Residual')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        plt.legend()
        plt.grid()
        if(path != None):
            plt.savefig(path, dpi=300)
            logging.info(f"Residual plot saved at: {path}")
        #plt.show()
        plt.close()

    def plot_spline(self, ax=None):
        """
        Plot the fitted spline alongside the original data points.
        """
        from mpl_toolkits.mplot3d import Axes3D

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        ax.plot(self.x_spline, self.y_spline, self.z_spline, color='blue', label='Fitted Spline')
        ax.scatter(self.curve.x, self.curve.y, self.curve.z, color='red', label='Data Points')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

class CurvatureCalculator:
    """
    A class for calculating curvature-based metric of a 3D parametric curve from spline fitting.

    Attributes
    ----------
    curve : ParametricCurve
        The parametric curve to analyze.
    spline_fitter : SplineFitte
        An instance of SplineFitter used for spline-based curvature computation.
    curvature_spline : np.ndarray or None
        Curvature computed using spline-based approximation.
    """
    def __init__(self, curve: ParametricCurve, spline_fitter: SplineFitter = None):
        """
        Initialize the CurvatureCalculator.

        Parameters
        ----------
        curve : ParametricCurve
            The parametric curve to analyze.
        spline_fitter : SplineFitter
            Fitted spline used for curvature computation.

        Notes
        -----
        If no `SplineFitter` instance is provided, spline-based curvature computation will not be available.
        """
        self.curve = curve
        self.spline_fitter = spline_fitter
        self.curvature_spline = None 


    def compute_spline_curvature(self) -> np.ndarray:
        """
        Compute the curvature using the spline-based method.

        Returns
        -------
        np.ndarray
            The curvature array computed from the second derivatives of the spline.

        Raises
        ------
        ValueError
            If no `SplineFitter` instance is provided.
        """
        if self.spline_fitter is None:
            raise ValueError("SplineFitter instance is required for spline-based curvature computation.")

        ddx_ds2, ddy_ds2, ddz_ds2 = self.spline_fitter.derivatives['second_derivative']
        curvature = np.sqrt(ddx_ds2**2 + ddy_ds2**2 + ddz_ds2**2)
        self.curvature_spline = curvature
        return curvature
    

    def total_curvature(self) -> float:
        """
        Compute the total curvature of the curve using the specified method.

        Returns
        -------
        float
            The total curvature of the curve.

        Raises
        ------
        ValueError
            If an invalid method is specified.
        """
        if self.curvature_spline is None:
            self.compute_spline_curvature()
        
        curvature = self.curvature_spline
        ds = np.gradient(self.spline_fitter.s_uniform)
        total_kappa = np.sum(curvature * ds)
        
        return total_kappa

    def mean_squared_curvature(self) -> float:
        """
        Compute the mean squared curvature of the curve.

        Returns
        -------
        float
            The mean squared curvature of the curve.

        Notes
        -----
        This method uses the spline-based curvature for computation.
        """
        if self.curvature_spline is None:
            self.compute_spline_curvature()
        
        curvature = self.curvature_spline
        ds = np.gradient(self.spline_fitter.s_uniform)
        mean_sq_kappa = np.sum((curvature ** 2) * ds)
        
        return mean_sq_kappa
 
    def rms_curvature(self) -> float:
        """
        Compute the Root Mean Square (RMS) curvature of the curve.

        Returns
        -------
        float
            The RMS curvature of the curve.

        Notes
        -----
        This method uses the mean squared curvature and the total arc length of the spline.
        """
        mean_sq_kappa = self.mean_squared_curvature()
        total_length = self.spline_fitter.total_length
        rms_kappa = np.sqrt(mean_sq_kappa * total_length)
        
        return rms_kappa


class Plotter:
    """
    A utility class for plotting 3D parametric curves and their related data.

    Methods
    -------
    plot_curve(curve, spline_fitter=None, save_path=None)
        Plot the original curve and the fitted spline, if provided.

    plot_curvature(s, curvature, method="Spline", save_path=None)
        Plot the curvature of the curve against arc length.

    plot_curve_by_curvature(spline_fitter, curvature, save_path=None)
        Plot the fit spline with segments colored according to curvature values.
    """
    @staticmethod
    def plot_curve(curve: ParametricCurve, spline_fitter: SplineFitter = None, save_path=None):
        """
        Plot the original parametric curve and the fitted spline, if provided.

        Parameters
        ----------
        curve : ParametricCurve
            The parametric curve to be plotted.
        spline_fitter : SplineFitter, optional
            The fitted spline to overlay on the original curve. Default is None.
        save_path : str, optional
            Path to save the plot as an image file. If None, the plot is not saved.

        Returns
        -------
        None
        """
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # original data points
        ax.scatter(curve.x, curve.y, curve.z, color='red', s=10, label='Data Points')

        # spline if available
        if spline_fitter is not None:
            ax.plot(spline_fitter.x_spline, spline_fitter.y_spline,
                    spline_fitter.z_spline, color='blue', label='Fitted Spline',
            )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=300)
            logging.info(f"Curvature plot saved at: {save_path}")

        #plt.show()
        plt.close()

    @staticmethod
    def plot_curvature(s: np.ndarray, curvature: np.ndarray, method: str = "Spline", save_path=None):
        """
        Plot curvature as a function of arc length.

        Parameters
        ----------
        s : np.ndarray
            Array of arc length values.
        curvature : np.ndarray
            Array of curvature values corresponding to the arc length.
        method : str, optional
            Label for the method used to compute curvature (default is "Spline").
        save_path : str, optional
            Path to save the plot as an image file. If None, the plot is not saved.

        Returns
        -------
        None
        """

        plt.figure(figsize=(10, 5))
        plt.plot(s, curvature, label=f'Curvature ({method})', color='green')
        plt.xlabel('Arc Length')
        plt.ylabel('Curvature')
        plt.title(f'Curvature vs Arc Length ({method})')
        plt.legend()
   
        if save_path:
            plt.savefig(save_path, dpi=300)
            logging.info(f"Curvature plot saved at: {save_path}")

        #plt.show()
        plt.close()

    @staticmethod
    def plot_curve_by_curvature(spline_fitter, curvature, save_path=None):
        """
        Plot the fitted spline with segments colored by curvature values.

        Parameters
        ----------
        spline_fitter : SplineFitter
            The fitted spline object containing the evaluated spline points.
        curvature : np.ndarray
            Array of curvature values for color mapping.
        save_path : str, optional
            Path to save the plot as an image file. If None, the plot is not saved.

        Returns
        -------
        None

        Notes
        -----
        The color of each segment of the spline is determined by the curvature values using a colormap.
        """
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Normalize the curvature values for color mapping
        norm = mcolors.Normalize(vmin=np.min(curvature), vmax=np.max(curvature))
        cmap = cm.get_cmap('plasma') 
        
        # Iterate over the points to draw line segments colored by curvature
        for i in range(len(spline_fitter.x_spline) - 1):
            color = cmap(norm(curvature[i]))
            ax.plot(
                spline_fitter.x_spline[i:i+2], 
                spline_fitter.y_spline[i:i+2], 
                spline_fitter.z_spline[i:i+2], 
                color=color, linewidth=2
            )

        # Add a colorbar to show the mapping from curvature to color
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(curvature)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Curvature')

        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('Spline Colored by Curvature')

        if save_path:
            plt.savefig(save_path, dpi=300)
            logging.info(f"Curve colored by curvature saved at: {save_path}")

        plt.show()
        plt.close()
    @staticmethod
    def plot_interactive_curve(curve, spline_fitter=None, save_path=None, title="3D Vessel Curve"):

        
    
        # Original points
        x_orig = curve.x
        y_orig = curve.y
        z_orig = curve.z
    
        # Create the figure
        fig = go.Figure()
    
        # Add original points
        fig.add_trace(go.Scatter3d(
            x=x_orig, y=y_orig, z=z_orig,
            mode='markers',
            name='Skeleton points',
            marker=dict(
                size=4,
                color='blue',
                opacity=0.8
            )
        ))
    
    # Add spline curve if available
        if spline_fitter is not None:
            fig.add_trace(go.Scatter3d(
                x=spline_fitter.x_spline, y=spline_fitter.y_spline, z=spline_fitter.z_spline,
                mode='lines',
                name='Fitted spline',
                line=dict(
                    color='red',
                    width=5
                )
            ))
    
    # Update layout with nice defaults
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            width=900,
            height=700,
            margin=dict(l=65, r=50, b=65, t=90)
        )
    
        # Save as HTML
        if save_path:
            fig.write_html(save_path)
            logging.info(f"Interactive 3D plot saved to {save_path}")


class CurveAnalyzer:
    """
    A high-level interface for analyzing 3D parametric curves, including spline fitting,
    curvature computation, and various evaluation metrics.

    Attributes
    ----------
    curve : ParametricCurve
        The parametric curve to analyze.
    spline_fitter : SplineFitter or None
        Fitted spline object for the curve.
    curvature_calculator : CurvatureCalculator or None
        Calculator for computing curvature metrics.
    lambda_weight : float
        Weighting factor for the final loss computation.
    optimizer : SplineOptimizer
        An instance of SplineOptimizer for iterative fitting.
    """
    def __init__(self, curve: ParametricCurve):
        """
        Initialize the CurveAnalyzer with a ParametricCurve.

        Parameters
        ----------
        curve : ParametricCurve
            The parametric curve to analyze.
        """
        self.curve = curve
        self.spline_fitter = None
        self.curvature_calculator = None
        self.lambda_weight = 0.5
        self.optimizer = SplineOptimizer(self)

    def fit_spline(self, smoothing: float = 10.0, k: int = 3, eval_points: int = None):
        """
        Fit a spline to the curve using specified parameters.

        Parameters
        ----------
        smoothing : float, optional
            Smoothing factor for the spline fitting (default is 10.0).
        k : int, optional
            Degree of the spline (default is 3).
        eval_points : int, optional
            Number of evaluation points for the spline (default is None).
        """
        self.spline_fitter = SplineFitter(self.curve, smoothing=smoothing, k=k, eval_points=eval_points)

    def compute_curvatures(self):
        """
        Compute the curvature of the curve using the fitted spline.
        """
        self.curvature_calculator = CurvatureCalculator(self.curve, spline_fitter=self.spline_fitter)
        if self.spline_fitter is not None:
            self.curvature_calculator.compute_spline_curvature()

    def compute_final_loss(self) -> float:
        """
        Compute the final objective loss as MSE + lambda * RMS curvature.

        Returns
        -------
        dict
            A dictionary with RMSE and final loss value.
        """

        rmse = self.spline_fitter.compute_rmse()
        rms_curvature = self.curvature_calculator.rms_curvature()
        # Final loss as addition of rmse and lambda_weight * rms_curvature 
        final_loss = rmse + self.lambda_weight * rms_curvature
        return {"rmse": rmse, "final_loss": final_loss}
    
    def calculate_aoc(self) -> float:
        """
        Calculate the Arc Over Chord (AOC) ratio.
        
        This metric is a measure of tortuosity based on the ratio of the arc length
        (measured along the spline) to the straight-line chord length between the first
        and last points of the curve.

        Returns
        -------
        float
            The Arc Over Chord (AOC) ratio.

        Raises
        ------
        ValueError
            If spline fitting has not been performed.
        """
        if self.spline_fitter is None:
            raise ValueError("Spline fitting must be performed before calculating AOC.")

        # Calculate the Euclidean distance (chord) between the first and last point
        start_point = np.array([self.curve.x[0], self.curve.y[0], self.curve.z[0]])
        end_point = np.array([self.curve.x[-1], self.curve.y[-1], self.curve.z[-1]])
        chord_length = np.linalg.norm(start_point - end_point)

        # Use the total spline length as the arc length
        arc_length = self.spline_fitter.total_length

        # Calculate the AOC ratio (arc length over chord length)
        aoc = arc_length / chord_length
        logging.info(f"AOC (Arc Over Chord): {aoc:.6f}")
        return aoc
    
    def plot_curve(self, save_path=None):
        """
        Plot the curve and the fitted spline.
        """
        Plotter.plot_curve(self.curve, self.spline_fitter, save_path=save_path)

    def plot_curve_by_curvature(self, save_path=None):
        curvature = self.curvature_calculator.curvature_spline
        Plotter.plot_curve_by_curvature(self.spline_fitter, curvature, save_path=save_path)

    def plot_residuals_vs_arc_length(self, save_path=None):
        self.spline_fitter.plot_residuals_vs_arc_length(
            path=save_path
        )
    def plot_interactive(self, save_path=None, title=None):

        if title is None:
            title = "3D Vessel Curve"
        Plotter.plot_interactive_curve(self.curve, self.spline_fitter, save_path=save_path, title=title)


class SplineOptimizer:
    """
    A class for optimizing the spline fitting process by adjusting the smoothing factor.

    Attributes
    ----------
    analyzer : CurveAnalyzer
        An instance of CurveAnalyzer used for spline fitting and curvature computation.
    """
    def __init__(self, analyzer: CurveAnalyzer):
        """
        Initialize the SplineOptimizer with a CurveAnalyzer.

        Parameters
        ----------
        analyzer : CurveAnalyzer
            An instance of CurveAnalyzer to perform fitting and analysis.
        """
        self.analyzer = analyzer
    def estimate_initial_smoothing(self) -> float:
        """
        Estimate an initial smoothing factor based on average point spacing.
        """
        curve_points = np.column_stack((self.analyzer.curve.x, self.analyzer.curve.y, self.analyzer.curve.z))
        distances = np.sqrt(np.sum(np.diff(curve_points, axis=0)**2, axis=1))
        avg_spacing = np.mean(distances)
        smoothing_factor = avg_spacing * len(curve_points) * 0.1  # Adjust 0.1 as needed
        return smoothing_factor
    
    def iterative_spline_fit(self, initial_smoothing: float, k: int = 3,
                            max_iterations: int = 100, tolerance: float = 1e-6,
                            eval_points: int = None)-> float:
        """
        Iteratively adjust the smoothing factor to minimize RMS curvature.

        Parameters
        ----------
        initial_smoothing : float
            Initial smoothing factor.
        k : int, optional
            Degree of the spline (default is 3).
        max_iterations : int, optional
            Maximum number of iterations (default is 100).
        tolerance : float, optional
            Convergence tolerance (default is 1e-6).
        eval_points : int, optional
            Number of evaluation points (default is None).

        Returns
        -------
        float
            Final RMS curvature after fitting.
        """
        smoothing = initial_smoothing
        previous_rms = np.inf
        smoothing_increment = 1.0

        for iteration in range(1, max_iterations + 1):
            self.analyzer.fit_spline(smoothing=smoothing, k=k, eval_points=eval_points)
            self.analyzer.compute_curvatures()
            rms_curvature = self.analyzer.curvature_calculator.rms_curvature()
            logging.info(f"Iteration {iteration}: Smoothing = {smoothing:.6f}, RMS Curvature = {rms_curvature:.6f}")

            if abs(previous_rms - rms_curvature) < tolerance:
                logging.info("Convergence achieved.")
                break

            if rms_curvature < previous_rms:
                smoothing += smoothing_increment
            else:
                smoothing -= smoothing_increment
                smoothing_increment /= 2.0

            previous_rms = rms_curvature

        return rms_curvature
    
    def fit_with_multiple_initial_smoothings(self, k: int = 3, max_iterations: int = 100,
                                            tolerance: float = 1e-6, eval_points: int = None):
        """
        Test multiple initial smoothing factors and choose the one with the best (lowest) RMS curvature.

        Parameters
        ----------
        k : int, optional
            Degree of the spline (default is 3).
        max_iterations : int, optional
            Maximum number of iterations (default is 100).
        tolerance : float, optional
            Convergence tolerance (default is 1e-6).
        eval_points : int, optional
            Number of evaluation points (default is None).

        Returns
        -------
        float
            The final loss after fitting with the best smoothing factor.
        """

        initial_smoothing = self.estimate_initial_smoothing()
        candidate_smoothings = [initial_smoothing * factor for factor in [0.5, 1.0, 1.5, 2.0]]
        
        best_smoothing = None
        best_rms_curvature = np.inf

        # Try all 4 possible initial_smoothing factor and keep the best one
        for initial_smoothing in candidate_smoothings:
            logging.info(f"\nTesting initial smoothing factor: {initial_smoothing:.6f}")
            rms_curvature = self.iterative_spline_fit(
                initial_smoothing=initial_smoothing, k=k, max_iterations=max_iterations,
                tolerance=tolerance, eval_points=eval_points
            )
            logging.info(f"Finished with smoothing: {initial_smoothing}, RMS Curvature: {rms_curvature}")

            if rms_curvature < best_rms_curvature:
                best_rms_curvature = rms_curvature
                best_smoothing = initial_smoothing

        logging.info(f"\nBest initial smoothing factor: {best_smoothing}, with RMS Curvature: {best_rms_curvature}")
        # Perform final fit with the best smoothing factor
        self.iterative_spline_fit(initial_smoothing=best_smoothing, k=k,
                                  max_iterations=max_iterations, tolerance=tolerance,
                                  eval_points=eval_points)

        # Calculate and output the final objective function value as a reference for fit quality
        final_loss = self.analyzer.compute_final_loss()
        return final_loss
    

class Shape:
    """
    Abstract base class for 3D parametric shapes.

    Methods
    -------
    generate() -> ParametricCurve
        Generate the ParametricCurve for the shape. Must be implemented by subclasses.
    """
    def generate(self) -> ParametricCurve:
        """
        Generate the ParametricCurve for the shape.

        Returns
        -------
        ParametricCurve
            The generated parametric curve.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
class ElipticHelixShape(Shape):
    """
    Represents an elliptic helix shape.

    Attributes
    ----------
    a : int
        Amplitude of the x-coordinate.
    num_turns : int
        Number of turns of the helix.
    num_points : int
        Number of points to generate along the curve.

    Methods
    -------
    generate() -> ParametricCurve
        Generate the parametric curve for the elliptic helix.
    theoretical_length() -> float
        Compute the theoretical arc length of the elliptic helix.
    theoretical_curvature() -> np.ndarray
        Compute the theoretical curvature of the elliptic helix.
    """
    def __init__(self, a: int, num_turns: int,  num_points: int = 1000):
        self.a = a
        self.num_turns = num_turns
        self.num_points = num_points
    def generate(self) -> ParametricCurve:
        """
        Generate the parametric curve for the elliptic helix.

        Returns
        -------
        ParametricCurve
            The generated parametric curve.
        """
        t = np.linspace(0, 2 * np.pi * self.num_turns, self.num_points)
        x = self.a * np.cos(t)  
        y = np.sin(t)           
        z = t                  
        return ParametricCurve(x, y, z)
    
    def theoretical_length(self) -> float:
        """
        Compute the theoretical arc length of the elliptic helix.

        Returns
        -------
        float
            Theoretical arc length.
        """
        return np.sqrt(1 + self.a**2) * 2 * np.pi * self.num_turns
    
    def theoretical_curvature(self) -> np.ndarray:
        """
        Compute the theoretical curvature of the elliptic helix.

        Returns
        -------
        np.ndarray
            Curvature values (constant for elliptic helix).
        """
        curvature = self.a / np.sqrt(self.a**2 + 1)
        return np.full(self.num_points, curvature)
    

class HelixShape(Shape):
    """
    Represents a standard helical shape.

    Attributes
    ----------
    radius : float
        Radius of the helix.
    pitch : float
        Pitch (vertical step) of the helix.
    num_points : int
        Number of points to generate along the curve.

    Methods
    -------
    generate() -> ParametricCurve
        Generate the parametric curve for the helix.
    theoretical_length() -> float
        Compute the theoretical arc length of the helix.
    theoretical_curvature() -> np.ndarray
        Compute the theoretical curvature of the helix.
    """
    def __init__(self, radius: float, pitch: float, num_points: int = 1000):
        self.radius = radius
        self.pitch = pitch
        self.num_points = num_points

    def generate(self) -> ParametricCurve:
        """
        Generate the parametric curve for the helix.

        Returns
        -------
        ParametricCurve
            The generated parametric curve.
        """
        t = np.linspace(0, 2 * np.pi, self.num_points)
        x = self.radius * np.cos(t)
        y = self.radius * np.sin(t)
        z = self.pitch * t
        return ParametricCurve(x, y, z)

    def theoratical_length(self) -> float:
        """
        Compute the theoretical arc length of the helix.

        Returns
        -------
        float
            Theoretical arc length.
        """
        return (2 * np.pi) * np.sqrt(self.radius**2 + self.pitch**2) 

    def theoratical_curvature(self) -> np.ndarray:
        """
        Compute the theoretical curvature of the helix.

        Returns
        -------
        np.ndarray
            Curvature values (constant for helix).
        """
        curvature = self.radius / (self.radius**2 + self.pitch**2)
        return np.full(self.num_points, curvature)


class VivianiShape(Shape):
    """
    Represents a Viviani curve, a special type of space curve.

    Attributes
    ----------
    a : float
        Amplitude of the curve.
    num_turns : int
        Number of turns of the Viviani curve.
    num_points : int
        Number of points to generate along the curve.

    Methods
    -------
    generate() -> ParametricCurve
        Generate the parametric curve for the Viviani curve.
    theoretical_length() -> float
        Compute the theoretical arc length of the Viviani curve.
    theoretical_curvature() -> np.ndarray
        Compute the theoretical curvature of the Viviani curve.
    """
    def __init__(self, a: float, num_turns: int = 2, num_points: int = 1000):
        self.a = a
        self.num_turns = num_turns
        self.num_points = num_points

    def generate(self) -> ParametricCurve:
        """
        Generate the parametric curve for the Viviani curve.

        Returns
        -------
        ParametricCurve
            The generated parametric curve.
        """
        t = np.linspace(0, 2 * np.pi * self.num_turns, self.num_points)
        x = self.a * (1 + np.cos(t))
        y = self.a * np.sin(t)
        z = 2 * self.a * np.sin(t / 2)
        return ParametricCurve(x, y, z)

    def theoratical_length(self) -> float:
        """
        Compute the theoretical arc length of the Viviani curve.

        Returns
        -------
        float
            Theoretical arc length.
        """
        from scipy.special import ellipeinc
        m = 0.5  
        arc_length_per_turn = 2 * self.a * np.sqrt(2) * ellipeinc(np.pi, m)
        total_arc_length = arc_length_per_turn * self.num_turns
        return total_arc_length

    def theoratical_curvature(self) -> np.ndarray:
        """
        Compute the theoretical curvature of the Viviani curve.

        Returns
        -------
        np.ndarray
            Curvature values for the Viviani curve.
        """
        curvature = np.sqrt(13 + 3 * np.cos(self.t)) / (self.a * (3 + np.cos(self.t))**(3/2))
        return curvature
    
def main():
    helix = HelixShape(radius=10, pitch=10, num_points=40)
    helix_curv = helix.generate()
    np.random.seed(42)

    snr_db = 30
    signal_power_metric = helix.radius
    helix_curv.add_noise(snr_db=snr_db, signal_power_metric=signal_power_metric)
    logging.info(f"Add Gaussian noise with SNR = {snr_db} dB.")

    analyzer = CurveAnalyzer(helix_curv)
    points_per_unit_length = 10 

    total_length = helix_curv.total_length
    eval_points = int(total_length * points_per_unit_length)
    logging.info(f"length is: {total_length:.2f}")
    logging.info(f"eval points: {eval_points}")
    
    optimizer = SplineOptimizer(analyzer)
    final_loss = optimizer.fit_with_multiple_initial_smoothings(
        k=3, max_iterations=200, tolerance=1e-6, eval_points=eval_points
    )

    analyzer.plot_curve()
    analyzer.plot_curve_by_curvature()
    analyzer.plot_residuals_vs_arc_length()
    analyzer.spline_fitter.plot_residuals_vs_arc_length()

    logging.info(f"spline_len: {analyzer.spline_fitter.total_length}")
    total_curv_spline = analyzer.curvature_calculator.total_curvature()

    logging.info(f"Total Curvature (Spline-Based): {total_curv_spline:.6f}")
    ms_curv_spline = analyzer.curvature_calculator.mean_squared_curvature()

    logging.info(f"ms Curvature (Spline-Based): {ms_curv_spline:.6f}")
    rms_curv_spline = analyzer.curvature_calculator.rms_curvature()
    
    logging.info(f"rms Curvature (Spline-Based): {rms_curv_spline:.6f}")
    logging.info(f"Final Objective Loss (MSE + lambda * RMS Curvature): {final_loss['final_loss']:.6f}")
    logging.info(f"Final MSE : {final_loss['rmse']:.6f}")

if __name__ == "__main__":
    main()