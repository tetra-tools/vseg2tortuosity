# Vessel Graph Processing Pipeline

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This project processes segmented 3D vessel images by skeletonizing them, constructing a graph representation, calculating tortuosity metrics, visualizing the results, and exporting the findings in CSV format. The pipeline leverages Docker for consistent and reproducible environments, ensuring seamless execution across different systems.


## Publication

If you use this pipeline for your projects, please cite:

```
Pan Y., Kahru K., Barinas-Mitchell E., Ibrahim T., Andreescu C., Karim H. Measuring arterial tortuosity in the cerebrovascular system using Time-of-Flight MRI. medRxiv. 2024 Decemeber 26
```


---

## Table of Contents
- [Vessel Graph Processing Pipeline](#vessel-graph-processing-pipeline)
  - [Publication](#publication)
  - [Table of Contents](#table-of-contents)
  - [Download Example Testing Data](#download-example-testing-data)
  - [Input Image Requirement](#input-image-requirement)
  - [Installation](#installation)
    - [Option 1: Docker Image](#option-1-docker-image)
      - [Run Docker Image](#run-docker-image)
      - [Docker video demo](#docker-video-demo)
      - [Output of Docker](#output-of-docker)
    - [Option 2: Manual Setup](#option-2-manual-setup)
      - [Running the Repo](#running-the-repo)
      - [Manual Set up Video Demo](#manual-set-up-video-demo)
      - [Manual Example Output](#manual-example-output)
  - [Name and Label of Vessels](#name-and-label-of-vessels)
  - [Tortuosity Metrics](#tortuosity-metrics)
  - [Approximation Method](#approximation-method)
  - [Visualization Capabilities](#visualization-capabilities)
  - [References](#references)

---
## Download Example Testing Data

You could download the one MRA from [here](https://data.kitware.com/#collection/591086ee8d777f16d01e0724/folder/58a372e38d777f0721a64dc6) for testing the pipeline. This data is collected and made available by the CASILab at The University of North Carolina at Chapel Hill and were distributed by the MIDAS Data Server at Kitware, Inc. You can cite this paper if you also want to use this data [E. Bullitt, D. Zeng, G. Gerig, S. Aylward, S. Joshi, J. K. Smith, W. Lin, and M. G. Ewend, *"Vessel tortuosity and brain tumor malignancy: a blinded study,"* Academic Radiology, vol. 12, no. 10, pp. 1232–1240, 2005](https://pubmed.ncbi.nlm.nih.gov/16179200/)

---
## Input Image Requirement
The input TOF image needs to be in Nifti format specificially `.nii.gz`. We tested the TOF segmentation of the Circle of Willis with `0.625` mm isotropic resolution. If your Nifti segmented TOF cannot run with this pipeline, please submit a issue to this repo and we will work on accomodating your input type. 

The pipeline defaults to run on labels 1 and 2 and in our images, where labels 1 and 2 correspond to the left and right ICA, respectively. If these are not how your ICA is labeled, you need to use the manual set up version instead of the pre-built docker image and put in the label with respect to your ICA in argument like 
`python -m src.graph_processing.main -t /path/to/workdir -i segmentation.nii.gz --labels l1 l2` with `l1` and `l2` being your target vessel that you wish to get a tortuosity measurement. You can put more or less if you want as the number of labels is not limit to `2`. 

Alternatively, set the labels in the segmentation such that values 1 and 2 correspond to the appropriate structure. For manual set-up user, you can use the `-i` flag to provide your own input file name. 

---

## Installation 

### Option 1: Docker Image

To use the Vessel Graph Processing Pipeline, we **strongly recommend** installing Docker for a consistent and reproducible environment for AMD64 platform and ARM64 platform:

* [Download Docker](https://docs.docker.com/get-docker/)
* If you use the Docker image, you don't have to clone the project. You just have to install Docker on your computer and you are good to go.
* Tested with Docker Version `20.10.7`


#### Run Docker Image

docker will automatically pull images that you don't have locally from dockerhub
```bash
docker run --rm -v /path/to/work/dir/with/TOF/image:/app/workdir pyiyan/vseg2tortuosity:v1.0.0
```
#### Docker video demo

[![Watch the video](https://img.youtube.com/vi/XnBidVEEzYk/0.jpg)](https://www.youtube.com/watch?v=XnBidVEEzYk)

For docker user, the input name of each file need to be: `TOF_eICAB_CW.nii.gz`

#### Output of Docker
If you use the docker installation option, the `run_pipeline.sh` script in the docker will handle the clean up of files and you will have 
1. CSV File: skeleton_metrics.csv with tortuosity metrics for each label.
2. Plots Folder: folder that contains fitted spline and scatter of skeleton plots as well as residual plot
3. pipeline.log file
  
---

### Option 2: Manual Setup
If you prefer not to use Docker, you can set up the environment manually by cloning the repo first, create a enviroment for the project and install all dependency 
```
git clone https://github.com/tetra-tools/vseg2tortuosity.git
cd vseg2tortuosity
conda create --name vseg2tortuosity_env python=3.9
conda activate vseg2tortuosity_env
python -m pip install -r requirements.txt
python -m pip install .
```

#### Running the Repo
To test the pipeline on a single TOF input, you can first pull the repo following installation option 2, and then while you are in the repository vessel_skeleton_map, use command:
`python -m src.graph_processing.main -t /path/to/workdir -i segmentation_file_name.nii.gz`

`-t`: Working directory path (required)
`-i`: Input NIfTI file (default `TOF_eICAB_CW.nii.gz`)
`-l`: List of labels to process (default 1, 2)
`-v`: enable verbose mode to display all INFO from logger

if you use the segmentation result from  [eICAB](https://gitlab.com/FelixDumais/vessel_segmentation_snaillab), here is the table of correspondence between the label numbers and vessels:


If you want to apply the pipeline on the whole dataset, use the docker version of it and write shell script to process batch data. 

#### Manual Set up Video Demo

[![Watch the video](https://img.youtube.com/vi/rNT4W5paKeA/0.jpg)](https://www.youtube.com/watch?v=rNT4W5paKeA)

For docker user, the input name of each file need to be: `TOF_eICAB_CW.nii.gz`

#### Manual Example Output
If you use the manual installation option, the output will include:
1. Ordered Points File: .npy files containing ordered skeleton points.
2. CSV File: skeleton_metrics.csv with tortuosity metrics for each label.
3. Curve Plot: Visualizations of the parametric curve and residuals.


---

## Name and Label of Vessels 
| Arteries | Left | Right | 
|----------|------|:-----:|
| ICA      | 1    |   2   |
| ACA-A1   | 5    |   6   |
| MCA-M1   | 7    |   8   |
| PComm    | 9    |  10   |
| PCA-P1   | 11   |  12   |
| PCA-P2   | 13   |  14   |
| SCA      | 15   |  16   |
| AChA     | 17   |  18   |

**ICA**: Internal Carotid Arteries
**BA**: Basilar Artery
**AComm**: Anterior Communicating Artery
**ACA**: Anterior Cerebral Arteries
**MCA**: Middle Cerebral Arteries
**PComm**: Posterior Communicating Arteries
**PCA**: Posterior Cerebral Arteries
**SCA**: Superior Cerebellar Arteries
**AChA**: Anterior Choroidal Arteries


---

## Tortuosity Metrics  

| Metric                          | Symbol            | Formula                                      |
|---------------------------------|-------------------|---------------------------------------------|
| Mean Curvature                  |   $\kappa_{m}$    | $\int_a^b \kappa(t) \,dt$                   |
| Mean Square Curvature           |   $\kappa_{ms}$   | $\int_a^b \kappa(t)^2 \,dt$                 |
| Root Mean Square Curvature      |  $\kappa_{rms}$   | $\sqrt\frac{\int_a^b\kappa(t)^2 \, dt}{L}$  |
| Arc Over Chord                  |  $AOC$            | $\frac{L}{C}$                               |


---
## Approximation Method


The pipeline uses `cubic B-spline fitting` to approximate the parametric curve and iteratively finds the best smoothing factor that minimizes the curvature. This method, which involves iterative spline fitting and minimizing the RMS curvature, is described in [M.J. Johnson and G. Dougherty, Medical Engineering & Physics, 29 (2007), pp. 677–690](https://www.sciencedirect.com/science/article/pii/S1350453306001652).


---


## Visualization Capabilities
The Plotter class provides the following visualizations:

1. `3D Parametric Curve Plot`: Displays the original points and fitted spline.
2. `Curvature Plot`: Shows the estimated curvature as a function of arc length.
3. `Residual Plot`: Plots the residuals of the spline fit against arc length.
  
---


## References

1. M.J. Johnson, G. Dougherty, *"Iterative spline-fit minimization for RMS curvature,"* Medical Engineering & Physics, vol. 29, 2007, pp. 677–690.
2. Dumais, F., et al., *"eICAB: A novel deep learning pipeline for Circle of Willis multiclass segmentation and analysis,"* NeuroImage, vol. 260, 2022, p. 119425.
3. E. Bullitt, D. Zeng, G. Gerig, S. Aylward, S. Joshi, J. K. Smith, W. Lin, and M. G. Ewend, *"Vessel tortuosity and brain tumor malignancy: a blinded study,"* Academic Radiology, vol. 12, no. 10, pp. 1232–1240, 2005.
