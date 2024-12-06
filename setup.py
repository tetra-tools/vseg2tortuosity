# setup.py

from setuptools import setup, find_packages

setup(
    name='vessel_analysis',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.19.2',
        'pandas>=1.1.3',
        'matplotlib>=3.3.2',
        'seaborn>=0.11.0',
        'scipy>=1.5.2',
        'networkx>=2.5',
        'scikit-learn>=0.24.0',
        'trimesh>=3.9.0',
        'nibabel>=3.2.1',
    ],
    author='Yiyan Pan',
    author_email='yip33@pitt.edu',
    description='A project for analyzing blood vessel segmentations and calculating tortuosity metrics.',
    url='https://github.com/tetra-tools/vessel_skeleton_map',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    license='GPL-3.0',
    python_requires='>=3.8',
)