# Spark_Analysis

This repo is houses code for the detection and analysis of calcium sparks in cardiomyocytes. This code was utilized in the paper, "3D dSTORM Imaging Reveals Novel Detail of Ryanodine Receptor Localization in Rat Cardiac Myocytes" by Shen et al.

### Dependencies

The following packages are essential for the detection and analysis of sparks used within the publication.

- Python, version 3.6.5
- tifffile, version 0.15.1
	- "pip3 install tifffile==0.15.1"
- numpy, version 1.15.2
	- "pip3 install numpy==1.15.2"
- scipy, version 1.1.0
	- "pip3 install scipy==1.1.0"
- scikit-image, version 0.14.0
	- "pip3 install scikit-image==0.14.0"
- JupyterLab, version 0.33.1 or similar Jupyter Notebook/iPython Notebook interface.
	- "pip3 install jupyterlab==0.33.1"
- pandas, version 0.23.4
	- "pip3 install pandas==0.23.4"
	
#### Non-Essential Packages

The following packages can be commented out of code and are non-essential to running the analysis.

- matplotlib.pyplot, version 3.0.0
	- "pip3 install matplotlib==3.0.0"
- ipywidgets, version 7.4.1
	- "pip3 install ipywidgets==7.4.1"

### Directions

To replicate analysis performed on confocal images of cardiomyocytes found within the publication on personal confocal images, perform the following steps:

1. Pull the code from the bitbucket source and ensure all packages are correctly installed.
2. Ensure all confocal images are saved in a singular directory and saved as .tif files.
3. Open the notebook titled "detection_runner.ipynb"
4. Type the path to the directory in the "root" variable"
5. Run the notebook to completion to save data to .csv file

### Troubleshooting/Future Use

For any questions, comments, or concerns, please open a GitHub issue and I will address it as soon as possible.
If there is an error, please include the error message and all arguments used to run the command in the GitHub issue.
Open-source software and reproducibility is an important facet of scientific computing, so please acknowledge this work if it is used.

### Licensing

Please see LICENSE.txt for information concerning software licensure.
