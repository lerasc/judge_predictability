Code to reproduce the results from the paper **Early Career Citations Capture Judicial Idiosyncrasies and Predict Judgments**.

File overview and instructions: 

Set up the environment with the `enviornment.yml` and activate with `conda activate judge_prediction`.

Inside `routines.py` we store various routine functions. To reproduce our results, the file path inside the function 
`get_project_folder` needs to be adjusted to point to the folder in which this repository is stored. Download the 
`features.parquet` file (available upon request from slera@mit.edu) and place it inside the subfolder `./data/features/`.
Now all subsequent files can be executed to reproduce our results. 

Inside `case_assignment.py` we test whether or not cases are assigned to judges at random.

Inside `case_classifications.py` we predict the outcome of cases (whether plaintiff won or not) as a function of judge
characteristics and past citations. This represents the core our analysis.

Inside `data_quality_checks.py` we further test the sensitivity of our results with respect to data quality. 

Inside `classification_analysis.py` we run additional analyses to cross-check our results. 

Inside `visualizations.py` we aggregate several plotting routines to produce the paper figures.

Reach out to slera@mit.edu in case of issues.