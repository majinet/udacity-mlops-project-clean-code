# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
to identify credit card customers that are most likely to churn.

## Files and data description
.
├── churn_library.py     # The churn_library.py is a library of functions to find customers who                            # are likely to churn.
├── churn_script_logging_and_tests.py # Contain unit tests for the churn_library.py functions and                                       # provide logging.
├── README.md            # Provides project overview, and instructions to use the code
├── data                 # Read this data
│   └── bank_data.csv
├── images               # Store EDA results 
│   ├── eda
│   └── results
├── logs                 # Store logs
└── models               # Store models

## Running Files
* ipython churn_library.py # Run main python file
* pytest churn_script_logging_and_tests.py # Run unit test for churn_library.py

## CI Integration
Setup a workflow to perform the following actions when push the change.
* run autopep8 and pylint
* run pytest
* run churn_library.py
