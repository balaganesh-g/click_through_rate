click_through_rate_prediction
==============================

This is the POC project on complete machine learning life-cycle
1) <p><small>Project structure is based on <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
2) <p><small>To manage and version the data <a target="_blank" href="https://dvc.org/doc">Data Version Control(DVC) is used</a>.</small></p>
3) <p><small>To track and manage the model <a target="_blank" href="https://www.mlflow.org/docs/latest/tutorials-and-examples/tutorial.html">MLflow is used</a>.</small></p>
4) <p><small>To access the trained model as a service <a target="_blank" href="https://flask.palletsprojects.com/en/2.2.x/">flask is used</a>.</small></p> 
5) <p><small>To unittest the application <a target="_blank" href="https://docs.pytest.org/en/7.1.x/">Pytest is used</a>.</small></p> 
6) Github actions are used to automate the CI-CD
7) <p><small>To monitor the drift <a target="_blank" href="https://docs.evidentlyai.com/tutorial">Evidently AI is used</a>.</small></p>

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
