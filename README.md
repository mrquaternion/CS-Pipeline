# CS-Pipeline

## About
This package is designed to fill missing values in environmental predictor datasets from EC stations. It works by providing a CSV file and the coordinates of the site. New columns will be created with values coming from ERA5 dataset.

## Install
The `carbonpipeline` package can be installed in the following way:
1. Clone the repository on your computer
2. Navigate to the root of the project
3. Create a virtual environment and activate it
4. Make sure your CDS API credentials are set up in `~/.cdsapirc`, otherwise follow this instruction https://arc.net/l/quote/ilmawrkf
3. Finally, enter the command `pip install -e .` in your CLI
4. Use `carbonpipeline --help` for more info!