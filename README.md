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

## Notes
Currently, the pipeline only works with the following AMERIFLUX predictors: 
- TA 
- PA 
- P 
- WS 
- WD 
- USTAR 
- NETRAD 
- SW_IN, SW_IN_POT, SW_OUT 
- LW_IN, LW_OUT 
- LE 
- G 
- H 
- VPD 
- RH
- PPFD_IN, PPFD_OUT
- CO2
- SWC_1, SWC_2, SWC_3, SWC_4, SWC_5 (values may not be currently accurate)
- TS_1, TS_2, TS_3, TS_4, TS_5 (values may not be currently accurate)