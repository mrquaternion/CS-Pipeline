# CS-Pipeline overview

`CS-Pipeline` is a command‑line workflow that enriches eddy-covariance (EC) station data with reanalysis variables from ERA5, and optionally gap‑fills AmeriFlux predictors. It also helps getting data to feed a neural network (previously used to analyze fires conditions across Canada). It operates in two main stages driven by YAML configuration files. It starts by querying data from the [Copernicus Data Store](https://cds.climate.copernicus.eu).



## Core workflow
**1. Prepare configuration**
- Create a YAML file (please use the same structure as in the repo, `download_config.yaml`) describing [the date range, target predictors, geographic footprint, aggregation level, and an optional field name to label features](#test).
- When gap-filling a specific station, supply a CSV file _only_ in the `process_config.yaml`.










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
- CO2 (may not work for point processing, i.e. `carbonpipeline point ...`)
- WTD (may not work for point processing, i.e. `carbonpipeline point ...`)
- SWC_1, SWC_2, SWC_3, SWC_4, SWC_5 (values may not be currently accurate)
- TS_1, TS_2, TS_3, TS_4, TS_5 (values may not be currently accurate)

## <a name="test"></a> Configuration options 