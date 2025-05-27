# What is ERA5?

It is an *atmospheric reanalysis* project. What is an atmospheric reanalysis? It's a meteorological method used to reanalyse objectively global surface data and altitude taken on a long period of time for the assimilation of data in numerical weather prediction (NWP) models.

## What is it made of?

It is made out of climate conditions data over the past decades. Its content range from temperature to soil moisture. Basically, it contains reanalysis of atmospheric, sea surface and land surface conditions for any hour since 1 January 1940 and until only five days behind present (current date).

## What is the principle of this reanalysis

It is to provide "maps without gaps". In other words, to gather as much information as possible about variables of the Earth system. It is also combined with data that was generated from model simulations in order to "fill the gaps" (that's what we are trying to do) in the observations.

## Its competitors 

- E-OBS : doesn't use a model to simulate data (no 3D-var or 4D-var assimilation)
- MERRA-2 (3D-var assimilation)
- JRA-55 (4D-var assimilation)

# Current environment

At the path `.../Developper/Stage_25/era5_workbench/cdsenv`, I did the following command to install extra package with `conda install -c conda-forge [package name]` once the env is activated:

- `cdsapi`
- `xarray`
- `metview`
- `eccodes`

To see all the packages install manually (all excepts the one by default by creation), do the command `conda env export --from-history`.

The `conda` environment can be activated by doing `. ./activate_cdsenv`.

# API request

The code from one's dataset will pack the data in a `.zip` file (preferable if the data is huge). Here's useful commands :

- `du -h name_of_file.zip` : show `.zip` file size (before unpacking)
- `unzip name_of_file.zip -d destination_folder/ && rm name_of_file.zip` : unzip the file to a specific folder and delete the zip file

We will be working with GRIB files since it's the praised way even though most people work with the netCDF format.

# Understand GRIB files

There's 2 options :

- `ecCodes` (CLI) : https://confluence.ecmwf.int/display/ECC/ecCodes+Home
- `Metview` (GUI) : https://metview.readthedocs.io/en/latest/

More can be read here https://confluence.ecmwf.int/display/CKB/What+are+GRIB+files+and+how+can+I+read+them.

In CLI, grib files content can be displayed as such: `grib_ls your_file.grib`

# Predictors in CarbonSense

| Carbonsense PREDICTORS | full name                                         | Units                                                        |
| ---------------------- | ------------------------------------------------- | ------------------------------------------------------------ |
| TA                     | Aire temperature                                  | $\degree \text{C}$                                           |
| P                      | Precipitation                                     | $\text{mm}$                                                  |
| RH                     | Relative humidity (0-100)                         | $\%$                                                         |
| VPD                    | Vapor pressure deficit                            | $\text{hPa}$                                                 |
| PA                     | Atmospheric pressure                              | $\text{kPa}$                                                 |
| CO2                    | Carbon dioxide mole fraction in wet air           | $\frac{\mu \text{mol}_{\text{CO}_2}}{\text{mol}}$            |
| SW_IN                  | Shortwave radiation (incoming)                    | $\frac{\text{W}}{\text{m}^2}$                                |
| SW_IN_POT              | Potential shortwave radiation (incoming)          | $\frac{\text{W}}{\text{m}^2}$                                |
| SW_OUT                 | Shortwave radiation (outgoing)                    | $\frac{\text{W}}{\text{m}^2}$                                |
| LW_IN                  | Longwave radiation (incoming)                     | $\frac{\text{W}}{\text{m}^2}$                                |
| LW_OUT                 | Longwave radiation (outgoing)                     | $\frac{\text{W}}{\text{m}^2}$                                |
| NETRAD                 | Net radiation                                     | $\frac{\text{W}}{\text{m}^2}$                                |
| PPFD_IN                | Photosynthetic photon flux density (incoming)     | $\frac{\mu \text{mol}_\text{photon}}{\text{m}^2\ \cdot\ \text{s}} $ |
| PPFD_OUT               | Photosynthetic photon flux density (outgoing)     | $\frac{\mu \text{mol}_\text{photon}}{\text{m}^2\ \cdot\ \text{s}} $ |
| WS                     | Wind speed                                        | $\frac{\text{m}}{\text{s}}$                                  |
| WD                     | Wind direction                                    | $\text{decimal}\ \degree \text{C}$                           |
| USTAR                  | Friction velocity                                 | $\frac{\text{m}}{\text{s}}$                                  |
| SWC_1                  | Soil water content (volumetric, 0-100) at level 1 | $\%$                                                         |
| SWC_2                  | Soil water content (volumetric, 0-100) at level 2 | $\%$                                                         |
| SWC_3                  | Soil water content (volumetric, 0-100) at level 3 | $\%$                                                         |
| SWC_4                  | Soil water content (volumetric, 0-100) at level 4 | $\%$                                                         |
| SWC_5                  | Soil water content (volumetric, 0-100) at level 5 | $\%$                                                         |
| TS_1                   | Soil temperature at level 1                       | $\degree \text{C}$                                           |
| TS_2                   | Soil temperature at level 2                       | $\degree \text{C}$                                           |
| TS_3                   | Soil temperature at level 3                       | $\degree \text{C}$                                           |
| TS_4                   | Soil temperature at level 4                       | $\degree \text{C}$                                           |
| TS_5                   | Soil temperature at level 5                       | $\degree \text{C}$                                           |
| WTD                    | Water table depth                                 | $\text{m}$                                                   |
| G                      | Soil heat flux                                    | $\frac{\text{W}}{\text{m}^2}$                                |
| H                      | Sensible heat turbulent flux                      | $\frac{\text{W}}{\text{m}^2}$                                |
| LE                     | Latent heat turbulent flux                        | $\frac{\text{W}}{\text{m}^2}$                                |

Now, let's find the corresponding variables in ERA5. There is 3 possible cases:

1. The variable is natively supported by ERA5
2. The variable in ERA5 needs to be processed (it's in a vector or it doesn't have quite the same units)
3. The variable is not supported (need to look elsewhere)

Legend for each column:

- Column 1: CarbonSense predictors from the above table.
- Column 2: Corresponding case from the above list.
- Column 3: Potential ERA5 corresponding predictors, depending on the case. If blank, it's because we are in `case 3`. If the variables are from CarbonSense, it's just a way to say "we re-use the values from these variables".
- Column 4: Modification that needs to be done on the data so it match CarbonSense variables context. If blank, we are in `case 1`.

| CARBONSENSE                    | CASE | ERA5                                                         | MODIFICATION                                                 |
| ------------------------------ | :--- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| TA                             | 1    | 2m temperature $(\text{K})$                                  | Convert $\degree \text{K}$ to $\degree \text{C}$             |
| P                              | 1    | Total precipitation $(\text{m})$                             |                                                              |
| RH                             | 2    | `TA` & 2m dewpoint temperature $(\text{K})$                  | Since we want a percentage, we will need to use https://arc.net/l/quote/lrazgyii |
| VPD                            | 2    | `RH` & `TA`                                                  | We can obtain with this formulae https://arc.net/l/quote/wnuzoezk and obtain $e_s$ with https://arc.net/l/quote/rxilyjwf if $\text{TA}>0$. Convert `TA` in celsius first and finally in  $\text{hPa}$ |
| PA                             | 1    | Surface pressure $(\text{Pa})$                               | Convert $\text{Pa}$ to $\text{kPa}$                          |
| CO2                            | 3    |                                                              | Look at https://cds.climate.copernicus.eu/datasets/satellite-carbon-dioxide?tab=overview? There's only dry air but we have to convert to wet air[^fn1] |
| SW_IN                          | 1    | Mean surface downward short-wave radiation flux $(\frac{\text{W}}{\text{m}^2})$ |                                                              |
| SW_IN_POT                      | 1    | Mean surface downward short-wave radiation flux, clear sky $(\frac{\text{W}}{\text{m}^2})$ |                                                              |
| SW_OUT                         | 2    | `SW_IN` & Forecast albedo (Dimensionless)                    | Albedo value ($\in [0,1]$) indicate the reflectivity $\rightarrow$ multiplied by `SW_IN`, we have what goes back in space |
| LW_IN                          | 1    | Mean surface downward long-wave radiation flux $(\frac{\text{W}}{\text{m}^2})$ |                                                              |
| LW_OUT                         | 2    | `LW_IN` & Forecast albedo (Dimensionless)                    | Same thing as in `SW_OUT`                                    |
| NETRAD                         | 2    | `SW_IN`, `SW_OUT`, `LW_IN` & `LW_OUT`                        | Use this formulae https://arc.net/l/quote/sjdmjlax           |
| PPFD_IN                        | 3    |                                                              | ?? Supposed bug: https://arc.net/l/quote/jvbszbju. Maybe find other dataset but it's a pretty niche variable |
| PPFD_OUT                       | 3    |                                                              | Same as in `PPFD_IN`                                         |
| WS                             | 2    | 10m v-component of wind $(\frac{m}{s})$ & 10m u-component of wind $(\frac{m}{s})$ | Calculate the magnitude with $\text{WS} = \sqrt{u^2 + v^2}$  |
| WD                             | 2    | 10m v-component of wind $(\frac{m}{s})$ & 10m u-component of wind $(\frac{m}{s})$ | Calculate the direction with $\text{WD} = \tan^{-1}(\frac{v}{u})$ |
| USTAR                          | 1    | Friction velocity $(\frac{m}{s})$                            |                                                              |
| SWC_1 (at 2 cm under surface)  | 1    | Volumetric soil water layer 1 (Dimensionless)                | Multiply by 100                                              |
| SWC_2 (at 5 cm under surface)  | 1    | Volumetric soil water layer 1 (Dimensionless)                | Multiply by 100                                              |
| SWC_3 (at 10 cm under surface) | 1    | Volumetric soil water layer 2 (Dimensionless)                | Multiply by 100                                              |
| SWC_4 (at 20 cm under surface) | 1    | Volumetric soil water layer 2 (Dimensionless)                | Multiply by 100                                              |
| SWC_5 (at 30 cm under surface) | 1    | Volumetric soil water layer 3 (Dimensionless)                | Multiply by 100                                              |
| TS_1 (at 2 cm under surface)   | 1    | Soil temperature level 1 $(\text{K})$                        | Convert $\degree \text{K}$ to $\degree \text{C}$             |
| TS_2 (at 5 cm under surface)   | 1    | Soil temperature level 1 $(\text{K})$                        | Convert $\degree \text{K}$ to $\degree \text{C}$             |
| TS_3 (at 10 cm under surface)  | 1    | Soil temperature level 2 $(\text{K})$                        | Convert $\degree \text{K}$ to $\degree \text{C}$             |
| TS_4 (at 20 cm under surface)  | 1    | Soil temperature level 2 $(\text{K})$                        | Convert $\degree \text{K}$ to $\degree \text{C}$             |
| TS_5 (at 20 cm under surface)  | 1    | Soil temperature level 3 $(\text{K})$                        | Convert $\degree \text{K}$ to $\degree \text{C}$             |
| WTD                            | 3    |                                                              | Look here https://github.com/UU-Hydro/GLOBGM                 |
| G                              | 2    | `NETRAD`, `LE` & `H` $(\frac{W}{m^2})$                       | The formula is $\text{G} = \text{NETRAD} - \text{H} - \text{LE}$. Look here https://iahs.info/uploads/dms/16743.28-140-144-343-10-Jansen.pdf |
| H                              | 1    | Mean surface sensible heat flux $(\frac{W}{m^2})$                 |                                                              |
| LE                             | 1    | Mean surface latent heat flux $(\frac{W}{m^2})$                   |                                                              |

Wind speed, as said in the variable description of ERA5, vary on small space and time scales and are affected by the local terrain, vegetation and buildings. So we are not considering the neutral wind because it ignores too much the context where the EC towers are placed.

[^fn1]: With this video https://www.youtube.com/watch?v=uiUll-xAFMc, I was able to figure out something. The initial supposition is that in the air, the only elements that can vary is the water (humidity) and carbon dioxide (GHG). Go to Annex 1.

# Coding

## Optimizing API calls (not valid, see UPDATE)

If you group columns by any shared missing timestamp (union-find), you end up with overly coarse clusters. For example, if column X is missing every hour, it unions with column Y (missing only 1/7 of hours), and then chains through Y to every other variable that shares at least one hole—so you’ll request X data even on days when you only needed Y.

**Instead**, build a “missing-variable set” *per row* and then group rows by that set. Take for example this toy dataset with three variables A, B, C and hourly timestamps:

| Index |  A   |  B   |  C   | timestamp |
| :---- | :--: | :--: | :--: | :-------: |
| 1     |  a   |      |  c   |   00:00   |
| 2     |  c   |      |      |   01:00   |
| 3     |  b   |      |  b   |   02:00   |
| 4     |  g   |      |      |   03:00   |
| 5     |      |      |      |   04:00   |
| 6     |  e   |      |  d   |   05:00   |

**Group 1** $\rightarrow$ would have the rows 1, 3 and 6 with the timestamps [00:00, 02:00, 05:00]
**Group 2** $\rightarrow$ would have the rows 2 and 4 with the timestamps [01:00, 03:00]
**Group 3** $\rightarrow$ would have the row 5 with the timestamp [04:00]

### UPDATE

This isn't working neither. Grouping like this makes too many API calls even if we coarse to the max. Will not make sense. Will follow for now with the original approach of getting fetching for all the variables.

## Output

The wanted target goal is to give back the same original given dataset but instead of having a simple table of predictors, the table would have a hierarchical structure where each predictor is a category (first level of column MultiIndex) and the subcolumns (second level of column MultiIndex) would be the values from

1. Original dataset
2. ERA5 processed"[v]

# Annex 1

We know the relative humidity `RH`, the atmospheric temperature `TA` (at 2 meters) and the surface pressure `PA`. We also know the molar fraction of carbon dioxide in dry air $X^\text{dry air}_{\text{CO}_2}\ [\mu \text{mol}/\text{mol}]$ given by the other dataset (see the corresponding link in the Modification column). Thus, we can find the fraction of $\text{H}_2\text{O}$ in dry air.
$$
x^\text{wet air}_{\text{H}_2\text{O}} = \frac{p_{\text{H}_2\text{O}}}{\text{PA}} = \frac{\text{RH}\ \cdot\ p_s(\text{TA})\ [\text{Pa}]}{\text{PA}}, \quad \text{where}\ p_s(\text{TA}) = 10^{A - \frac{B}{C + \text{TA}}}\ [\text{mmHg}] \\
x^\text{dry air}_{\text{H}_2\text{O}} = x^\text{wet air}_{\text{H}_2\text{O}}\ \cdot\ (x^\text{wet air}_{\text{dry air}})^{-1} = \frac{\text{\# moles H}_2\text{O}}{\text{1 mole total wet air}}\ \cdot\ \frac{\text{1 mole total wet air}}{\text{\# moles dry air}}
$$
We know that the composition of dry air is :

- 0.7808 moles of $\text{N}_2$
- 0.2095 moles of $\text{O}_2$
- 0.0093 moles of $\text{Ar}$
- $x^\text{dry air}_{\text{CO}_2} = X^\text{dry air}_{\text{CO}_2}/10^6$ moles
- $x^\text{dry air}_{\text{H}_2\text{O}}$ moles

The sum would be $>1\ \text{mole}$ and denoted as $n_\text{tot}$. Thus, we find that
$$
X^\text{wet air}_{\text{CO}_2}=\frac{X^\text{dry air}_{\text{CO}_2}}{n_\text{tot}}
$$
For `(1)`, see https://en.wikipedia.org/wiki/Antoine_equation.

Note: The CO2 dataset has a larger resolution than ERA5 on single levels. Maybe something has to be done here beforehand?

