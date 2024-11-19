EARLY INSTRUMENTAL TEMPERATURE RECORDS FOR BASEL, BERN, GENEVA, AND ZURICH
Yuri Brugnara - Oeschger Centre for Climate Change Research - University of Bern
11-Aug-2022


Description
-----------

This data set is composed of processed daily and monthly air temperature data for four Swiss cities: Basel, Bern, Geneva, and Zurich. The data cover the pre-industrial period 1756-1863.

For each city the following folders are provided - number indicates processing level (lower numbers are the input of higher numbers):

- 1_daily: Daily means, corrected for observation times (one file per observer)
- 2_monthly: Monthly means calculated from the daily means (one file per observer)
- 3_homogenized: Homogenized daily and monthly means (two files per observer)
- 4_merged: Merged version of homogenized daily and monthly means (two files per city)
- 5_filled: Merged homogenized monthly means with missing values interpolated from other cities (one file per city)

in addition, the SwissPlateau folder contains series calculated as the arithmetic averages of the homogenized Bern and Zurich series.


Data Format
-----------

Data in the folders 1-2-3 are provided in the C3S Station Exchange Format (https://datarescue.climate.copernicus.eu/st_formatting-land-stations) - see attached file SEF_1.0.0.html.

Data in the folders 4-5 are provided as simple csv files and include standard errors. Daily files also include data source and observation times for each day. Units are degrees Celsius.


License
-------
The data are provided under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. For more details see https://creativecommons.org/licenses/by/4.0/


References
-----------
Brugnara Y, et al (2022): Pre-industrial Temperature Variability on the Swiss Plateau Derived from the Instrumental Daily Series of Bern and Zurich, Clim. Past Discuss. [preprint], https://doi.org/10.5194/cp-2022-34, in review

Brugnara Y, et al (2022): Revisiting the early instrumental temperature records of Basel and Geneva [in preparation]
