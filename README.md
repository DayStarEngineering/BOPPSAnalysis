BOPPSAnalysis
-----
*Southwest Research Institute*
![alt tag](http://www.boulder.swri.edu/clark/swrilogo.gif)

Repository hosting UVVIS Analysis tools used for parsing BOPPS mission data

## Intent

Run the notebooks. 

## Installation:
Perform a recursive clone of this repo in order to work locally. 

    git clone --recursive git@github.com:DayStarEngineering/BOPPSAnalysis.git

Once downloaded, start your analysis in two steps:

    cd dir/to/repo
    source ./setEnv.sh
    make clean && make
    ipython --notebook ControlLawAnalysis.ipynb