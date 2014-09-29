##! /usr/bin/python
__author__ = 'Zach Dischner, Kevin Dinkel'
__copyright__ = "NA"
__credits__ = ["NA"]
__license__ = "NA"
__version__ = "0.0.1"
__maintainer__ = "Zach Dischner"
__email__ = "zach.dischner@gmail.com"
__status__ = "Dev"

"""
File name: UVISAnalysisTools.py
Authors: Zach Dischner
Created: 7/21/2014
Modified:7/21/2014


Todos:
*Make it work and stuff dood

"""
NUMPTSTOFILTER = 128
# -------------------------
# --- IMPORT AND GLOBAL ---
# -------------------------
import pandas as pd
import sys as sys
import numpy as np
import matplotlib.pyplot as plt
import os, string, fnmatch, argparse, thread, pickle, scipy, pylab,ctypes, bisect
import matplotlib
import matplotlib.pyplot as plt
from scipy import fftpack, pi
from ZD_Utils import DataFrameUtils as dfutil
from multiprocessing.pool import ThreadPool
import multiprocessing as mp
import binaryparser as bp

## Import the PyPredictiveFilter Wrapper
# Like `path.append('../')` but works everywhere
# sys.path.append("../")
# sys.path.append(string.join(os.path.abspath(__file__).split("/")[0:-2],"/"))
from predictiveFilter import PyPredictiveFilter as pypfilt

###### Some globally used variables
## Debugging on or off
DEBUG = False
## Standard Filter Configuration
PLAN="FFTW_ESTIMATE"

pddf=pd.DataFrame


def findMostRecentFile(search_dir,starts_with="controllaw",offset=0):
    #print starts_with
    # clfiles = os.listdir
    clfiles = [ f for f in os.listdir(search_dir) if f.startswith(starts_with)]    

    print clfiles
    clfiles.sort(key=lambda fn: os.path.getmtime(os.path.join(search_dir, fn)))
    print clfiles
    
    return clfiles[-1+offset]

def loadControlLawFile(fname, version=1, nofilter=True):
    """Load Controllaw file into a dataframe

    Args: 
    	fname: String with filename and path of controllaw file to open

    Kwargs:
    	version:    Prob won't be used. Should extract from filename

    Returns:
    	df:	Return dataframe representing controllaw data, with pertinent transformations applied

    Examples:
       	
    """
    ###### Get meta from filename
    clFilename = fname.split('/')[-1]
    print "Returning dataframes from " + str(clFilename) + "..."
    print "..."

    metaoffset = 1
    ## Extract version number
    version = fname.split("_")[-1].split(".")[0]
    if not fnmatch.fnmatch(version,"v*"):
        if len(fname.split("_")) > 2:
            version = "old"
        else:
            version = "reallyold..."
    if version == "v3" or version == "v4" or version == "v5":
        metaoffset = 2
 
    # Parse data in filename:
    if version == "v6":
        with open(fname,'r') as f:
            header = f.readline()

        ## Extract unix timestamp from file
        metaData = [fname.split("_")[2]]
        metaData += header.split(":")[1:].pop().split("\n")[0].split("_")
    else:
        metaData = '.'.join(clFilename.split('.')[0:-1]).split("_")[metaoffset:]

    df_h = pd.DataFrame()
    df_h['fileTimestamp'] = pd.Series(int(metaData[0]))
    df_h['numFilterPoints'] = pd.Series(int(metaData[1]))
    df_h['azFilterCutoffFrequency'] = pd.Series(float(metaData[2]))
    df_h['elFilterCutoffFrequency'] = pd.Series(float(metaData[3]))
    df_h['AzNumPolyPoints'] = pd.Series(int(metaData[4]))
    df_h['ElNumPolyPoints'] = pd.Series(int(metaData[5]))
    df_h['slowPredictionOrder'] = pd.Series(int(metaData[6]))
    df_h['exposureTime'] = pd.Series(float(metaData[7]))
    df_h['roiTop'] = pd.Series(float(metaData[8]))
    df_h['roiLeft'] = pd.Series(float(metaData[9]))
    df_h['roiWidth'] = pd.Series(float(metaData[10]))
    df_h['roiHeight'] = pd.Series(int(metaData[11]))
    df_h['sigmaReject'] = pd.Series(float(metaData[12]))
    df_h['sigmaThresh'] = pd.Series(float(metaData[13]))
    df_h['sigmaPeak'] = pd.Series(float(metaData[14]))
    df_h['minPixPerStar'] = pd.Series(int(metaData[15]))
    df_h['maxPixPerStar'] = pd.Series(int(metaData[16]))
    df_h['oblongRatio'] = pd.Series(float(metaData[17]))
    df_h['backgroundGridSize'] = pd.Series(int(metaData[18]))
    df_h['rejectOnEdgeStars'] = pd.Series(bool(metaData[19]))
    df_h['rejectSaturatedStars'] = pd.Series(bool(metaData[20]))
    df_h['subwindowFactor'] = pd.Series(float(metaData[21]))
    df_h['numPts'] = df_h.shape[0]
    df_h['filename'] = pd.Series(str(fname))
    df_h['nakedFilename'] = pd.Series(str(fname.split('/')[-1]))
    df_h['inputdir'] = pd.Series(str(string.join(fname.split('/')[0:-1],'/')))


    print "..."

    ## Extract version number
    version = fname.split("_")[-1].split(".")[0]
    if not fnmatch.fnmatch(version,"v*"):
        if len(fname.split("_")) > 2:
            version = "old"
        else:
            version = "reallyold..."

    ## Gather column names for loading dataframe. 
    if (version == "old"):			## Old file
        print "Loading old control law file with all the goodies in it"
        # controlNames =  ["Loop Iteration","Timestamp", "initialX", "initialY", "centX", "centY", "measuredAz", "measuredEl", \
        # 					"Az", "El", "predAz", "cmndAz", "DACAz", "DACEl", "filteredAz", "filteredEl","polyAz","polyEl"]  ## Yeah we should see what we need...
        controlNames =  ["Loop Iteration", "Timestamp","Initial X", "Initial Y", "currX", "currY", "errorX", "errorY", "actAz", "actEl", "predAz", "predEl", "DACAz", "DACEl", "filteredAz", "filteredEl","Az Poly Points","El Poly Points"]

        centroidNames = ["brightPixelCount","numBackgroundPixels","numGoodPix","mean","std","limit",'peaklimit','numStarsFound', \
        					'falseStarCount','numBlobSaturated','numBlobLowPeak','numBlobTooSmall','numBlobTooBig','numBlobTooOblong', \
            				'numBlobOnEdge','xCenterBrightest','yCenterBrightest','IWBBrightest','widthBrightest','heightBrightest','numPixBrightest', \
            				'roundnessBrightest','maxValBrightest','efyBrightest','subWinLeft','subWinTop','subWinRight','subWinBottom', \
            				'xCentroid','yCentroid','xCentroid2','yCentroid2','iCentroid']

        # Check number of columns in the file maybe
        cols = controlNames + centroidNames
        df = dfutil.import_csv(fname,column_names=cols,no_spaces=False)
        ####### Massage some datapoints
        df["Az Motion"] = df["actAz"]
        df["El Motion"] = df["actEl"]
        df["Az Residuals"] = df["errorX"]
        df["El Residuals"] = df["errorY"]
        func = lambda x: x/1e9
        df["Time"] = df["Timestamp"].apply(func)
        df["numFilterPoints"] = df_h["numFilterPoints"].iloc[0]

    elif ((version == "v2") or (version == "v3") or (version == "v4") or (version == "v5") or (version == "v6")):               

        if version == "v4":
            df = bp.parse(fname,'packetdefinition_v4.txt')
        elif ((version == "v5") or (version == "v6")):
            df = bp.parse(fname,'packetdefinition_v5.txt')
        else:
            df = bp.parse(fname,'packetdefinition_v2.txt')
        

        ###### Reconstruct some datapoints
        # xPrediction is the predicted location of mirror at this timestamp. Recreate true motion with xCentroid + xPrediction
        df["Timestamp"]=df["Timestamp"].astype(ctypes.c_ulong)
        df['Az Motion'] = df['Centroid X'] + df['Prediction X']
        df['El Motion'] = df['Centroid Y'] + df['Prediction Y']
        df['Az Residuals'] = df['Centroid X']
        df['El Residuals'] = df['Centroid Y']
        
        ## Run predictive filter on dataset
        if nofilter:
            print "NOT RUNNING PREDICTIVE FILTER ON DATA, USING MINIMAL DATASET"
            rA = np.zeros(len(df["Timestamp"]))
            pA = np.zeros(len(df["Timestamp"]))
            fA = np.zeros(len(df["Timestamp"]))
            nA = np.zeros(len(df["Timestamp"]))
            rE = np.zeros(len(df["Timestamp"]))
            pE = np.zeros(len(df["Timestamp"]))
            fE = np.zeros(len(df["Timestamp"]))
            nE = np.zeros(len(df["Timestamp"]))
        else:
            AzFilt,ElFilt = getFilters(numPointsToFilter=df_h['numFilterPoints'].iloc[0],multiThread=False)

            rA,pA,fA,nA,rE,pE,fE,nE = getAzElPredictionResiduals(df["Timestamp"].values,df["Az Motion"].values,df["El Motion"].values, AzFilter=AzFilt, \
                                    ElFilter=ElFilt, polyorder=df_h['slowPredictionOrder'].values, numPointsToFilter=df_h['numFilterPoints'].values, \
                                    cutoffFreq=df_h['azFilterCutoffFrequency'].values, showPlot=False,multiThread=False)
        df['filteredAz'] = fA
        df['filteredEl'] = fE
        df["predAz"] = pA
        df["predEl"] = pE
        df["Az Prediction Residuals"] = rA
        df["El Prediction Residuals"] = rE
        # Hack, just use Az for now
        df["numFilterPoints"] = nA

    else:
    	print "Version " + str(version) + " is not supported yet!"
    	return None

    ###### Convert timestamps into DTs
    df["dt"] = np.append(map(lambda x,y:x-y,df["Timestamp"].iloc[1:].tolist(),df["Timestamp"].iloc[:-1].tolist()),0)
    df["dt"].iloc[-1] = df["dt"].iloc[-2]
    # Basically does this but smart (complicated) er
    # time = np.array(df["Timestamp"].tolist())/1000000000.0
    # time = time - time[0]
    # dt = time[1:] - time[:-1]
    df["Time"] -= df["Time"].iloc[0]
    # df["dt"] = np.append(dt,0)

    print "..."
    print "Complete."
    return df, df_h

def predictionPointCaseStudy(time,dataset,numFilters=3,cutoff=1,predOrder=1,baseFilterPoints=128):

    filterPointAry = [baseFilterPoints + baseFilterPoints/numFilters*i for i in np.linspace(-numFilters,numFilters,numFilters)]

    Pool = mp.Pool(processes=2)
    evenNumRange = [2*n for n in np.arange(1000)+20]
    filterPointAry = [evenNumRange[bisect.bisect_left(evenNumRange,f)] for f in filterPointAry]
    print "Prediction case study for points: ",filterPointAry

    Results = []
    def extractResults(result):
        Results.append(result)

    for fp in filterPointAry:
        if DEBUG:
            print "processing numfilterpoints: ", fp
        Pool.apply_async(generatePredictionResiduals,(time,dataset), dict(polyorder=predOrder,numPointsToFilter=fp, cutoffFreq=cutoff, showPlot=False),callback=extractResults)
    Pool.close()
    Pool.join()

    return extractPFResults(filterPointAry,Results,conditionName='Filter Points')


def cutoffCaseStudy(time,dataset,numCutoffs=10,baseCutoff=1,predOrder=1,filterPoints=128,mult=8):  #step percent

    caseStudyAry = [baseCutoff + i for i in np.linspace(-baseCutoff*0.9,baseCutoff*mult,numCutoffs)]


    Pool = mp.Pool(processes=2)
    if DEBUG:
        print "Cutoff Freq case study for cutoff: ",baseCutoff

    Results = []
    def extractResults(result):
        Results.append(result)

    for c in caseStudyAry:
        if DEBUG:
            print "processing cutoff frequency: ", c
        Pool.apply_async(generatePredictionResiduals,(time,dataset), dict(polyorder=predOrder,numPointsToFilter=filterPoints, cutoffFreq=c, showPlot=False),callback=extractResults)
    Pool.close()
    Pool.join()

    return extractPFResults(caseStudyAry,Results,conditionName='Cutoff Frequency')


def orderCaseStudy(time,dataset,orders=[0,1,2,3,4],filterPoints=128,cutoff=1):  #step percent

    caseStudyAry = orders


    Pool = mp.Pool(processes=2)
    if DEBUG:
        print "Prediction order case study for orders: ",orders

    Results = []
    def extractResults(result):
        Results.append(result)

    for c in caseStudyAry:
        if DEBUG:
            print "processing cutoff frequency: ", c
        Pool.apply_async(generatePredictionResiduals,(time,dataset), dict(polyorder=c,numPointsToFilter=filterPoints, cutoffFreq=cutoff, showPlot=False),callback=extractResults)
    Pool.close()
    Pool.join()

    return extractPFResults(caseStudyAry,Results,conditionName='Polynomial Order')


def extractPFResults(conditions,async_results,conditionName="Points"):
    residuals = {}
    predictedPoints = {}
    filteredPoints = {}
    pointsToFit = {}
    rMean=[]
    rStd=[]
    
    for condition,result in zip(conditions,async_results):
        key = str(condition)
        residuals[key], predictedPoints[key], filteredPoints[key], pointsToFit[key] = result
        rMean.append(np.mean(np.abs(residuals[key])))
        rStd.append(np.std(residuals[key]))
        if DEBUG:
            print "Mean and SDT for ", condition,":\t[",rMean[-1],",",rStd[-1],"]"
    # for filterPoints in filterPointAry:
    #     residuals[filterPoints], predictedPoints[filterPoints], filteredPoints[filterPoints], pointsToFit[filterPoints] = \
    #         generatePredictionResiduals(time,dataset,polyorder=predOrder,numPointsToFilter=filterPoints, cutoffFreq=cutoff, showPlot=False)
    resMean = pddf(columns=[conditionName,'Mean Residual'],data=zip(conditions,rMean))
    resStd = pddf(columns=[conditionName,'Std Residual'],data=zip(conditions,rStd))
    return pddf(residuals),pddf(predictedPoints),pddf(filteredPoints),pddf(pointsToFit), resMean, resStd




def generatePredictionResiduals(time, dataset, polyorder=1, numPointsToFilter=NUMPTSTOFILTER, cutoffFreq=1, showPlot=True, pf=None):
    """Generate Residual plot for dataset. Basic for now

    Args:
        time:       time array
        dataset:    1D dataset tied to time

    Kwargs:
        polyorder:          Polynomial prediction order to use
        numPointsToFilter:  Number of points to use in the polynomial prediction
        cutoffFreq:         Cutoff Frequency of lowpass filter in Hz
        showPlot:           Show a plot of the results
        pf:                 Predictive Filter object (assumes this has been pre-configured...)

    Returns:
        residuals:          Residual difference between next the actual observation and what that 
                            observation was predicted to be
        predictedPoints:    Predicted observation point at each time. That is to say, 
                            predictedPoints[ii] is the result of calculations using obs[ii-numPtstoFilter-1:ii]
    Examples:
    	residuals,predictions = generatePredictionResiduals(time, azimuth)  
    """
    ###### Create and configure the predictive filter
    if pf is None:
        if DEBUG:
            print "Creating a predictive filter object"
        pf = pypfilt.PyPredictiveFilter()
        if DEBUG:
            print "Configuring filter to use " + str(numPointsToFilter) + " points in in the low pass filter..."
        ret = pf.configure(numPointsToFilter=numPointsToFilter)
        if ret:
            print "*** configure() failed! returned", ret
            return 
    else:
        polypoints = pf.getNumDataPointsToFit()
        numPointsToFilter = pf.getNumDataPointsToFilter()
        if DEBUG:
            print "Using existing predictive filter object"

    ###### Determine order and filter frequency
    ## Possibly transform data, find power spikes, guess at cutoff frequency...

    results=[]

    n = len(dataset)-numPointsToFilter-1

    ###### Loop through dataset, generate residuals 
    ii=0
    nextTime = time[1:]
    nextTime = np.append(nextTime,time[-1] + (time[-1]-time[-2]))
    for t,d,nt in zip(time,dataset,nextTime):# in np.arange(len(dataset)-1):#-numPointsToFilter-1):
        if ii < numPointsToFilter:
            tpolyorder = 0
        else:
            tpolyorder = polyorder
        if DEBUG:
            print "Adding dataset[",str(index),":",str(index),"]"

        ret = pf.addData(np.array([t]), np.array([d]))
        if ret:
            print "*** addData() failed! returned", ret
            return None

        results.append(getPrediction(pf, cutoffFreq,tpolyorder, nt))
            # predictedPoints[ii+1], filteredPoints[ii],numDataPointsToFilter[ii]
        if DEBUG:
            if np.mod(ii,100) == 0: print "Iter ", str(ii),"/",str(len(dataset))
        ii+=1

    ###### Unzip and offset result data
    predictedPoints,filteredPoints,numDataPointsToFit = map(lambda x: list(x), zip(*results))

    #Fake the 0th order first point to let graphs scale goodly
    predictedPoints.insert(0,dataset[0])
    predictedPoints.pop()
    # predictedPoints[1] = dataset[0]

    # predictedPoints,filteredPoints,numDataPointsToFit = map()
    ###### Finally, compute residuals  
    residuals = dataset - predictedPoints

    # print "Residual [mean, std]: [", str(np.mean(residuals[numPointsToFilter:])), ",",str(np.std(residuals[numPointsToFilter:])),"]"

    ###### Generate plot
    if showPlot:
        plotPrediction(time,dataset,predictedPoints)

    return residuals,predictedPoints,filteredPoints,numDataPointsToFit

def plotPrediction(time,data,residuals,predictions):
        plt.figure()
        plt.plot(time,data,'b',label='RawData')
        plt.plot(time,residuals,'r',label='Residual')
        plt.plot(time,predictions,'gs',label='Predicted Points')
        plt.grid()
        plt.legend()
        plt.show()

def getPrediction(pf,cutoffFreq,polyorder,predictionTime):
    """Adds a dataset to a predictive filter object, filters, generates prediction. 
    Provide full X and Y datasets to this mehthod, and an index at which to examine them. This method uses 
    points up to (index) to predict what dataset[time[index+1]] should be. 

    Lamens:
    	Conidering all the data up tillh index, what do you think the next point should be? 

    Args:
        time:				FULL time array. Used multiple times
        dataset:    		1D dataset of data tied to the time array
        numPointsToFilter:	Number of points used by the predictive filter
        pf:					PredictiveFilter object, already pre-configured
        cutoffFreq:			Freq in HZ at which to apply the lowpass filter
        polyorder:			Polynomial order to use when generating prediction

    Returns:
        p:	Point predicted by previous (numPointstoFilter) until (index), guessing at what should happen at time[index+1]
    Examples:
    	residuals,predictions = generatePredictionResiduals(time, azimuth)  
    """
    ## Apply filter
    ret = pf.filter(cutoffFreq, polyorder)
    if ret:
        print "*** filter() failed! returned", ret

    ## Get next predicted point at next deltaT timestep
    p = pf.getPrediction(predictionTime)
    f = pf.getFilteredData()
    try:
        f = f[-1]
    except:
        f = None
    n = pf.getNumDataPointsToFit()

    ## OOH SOMETIMES THERE IS A PROBREM HERE!!!
    if (abs(p) > 1000): #(Faster, this way is more correct though): > abs(5*np.std(dataset[ii:currIdx]) + np.mean(dataset[ii:currIdx]))):
        print "UH OH! PROBLEM WITH PREDICTION "
        ## ADD A VERTICAL LINE TO THE PLOT SO WE CAN TELL WHERE THIS HAPPENED?
        p=0
    # predictedPoints[currIdx+1] = p
    return p,f,n


def getAzElPredictionResiduals(time,Az,El,AzFilter, ElFilter, polyorder=1, numPointsToFilter=NUMPTSTOFILTER, 
    						cutoffFreq=1, showPlot=False, multiThread=True):
    """Get Az and El prediction residuals given an observation dataset. Baiscally just treads and wraps generatePredictionResiduals()

    Args:
        time:				1D Time array
        Az:					1D Azimuth measurement array
        El:					1D Elevation measurement array
        polyorder:          Polynomial prediction order to use
        numPointsToFilter:  Number of points to use in the polynomial prediction
        cutoffFreq:         Cutoff Frequency of lowpass filter in Hz
        showPlot:           Show a plot of the results

    Kwargs:
    	AzFilter:	Pre-configured Azimuth filter to use
    	ElFilter:	Pre-configured Elevation filter to use

    Returns:
        resAz:	     Azimuth residuals just due to the prediction at each time step
        predAz:	     Points predicted at each time by the predicted filter
        filteredAz:  Filtered data point at each time 
        numpoints:   Numpoints used in the filter 
        resEl:	     Elevation residuals just due to the prediction at each time step
        predEl:	     Points predicted at each time by the predicted filter
        filteredEl:  Filtered data point at each time 
        numpoints:   Numpoints used in the filter 

    Examples:
    	AzFilter = pypfilt.PyPredictiveFilter()
    	AzFilter.configure(numPointstoFilter=128)
    	ElFilter = pypfilt.PyPredictiveFilter()
    	ElFilter.configure(numPointstoFilter=128)
    	rA,pA,rE,pE = UVVISAnalysisTools.AzElPredictionResiduals(time_array, az_array, el_array, AzFilter=AzFilter, ElFilter=ElFilter)
    """
    def process(time, dataset, pf, polyorder, numPointsToFilter,cutoffFreq, showPlot):
        r,p,f,n = generatePredictionResiduals(time, dataset, polyorder=polyorder, numPointsToFilter=numPointsToFilter, 
        		cutoffFreq=cutoffFreq, showPlot=showPlot, pf=pf)
        return r,p,f,n

    if multiThread:
        # Start Az thread
        Pool = ThreadPool(processes=2)

        azResult = Pool.apply_async(process, (time,Az,AzFilter,polyorder,numPointsToFilter,cutoffFreq,False)) # tuple of args
        elResult = Pool.apply_async(process, (time,El,ElFilter,polyorder,numPointsToFilter,cutoffFreq,False)) # tuple of args


        rA,pA,fA,nA = azResult.get()  # get the return value from your function.
        rE,pE,fE,nE = elResult.get()
        Pool.close()
    else:
        rA,pA,fA,nA = process(time,Az,AzFilter,polyorder,numPointsToFilter,cutoffFreq,False)
        rE,pE,fE,nE = process(time,El,AzFilter,polyorder,numPointsToFilter,cutoffFreq,False)

    if showPlot:
        plotPrediction(time,Az,rA,pA)
        plotPrediction(time,El,rE,pE)

    return rA,pA,fA,nA,rE,pE,fE,nE


def getFilters(numPointsToFilter=NUMPTSTOFILTER, multiThread=False):
    # AzFilter = setupFilter(numPointsToFilter)
    # ElFilter = setupFilter(numPointsToFilter)
    # NUMPTSTOFILTER = numPointsToFilter
    Az = pypfilt.PyPredictiveFilter()
    Az.configure(numPointsToFilter=numPointsToFilter)
    El = pypfilt.PyPredictiveFilter()
    El.configure(numPointsToFilter=numPointsToFilter)
    return Az, El


def setupFilter(numPointsToFilter):
    pf = pypfilt.PyPredictiveFilter()
    print "numPtstofilter",numPointsToFilter
    pf.configure(numPointsToFilter=numPointsToFilter)
    return pf












#########################################################################################################
####################################### Functions for analysis ##########################################
#########################################################################################################

# Returns the power spectrum of the dataset in DB
def powerSpectrum(time, data):
    # returns frequency and power data  
    dt = np.zeros(time.size-1,)
    for i in range(0,time.size-2):
        dt[i] = time[i+1] - time[i] 
    avgdt = np.mean(dt)
    sampFreq = 1.0/(avgdt)
    nyquistFreq = sampFreq/2    
    
    FFTData = np.abs(scipy.fftpack.fft(data))   

    freqsData = scipy.fftpack.fftfreq(data.size, avgdt)
    freq = freqsData[0:len(freqsData)/2]        
    power = 20*scipy.log10(FFTData)
    power = power[0:len(freq)]
    return (freq,power,(sampFreq,nyquistFreq))

# Returns the power spectrum of the dataset in time-domain amplitudes
def amplitudePowerSpectrum(time,data):
    dt = np.zeros(time.size-1,)
    for i in range(0,time.size-2):
        dt[i] = time[i+1] - time[i] 
    avgdt = np.mean(dt)
    sampFreq = 1.0/(avgdt)
    nyquistFreq = sampFreq/2.0  
    
    FFTData = np.abs(scipy.fftpack.fft(data))   
    ## Only care about positive frequencies
    FFTData = FFTData[0:len(FFTData)/2]
    ## This is how we get the power spectrum in terms of time-domain amplitudes
    amplitudePower = FFTData/len(FFTData)

    freqsData = scipy.fftpack.fftfreq(data.size, avgdt)
    freq = freqsData[0:len(freqsData)/2]        
    
    return (freq,amplitudePower,(sampFreq,nyquistFreq))

def runningMean(x, N):
    # moves average forward in time by using later points for average
    y = np.zeros((len(x),))
    for ctr in range(len(x)):
        y[ctr] = np.sum(x[ctr:(ctr+N)])
    return y/N

def rollingMeanCentered(x, N):
    # center average taking N/2 points on either side
    # EVENS N's ONLY!
    if N%2 != 0:
        print "N has to be even for rollingMeanCentered!"
        exit()

    l = len(x)
    y = np.zeros(l,)
    for i in range(l-N):
        ctr = i + N/2
        y[ctr] = np.sum(x[i:(ctr+N/2)])
    return y/N

def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]
    # return np.convolve(x, np.ones((N,))/N, mode='valid')


def rollingFrequency(time, data, N, M): 
    # center average taking N/2 points on either side
    # EVENS N's ONLY!
    # M     
    plotIter = 0

    if N%2 != 0:
        print "N has to be even for rollingMeanCentered!"
        exit()

    l = len(data)
    rolledFreq = np.zeros(l,)
    for i in range(l-N):
        ctr = i + N/2
        (freq, power, (sampFreq, nyquistFreq)) = powerSpectrum(time[i:(ctr+N/2)], data[i:(ctr+N/2)])    
        # if plotIter%100 == 0:
        #   plt.figure()
        #   plt.plot(freq,power)
        # plotIter = plotIter + 1

        # unsmoothed verison
        # rolledFreq[ctr] = freq[np.argmax(power)] 
        # smoothed version
        smoothedPower = rollingMeanCentered(power,M)
        rolledFreq[ctr] = freq[np.argmax(smoothedPower[1:])+1]

    return rolledFreq

def rollingRMS(data,N):
    if N%2 != 0:
        print "N has to be even for rollingMeanCentered!"
        exit()

    l = len(data)
    y = np.zeros(l,)
    for i in range(l-N):
        ctr = i + N/2
        y[ctr] = np.std(data[i:(ctr+N/2)])
    return y




#^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*
#^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*
#^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*     Runtime Method     ^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*
#^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*
#^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*
#^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*

if __name__ == "__main__":
    plt.close("all")
    # Behavior Toggles
    PLOTMINIMUM = False
    SAVE = False
    PRINTSTATS = True
    ROLLINGFREQON = True
    LPFPLOTS = True
    SHOWAVERAGE=True
    PLOTPOLYPOINTS=True
    ## Standalone plotting variables variables
    numRollingPoints = 4
    numFFTPts = 1024 # Has to be sufficiently high!!! 512 is still a little low
    numSigma = 1.0

    parser = argparse.ArgumentParser(description='This utility parses and plots data from a BOPPS Control law file')
    parser.add_argument('fname', metavar='controllawfile', type=str, help='A controllaw file to ingest and analyze')
    args = parser.parse_args()
    fname = args.fname
    inputdir = string.join(fname.split('/')[0:-1],'/')
    clFilename = fname.split('/')[-1]

    ###### Load the data
    data,header = loadControlLawFile(fname)
    # fname = sys.argv[1]
    ############################################################################################
    ####################################### Calculations #######################################
    ############################################################################################
    numPts = data["Loop Iter"].size
    avedt = np.mean(data["dt"])
    sampFreq = 1.0/avedt
    # Mean and stds
    rollMeanOutAz = rollingMeanCentered(data["Az Residuals"], numRollingPoints)
    rollMeanOutEl = rollingMeanCentered(data["El Residuals"], numRollingPoints)
    rollMeanInAz = rollingMeanCentered(data["Az Motion"], numRollingPoints)
    rollMeanInEl = rollingMeanCentered(data["El Motion"], numRollingPoints)
    rollSTDOutAz = rollingRMS(data["Az Residuals"], numRollingPoints)
    rollSTDOutEl = rollingRMS(data["El Residuals"], numRollingPoints)
    rollSTDInAz = rollingRMS(data["Az Motion"], numRollingPoints)
    rollSTDInEl = rollingRMS(data["El Motion"], numRollingPoints)

    # Frequencies
    (freqOutAz, powerOutAz, (sampFreq, nyquistFreq)) = powerSpectrum(data["Time"], data["Az Residuals"])
    (freqOutEl, powerOutEl, (sampFreq, nyquistFreq)) = powerSpectrum(data["Time"], data["El Residuals"])
    rollpowerOutAz = rollingMeanCentered(powerOutAz, numRollingPoints)
    rollpowerOutEl = rollingMeanCentered(powerOutEl, numRollingPoints)
    (freqInAz, powerInAz, (sampFreq, nyquistFreq)) = powerSpectrum(data["Time"], data["Az Motion"])
    (freqInEl, powerInEl, (sampFreq, nyquistFreq)) = powerSpectrum(data["Time"], data["El Motion"])
    rollpowerInAz = rollingMeanCentered(powerInAz, numRollingPoints)
    rollpowerInEl = rollingMeanCentered(powerInEl, numRollingPoints)
    # Rolling freqs
    if ROLLINGFREQON:
        rollFreqInAz = rollingFrequency(data["Time"], data["Az Motion"], numFFTPts, numRollingPoints)
        rollFreqInEl = rollingFrequency(data["Time"], data["El Motion"], numFFTPts, numRollingPoints)

    ## Amplitude power spectrum
    (freqAmpInAz, ampPowerInAz, (sF, nF)) = amplitudePowerSpectrum(data["Time"], data["Az Motion"])
    (freqAmpInEl, ampPowerInEl, (sF, nF)) = amplitudePowerSpectrum(data["Time"], data["El Motion"])
    (freqAmpOutAz, ampPowerOutAz, (sF, nF)) = amplitudePowerSpectrum(data["Time"], data["Az Residuals"])
    (freqAmpOutEl, ampPowerOutEl, (sF, nF)) = amplitudePowerSpectrum(data["Time"], data["El Residuals"])
    rollAmpPowerInAz = rollingMeanCentered(ampPowerInAz, numRollingPoints/2)
    rollAmpPowerInEl = rollingMeanCentered(ampPowerInEl, numRollingPoints/2)
    rollAmpPowerOutAz = rollingMeanCentered(ampPowerOutAz, numRollingPoints/2)
    rollAmpPowerOutEl = rollingMeanCentered(ampPowerOutEl, numRollingPoints/2)

    ############################################################################################
    ####################################### Plotting ###########################################
    ############################################################################################
    roiLeft = header["roiLeft"][0]
    roiTop = header["roiTop"][0]
    roiWidth = header["roiWidth"][0]
    roiHeight = header["roiHeight"][0]

    # Centroids
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title( str(numPts) + ' Centroid Positions at ' + str(roiWidth) + "x" + str(roiHeight) + (" Crop (%.2f Hz)" % sampFreq) )
    ax.plot(data["Initial X"]+data["Az Motion"]+roiLeft, data["Initial Y"]+data["El Motion"]+roiTop, 'k.', label="Input")
    ax.plot(data["Az Residuals"]+roiLeft, data["El Residuals"]+roiTop, 'b.', label="Residuals")
    ax.plot(data["Initial X"]+roiLeft, data["Initial Y"]+roiTop, 'r*', label="Target")
    ax.plot([roiLeft, roiLeft+roiWidth], [data["Initial Y"]+roiTop, data["Initial Y"]+roiTop], 'r',  linewidth=0.5)
    ax.plot([data["Initial X"]+roiLeft, data["Initial X"]+roiLeft], [roiTop, roiTop+roiHeight], 'r',  linewidth=0.5)
    ax.set_xlabel('X Position [pixels]')
    ax.set_ylabel('Y Position [pixels]')
    ax.set_xlim([roiLeft,roiLeft+roiWidth])
    ax.set_ylim([roiTop,roiTop+roiHeight])
    ax.grid()
    plt.legend()
    #ax.axis('equal')
    formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_formatter(formatter)


    # Output Azimuth and Elevation being read in
    if not PLOTMINIMUM:
        plt.figure()

        plt.subplot(211)
        plt.title('Residuals')
        plt.plot(data["Time"], data["Az Residuals"], 'k', label="Residuals")
        if SHOWAVERAGE:
            plt.plot(data["Time"], rollMeanOutAz,'r', label="Averaged Residuals")
        pylab.fill_between(data["Time"], rollMeanOutAz+rollSTDOutAz*numSigma, rollMeanOutAz-rollSTDOutAz*numSigma, facecolor='yellow', alpha=0.5)
        plt.ylabel('Azimuth [pixels]')
        plt.grid()
        plt.legend()

        plt.subplot(212)
        plt.plot(data["Time"], data["El Residuals"], 'k', label="Residuals")
        if SHOWAVERAGE:
            plt.plot(data["Time"], rollMeanOutEl,'r', label="Averaged Residuals")
        pylab.fill_between(data["Time"], rollMeanOutEl+rollSTDOutEl*numSigma, rollMeanOutEl-rollSTDOutEl*numSigma, facecolor='yellow', alpha=0.5)
        plt.ylabel('Elevation [pixels]')
        plt.xlabel('Time [s]')
        plt.grid()

    # Input Motion
    if not PLOTMINIMUM:
        plt.figure()
        plt.subplot(211)
        plt.title('Input Motion')
        plt.plot(data["Time"], data["Az Motion"], 'k', label="Input")
        if SHOWAVERAGE:
            plt.plot(data["Time"], rollMeanInAz, 'b', label="Averaged Input")
        plt.ylabel('Azimuth [pixels]')
        plt.grid()
        plt.legend()
        plt.subplot(212)
        plt.plot(data["Time"], data["El Motion"], 'k', label="Input")
        if SHOWAVERAGE:
            plt.plot(data["Time"], rollMeanInEl, 'b', label="Averaged Input")
        plt.ylabel('Elevation [pixels]')
        plt.xlabel('Time [s]')
        plt.grid()

    # Averaged Error and Input Motion
    if not PLOTMINIMUM:
        plt.figure()
        plt.subplot(211)
        plt.title('Input Motion and Residuals')
        plt.plot(data["Time"], data["Az Motion"],'b',label="Input")
        plt.plot(data["Time"], data["Az Residuals"],'r',label="Residuals")
        plt.grid()
        plt.legend()
        plt.ylabel('Azimuth [pixels]')

        plt.subplot(212)
        plt.plot(data["Time"], data["El Motion"],'b')
        plt.plot(data["Time"], data["El Residuals"],'r')
        plt.grid()
        plt.ylabel('Elevation [pixels]')
        plt.xlabel('Time [s]')

    #if not PLOTMINIMUM:
    plt.figure()
    plt.title("Number of Poly Points for order: " + str(header["slowPredictionOrder"][0]))
    plt.plot(data["Time"],data["Az Poly Points"],'b',label='Azimuth')
    plt.plot(data["Time"],data["El Poly Points"],'r',label="Elevation")
    plt.xlabel('Time [s]')
    plt.ylabel('Number of Points')
    plt.grid()
    plt.legend()

    # Prediction based off Input Motion
    plt.figure()
    plt.subplot(211)
    plt.title('Input Motion, Predicted Position, and Residuals')
    plt.plot(data["Time"], data["Az Motion"], 'k', label="Input")
    plt.plot(data["Time"], data["filteredAz"], 'b', label='Filtered (' + str(header["azFilterCutoffFrequency"][0]) + ' Hz on ' + str(header["numFilterPoints"][0]) + ' points)')
    plt.plot(data["Time"], data["predAz"], 'g', label=("Prediction (order " + str(header["slowPredictionOrder"][0]) + (" on %d" % np.mean(data["Az Poly Points"][data["numFilterPoints"]])) + " avg points)"))
    plt.plot(data["Time"], data["Az Residuals"],'r', label="Residuals")
    plt.legend()
    plt.ylabel('Azimuth [pixels]')
    plt.grid()
    plt.subplot(212)
    plt.plot(data["Time"], data["El Motion"], 'k', label="Input")
    plt.plot(data["Time"], data["filteredEl"], 'b', label='Filtered (' + str(header["azFilterCutoffFrequency"][0]) + ' Hz on ' + str(header["numFilterPoints"][0]) + ' points)')
    plt.plot(data["Time"], data["predEl"], 'g', label=("Prediction (order " + str(header["slowPredictionOrder"][0]) + (" on %d" % np.mean(data["El Poly Points"][data["numFilterPoints"]])) + " avg points)"))
    plt.plot(data["Time"], data["El Residuals"],'r', label="Residuals")
    plt.legend()
    plt.ylabel('Elevation [pixels]')
    plt.xlabel('Time [s]')
    plt.grid()

    # DT
    plt.figure()
    plt.plot(data["Time"].iloc[1:-1],data["dt"].iloc[0:-2]*1000, 'k')
    plt.plot(data["Time"].iloc[1:-1],avedt*1000*np.ones(len(data["Time"][1:-1])),color='#ff00cc')
    pylab.fill_between(data["Time"].iloc[1:-1], avedt*1000 + np.std(data["dt"])*1000*np.ones(len(data["Time"][1:-1])), avedt*1000 - np.std(data["dt"])*1000*np.ones(len(data["Time"][1:-1])), facecolor='yellow', alpha=0.5)
    plt.title('Delta Time Step (average: %.2f +- %.2f ms, %.2f Hz)' % (avedt/1e6, np.std(data["dt"])/1e6, sampFreq))
    plt.ylabel('Delta Time Step [ms]')
    plt.xlabel('Time [s]')
    plt.grid()


    ############# Power Spectrum Analysis #######################
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex=True, sharey=True)
    ax1.plot(freqInAz,powerInAz, 'k', label="Power")
    if SHOWAVERAGE:
        ax1.plot(freqInAz,rollpowerInAz, 'b', label="Averaged Power")
    ax1.set_title('Input %.2f Hz Avg Sampling' % sampFreq)
    ax1.set_ylabel('Azimuth Power (dB)')
    ax1.grid()
    ax1.legend()

    ax2.set_title('Residual %.2f Hz Avg Sampling' % sampFreq)
    ax2.plot(freqOutAz,powerOutAz, 'k', label="Power")
    if SHOWAVERAGE:
        ax2.plot(freqOutAz,rollpowerOutAz, 'r', label="Averaged Power")
    ax2.grid()
    ax2.legend()

    ax3.set_title('Input Attenuated (Input - Residual)')
    ax3.plot(freqOutAz,powerInAz - powerOutAz, 'k', label="Power")
    if SHOWAVERAGE:
        ax3.plot(freqOutEl,rollpowerInAz - rollpowerOutAz, 'm', label="Averaged Power")
    ax3.grid()
    ax3.legend()

    ax4.set_ylabel('Elevation Power (dB)')
    ax4.set_xlabel('Freq [Hz]')
    ax4.plot(freqInEl,powerInEl, 'k', label="Power")
    if SHOWAVERAGE:
        ax4.plot(freqInEl,rollpowerInEl, 'b',label="Averaged Power")
    ax4.grid()

    ax5.plot(freqOutEl,powerOutEl, 'k', label="Power")
    if SHOWAVERAGE:
        ax5.plot(freqOutEl,rollpowerOutEl, 'r', label="Averaged Power")
    ax5.set_xlabel('Freq [Hz]')
    ax5.grid()

    ax6.plot(freqOutEl,powerInEl - powerOutEl, 'k', label="Power")
    if SHOWAVERAGE:
        ax6.plot(freqOutEl,rollpowerInEl - rollpowerOutEl, 'm', label="Averaged Power")
    ax6.set_xlabel('Freq [Hz]')
    ax6.grid()
    f.subplots_adjust(hspace=0)
    f.subplots_adjust(wspace=0)     

    if PRINTSTATS:
        printStats(data,header)

    #################### Show plots ######################
    plt.show()

def printStats(header, data):

    print "----------------- File Info: ----------------"
    print "          File Name:",header.nakedFilename.iloc[0] 
    print "           File Dir:",header.inputdir.iloc[0] 
    print "     File timestamp:",header["fileTimestamp"].iloc[0] 
    print "   Number of images:",header.numPts.iloc[0] 
    print "---------------- Image Info: ----------------"
    print "      Exposure Time:",header["exposureTime"].iloc[0] ,"s"
    print "         Image Size:",header['roiWidth'].iloc[0] ,"x",header['roiHeight'].iloc[0]  
    print "   Number of Pixels:",header['roiWidth'].iloc[0] *header['roiHeight'].iloc[0] 
    print "------------- Centroid Inputs: --------------"
    print "    Sigma Threshold:",header.sigmaThresh.iloc[0] 
    print "         Sigma Peak:",header.sigmaPeak.iloc[0] 
    print "       Sigma Reject:",header.sigmaReject.iloc[0]
    print "         Oblongness:",header.oblongRatio.iloc[0]
    print "Minimum Pixel Count:",header.minPixPerStar.iloc[0]
    print "Maximum Pixel Count:",header.maxPixPerStar.iloc[0]
    print "  BG Grid Step Size:",header.backgroundGridSize.iloc[0]
    print "   Subwindow Factor:",header.subwindowFactor.iloc[0]
    print "     Reject On Edge:",header.rejectOnEdgeStars.iloc[0]
    print "   Reject Saturated:",header.rejectSaturatedStars.iloc[0]
    print "----------- Control Law Inputs: ------------"
    print "   Points to Filter:",data.numFilterPoints.iloc[0]
    print "          Az Cutoff:",header.azFilterCutoffFrequency.iloc[0],"Hz"
    print "          El Cutoff:",header.elFilterCutoffFrequency.iloc[0],"Hz"
    print "AzOverride Poly Pts:",header.AzNumPolyPoints.iloc[0]
    print "ElOverride Poly Pts:",header.ElNumPolyPoints.iloc[0]
    print "    Slow Loop Order:",header.slowPredictionOrder.iloc[0] 
    print "--------------- Statistics: ----------------"
    print " Average Time Delta: %.2f +- %.2f ms" % (np.mean(data["dt"])*1000,np.std(data["dt"].iloc[0])*1000)
    print "    Average Azimuth: %.2f +- %.2f pixels" % (np.mean(np.abs(data["Az Residuals"])),np.std(data["Az Residuals"]))
    print "  Average Elevation: %.2f +- %.2f pixels" % (np.mean(np.abs(data["El Residuals"])),np.std(data["El Residuals"]))
