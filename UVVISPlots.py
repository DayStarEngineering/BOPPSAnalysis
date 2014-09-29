#!/usr/bin/python
import matplotlib.pyplot as plt
import pylab
import numpy as np
import matplotlib
from ZD_Utils import DataFrameUtils as dfutil
from PyDyGraphs import pydygraphs
import UVVISAnalysisTools as uva
import pandas as pd

def plotCentroids2D(header,data):
    # Make plot sizes large:
    figureSizeX, figureSizeY = 16,6
    pylab.rcParams['figure.figsize'] = (8, 8)

    roiLeft = header["roiLeft"][0]
    roiTop = header["roiTop"][0]
    roiWidth = header["roiWidth"][0]
    roiHeight = header["roiHeight"][0]

    avedt = np.mean(data["dt"])
    sampFreq = 1.0/avedt

    # Centroids
    ax = plt.figure().add_subplot(111, aspect='equal')
    ax.set_title( str(header["numPts"][0]) + ' Centroid Positions at ' + str(int(roiWidth)) + "x" + str(int(roiHeight)) + (" Crop (%.2f Hz)" % sampFreq) )
    ax.plot(data["Az Motion"]+roiLeft, data["El Motion"]+roiTop, 'k.', label="Input")
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

    # Centroids
    ax = plt.figure().add_subplot(111, aspect='equal')
    ax.set_title( str(header["numPts"][0]) + ' Centroid Positions at ' + str(int(roiWidth)) + "x" + str(int(roiHeight)) + (" Crop (%.2f Hz)" % sampFreq) + " Zoomed")
    ax.plot(data["Az Motion"]+roiLeft, data["El Motion"]+roiTop, 'k.', label="Input")
    ax.plot(data["Az Residuals"]+roiLeft, data["El Residuals"]+roiTop, 'b.', label="Residuals")
    ax.plot(data["Initial X"]+roiLeft, data["Initial Y"]+roiTop, 'r*', label="Target")
    ax.plot([roiLeft, roiLeft+roiWidth], [data["Initial Y"]+roiTop, data["Initial Y"]+roiTop], 'r',  linewidth=0.5)
    ax.plot([data["Initial X"]+roiLeft, data["Initial X"]+roiLeft], [roiTop, roiTop+roiHeight], 'r',  linewidth=0.5)
    ax.set_xlabel('X Position [pixels]')
    ax.set_ylabel('Y Position [pixels]')
    ax.set_xlim([min(data["Az Motion"].min(),data["Az Residuals"].min())+roiLeft-15, max(data["Az Motion"].max(),data["Az Residuals"].max())+roiLeft+15])
    ax.set_ylim([min(data["El Motion"].min(),data["El Residuals"].min())+roiTop-15, max(data["El Motion"].max(),data["El Residuals"].max())+roiTop+15])
    ax.grid()
    plt.legend()
    #ax.axis('equal')
    formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_formatter(formatter)

def plotControllawSpeed(header,data):
    ([f1],[f2],[f3]) = pydygraphs.subplot(3,1, height=650, title="Controllaw Speed Characterization")

    df = dfutil.colmask(data,['Time', 'dt'])
    df['dt'] = df['dt'].apply(lambda x: x/1000.0/1000.0)
    aveFreq = 1/(df['dt'].apply(lambda x: x/1000.0)).mean()
    f1.plotDataFrame(df, xaxis = 'Time', color='orange')
    f1.title('Delta Time Step (' + str(aveFreq) + ' Hz)')
    f1.ylabel('Delta Time Step [ms]')
    f1.xlabel('Time [s]')
    f1.show()

    df = dfutil.colmask(data,['Time','NumStarsFound'])
    f2.plotDataFrame(df,xaxis='Time',color='green');
    f2.xlabel('Time [s]'); f2.ylabel('Stars Found')
    f2.title('Number of Stars Found (Average=' + str(df['NumStarsFound'].mean()) + ')')
    f2.show()

    df = dfutil.colmask(data,['Time','LostStarCount'])
    f3.plotDataFrame(df,xaxis='Time',color='purple');
    f3.xlabel('Time [s]'); f2.ylabel('Stars Lost')
    f3.title('Lost Star Count (Total=' + str(df['LostStarCount'].max()) + "/" + str(df['LostStarCount'].shape[0]) + ")")
    f3.show()

def plotControllawNumberofPoints(header,data):
    ([f1],[f2],[f3]) = pydygraphs.subplot(3,1, height=600, title="Controllaw Number of Points")

    try:
        df = dfutil.colmask(data,['Time','Az Num Points','El Num Points'])
        f1.plotDataFrame(df, xaxis = 'Time', color = ['gray','black'], showroller=False)
        f1.title('Number of Points In PF')
        f1.ylabel('Number of Points')
        f1.xlabel('Time [s]')
        f1.show()
    except:
         pass

    try:
        df = dfutil.colmask(data,['Time','Az Filter Points','El Filter Points'])
        f2.plotDataFrame(df, xaxis = 'Time', color = ['magenta','green'], showroller=False)
        f2.title(('Number of Filter Points (Average=%.1f,%0.1f)' % (np.mean(data['Az Filter Points']),np.mean(data['El Filter Points']))))
        f2.ylabel('Number of Points')
        f2.xlabel('Time [s]')
        f2.show()
    except:
        pass

    try:
        df = dfutil.colmask(data,['Time','Az Poly Points','El Poly Points'])
        f3.plotDataFrame(df, xaxis = 'Time', color = ['navy','blue'], showroller=False)
        f3.title(('Number of Poly Points Fit @ Order %d (Average=%.1f,%0.1f)' % (header['slowPredictionOrder'][0], np.mean(data['Az Poly Points']),np.mean(data['El Poly Points']))))
        f3.ylabel('Number of Points')
        f3.xlabel('Time [s]')
        f3.show()
    except:
         pass

def plotBackgroundCharacterization(header,data):
    ([f1],[f2]) = pydygraphs.subplot(2,1, height=700, title="Background Characterization")

    df = dfutil.colmask(data,['Time','Mean', 'Limit','PeakLimit','MaxValBrightest'])
    df['Average Star Brightness'] = data['IWBrightest']/data['NumPixBrightest']+data['Mean']
    df['Sigma 2.0'] = data['Std']*2.0+data['Mean']
    df['Sigma 3.0'] = data['Std']*3.0+data['Mean']
    df['Sigma 5.0'] = data['Std']*5.0+data['Mean']
    f1.plotDataFrame(df, xaxis = 'Time', color = ['black','blue','orange','red','green','gray','black','gray'])
    f1.title(('Background Characteristics (Average=%0.1f+-%0.1f)' % (np.mean(data['Mean']),np.mean(data['Std']))))
    f1.ylabel('Value')
    f1.xlabel('Time [s]')
    f1.show()

    df = dfutil.colmask(data,['Time','NumGoodPix', 'BrightPixelCount', 'NumBackgroundPixels'])
    f2.plotDataFrame(df, xaxis = 'Time', color = ['blue','red','black'])
    f2.title('Pixel Counts at SigmaReject=' + str(data['SigmaReject'][0]) + \
                            ', SigmaThresh=' + str(header['sigmaThresh'][0]))
    f2.ylabel('Number of Pixels')
    f2.xlabel('Time [s]')
    f2.show()

def plotStarCharacterization(header,data):
    ((f1,f2),(f3,f4)) = pydygraphs.subplot(2,2, height=700, title="Star Characterization")

    df = dfutil.colmask(data,['Time','MinPixThresh', 'NumPixBrightest','MaxPixThresh'])
    f1.plotDataFrame(df, xaxis = 'Time', color = ['red','green'])
    f1.title(('Number of Pixels in Star (Average=%.1f)' % np.mean(data['NumPixBrightest'])))
    f1.ylabel('Number of Pixels')
    f1.xlabel('Time [s]')
    f1.show()

    df = dfutil.colmask(data,['Time','WidthBrightest', 'HeightBrightest'])
    f2.plotDataFrame(df, xaxis = 'Time', color = ['navy','magenta'])
    f2.title(('Star Size (Average=%.1fx%0.1f)' % (np.mean(data['WidthBrightest']),np.mean(data['HeightBrightest']))))
    f2.ylabel('Length [pixels]')
    f2.xlabel('Time [s]')
    f2.show()

    df = dfutil.colmask(data,['Time','RoundnessBrightest'])
    f3.plotDataFrame(df, xaxis = 'Time', color = ['purple'])
    f3.title('Star Roundness at OblongRatio=' + str(header['oblongRatio'][0]) + (' (Average=%.1f)' % np.mean(data['RoundnessBrightest'])))
    f3.ylabel('Roundness Ratio')
    f3.xlabel('Time [s]')
    f3.show()

    df = dfutil.colmask(data,['Time','FalseStarCount','NumBlobLowPeak','NumBlobOnEdge',\
                              'NumBlobSaturated','NumBlobTooBig','NumBlobTooOblong','NumBlobTooSmall'])
    f4.plotDataFrame(df, xaxis = 'Time', color = ['red','blue','green','pink','magenta','orange','purple'])
    f4.title(('False Stars Found (Average=%.1f)' % np.mean(data['FalseStarCount'])))
    f4.ylabel('Number of False Stars')
    f4.xlabel('Time [s]')
    f4.show()

def plotFSMInputAndResiduals(header,data):
    [a],[b],[f2] = pydygraphs.subplot(3,1,title="FSM Performance",height=600)
    df1 = dfutil.colmask(data,['Time','Az Motion','Az Residuals'])
    df2 = dfutil.colmask(data,['Time','El Motion','El Residuals'])
    a.plotDataFrame(df1,'Time'); b.plotDataFrame(df2,'Time')
    df1['Az Residuals'].std()


    a.xlabel('Time [s]'); a.ylabel('Position [px]'); a.title(('Azimuth (%0.3f RMS)' % df1['Az Residuals'].std()))
    b.xlabel('Time [s]'); b.ylabel('Position [px]'); b.title(('Elevation (%0.3f RMS)' % df2['El Residuals'].std()))

    df = dfutil.colmask(data,['Time','NumStarsFound'])
    f2.plotDataFrame(df,xaxis='Time',color='green');
    f2.xlabel('Time [s]'); f2.ylabel('Stars Found')
    f2.title('Number of Stars Found (Average=' + str(df['NumStarsFound'].mean()) + ')')
    f2.show()

    a.show();b.show()

def plotFSMPowerSpectrums(header,data):
    (a,b,c),(d,e,f) = pydygraphs.subplot(2,3,title="Power Spectrum",height=600)
    (AzResFreq, AzResPower, (sampFreq, nyquistFreq)) = uva.powerSpectrum(data["Time"], data["Az Residuals"])
    (ElResFreq, ElResPower, (sampFreq, nyquistFreq)) = uva.powerSpectrum(data["Time"], data["El Residuals"])
    (AzMotionFreq, AzMotionPower, (sampFreq, nyquistFreq)) = uva.powerSpectrum(data["Time"], data["Az Motion"])
    (ElMotionFreq, ElMotionPower, (sampFreq, nyquistFreq)) = uva.powerSpectrum(data["Time"], data["El Motion"])


    d1=pd.DataFrame(data=dict(zip(['Az Motion Freq','Az Motion Power'],[AzMotionFreq,AzMotionPower])))
    d2=pd.DataFrame(data=dict(zip(['El Motion Freq','El Motion Power'],[ElMotionFreq,ElMotionPower])))
    a.plotDataFrame(d1, 'Az Motion Freq', color='black')
    d.plotDataFrame(d2, 'El Motion Freq', color='black')

    d3=pd.DataFrame(data=dict(zip(['Az Residual Freq','Az Residual Power'],[AzResFreq,AzResPower])))
    d4=pd.DataFrame(data=dict(zip(['El Residual Freq','El Residual Power'],[ElResFreq,ElResPower])))
    b.plotDataFrame(d3, 'Az Residual Freq', color='red')
    e.plotDataFrame(d4, 'El Residual Freq', color='red')

    d5=pd.DataFrame(data=dict(zip(['Az Attenuation Freq','Az Attenuation Power'],[AzResFreq,AzMotionPower-AzResPower])))
    d6=pd.DataFrame(data=dict(zip(['El Attenuation Freq','El Attenuation Power'],[ElResFreq,ElMotionPower-ElResPower])))
    c.plotDataFrame(d5, 'Az Attenuation Freq')
    f.plotDataFrame(d6, 'El Attenuation Freq')

    a.xlabel("Freq [Hz]"); a.ylabel("Az Motion Power"); a.show()
    b.xlabel("Freq [Hz]"); b.ylabel("Az Motion Power"); b.show()
    c.xlabel("Freq[Hz]"); c.ylabel("Az Residual Power"); c.show()
    d.xlabel("Freq [Hz]"); d.ylabel("El Residual Power"); d.show()
    e.xlabel("Freq [Hz]"); e.ylabel("El Attenuation Power"); e.show()
    f.xlabel("Freq [Hz]"); f.ylabel("El Attenuation Power"); f.show()

def plotMotionPowerSpectrums(header,data):
    [[a,b]] = pydygraphs.subplot(1,2,title="Power Spectrum",height=350)
    (AzResFreq, AzResPower, (sampFreq, nyquistFreq)) = uva.powerSpectrum(data["Time"], data["Az Residuals"])
    (ElResFreq, ElResPower, (sampFreq, nyquistFreq)) = uva.powerSpectrum(data["Time"], data["El Residuals"])
    (AzMotionFreq, AzMotionPower, (sampFreq, nyquistFreq)) = uva.powerSpectrum(data["Time"], data["Az Motion"])
    (ElMotionFreq, ElMotionPower, (sampFreq, nyquistFreq)) = uva.powerSpectrum(data["Time"], data["El Motion"])


    d1=pd.DataFrame(data=dict(zip(['Az Motion Freq','Az Motion Power'],[AzMotionFreq,AzMotionPower])))
    d2=pd.DataFrame(data=dict(zip(['El Motion Freq','El Motion Power'],[ElMotionFreq,ElMotionPower])))
    a.plotDataFrame(d1, 'Az Motion Freq', color='gray')
    b.plotDataFrame(d2, 'El Motion Freq', color='gray')

    a.xlabel("Freq [Hz]"); a.ylabel("Az Motion Power"); a.show()
    b.xlabel("Freq [Hz]"); b.ylabel("El Motion Power"); b.show()

def plotFSMSimulatedPerformance(header,data):
    [a],[b] = pydygraphs.subplot(2,1,title="Simulated FSM Performance")
    data['Relative Az Motion'] = data['Az Motion'] - data['Az Motion'].iloc[0]
    data['Relative El Motion'] = data['El Motion'] - data['El Motion'].iloc[0]
    data['Relative Az Prediction'] = data['predAz'] - data['predAz'].iloc[0]
    data['Relative El Prediction'] = data['predEl'] - data['predEl'].iloc[0]
    data['Relative Az Residuals'] = data['Az Residuals'] - data['Az Residuals'].iloc[0]
    data['Relative El Residuals'] = data['El Residuals'] - data['El Residuals'].iloc[0]
    # df1 = dfutil.colmask(data,['Time','Az Motion','predAz','Az Prediction Residuals'])
    # df2 = dfutil.colmask(data,['Time','El Motion','predEl','El Prediction Residuals'])
    df1 = dfutil.colmask(data,['Time','Relative Az Motion','Relative Az Prediction','Relative Az Residuals'])
    df2 = dfutil.colmask(data,['Time','Relative El Motion','Relative El Prediction','Relative El Residuals'])
    a.plotDataFrame(df1,'Time'); b.plotDataFrame(df2,'Time')
    a.show();b.show()

def plotPredictionCaseStudy(header,data):
    res,pred,filt,fit,resMean,resStd=uva.predictionPointCaseStudy(data["Timestamp"].values,data["Az Motion"].values,numFilters=20,cutoff=header['azFilterCutoffFrequency'][0], predOrder=header['slowPredictionOrder'][0], baseFilterPoints=header['numFilterPoints'][0])
    res['Time']=data['Time']
    [a],[b] = pydygraphs.subplot(2,1,title="Prediction Point Case Study")
    a.plotDataFrame(res,'Time'); b.plotDataFrame(resMean,'Filter Points')
    a.xlabel('Time [s]'); a.ylabel('Residual Error From Prediction [px]')
    b.xlabel('Filter Points');b.ylabel('Average error from predictions')
    a.show();b.show()

def plotCutoffCaseStudy(header,data):
    multiplicationLimit=2
    res,pred,filt,fit,resMean,resStd=uva.cutoffCaseStudy(data["Timestamp"].values,data["Az Motion"].values,numCutoffs=50,baseCutoff=header['azFilterCutoffFrequency'][0], predOrder=header['slowPredictionOrder'][0], filterPoints=header['numFilterPoints'][0],mult=multiplicationLimit)
    res['Time']=data['Time']
    [a],[b] = pydygraphs.subplot(2,1,title="Cutoff Frequency Case Study")
    a.plotDataFrame(res,'Time'); b.plotDataFrame(resMean,'Cutoff Frequency')
    a.xlabel('Time [s]'); a.ylabel('Residual Error From Prediction [px]')
    b.xlabel('Cutoff Frequency');b.ylabel('Average error from predictions')
    a.show();b.show()

def plotOrderCaseStudy(header,data):
    res,pred,filt,fit,resMean,resStd=uva.orderCaseStudy(data["Timestamp"].values,data["El Motion"].values,cutoff=header['azFilterCutoffFrequency'][0], filterPoints=header['numFilterPoints'][0])
    res['Time']=data['Time']
    [a],[b] = pydygraphs.subplot(2,1,title="Polynomial Order Case Study")
    a.plotDataFrame(res,'Time'); b.plotDataFrame(resMean,'Polynomial Order')
    a.xlabel('Time [s]'); a.ylabel('Residual Error From Prediction [px]')
    b.xlabel('Polynomial Order');b.ylabel('Average error from predictions')
    a.show();b.show()

def plotCentroidStability(header,data):
    ([f1],[f2],[f3]) = pydygraphs.subplot(3,1, height=600, title="Centroid Stability")

    df = dfutil.colmask(data,['Time', 'dt'])
    df['dt'] = df['dt'].apply(lambda x: x/1000.0/1000.0)
    aveFreq = 1/(df['dt'].apply(lambda x: x/1000.0)).mean()
    f1.plotDataFrame(df, xaxis = 'Time', color='orange')
    f1.title('Delta Time Step (' + str(aveFreq) + ' Hz)')
    f1.ylabel('Delta Time Step [ms]')
    f1.xlabel('Time [s]')
    f1.show()

    df = dfutil.colmask(data,['Time','NumStarsFound'])
    f2.plotDataFrame(df,xaxis='Time',color='green');
    f2.xlabel('Time [s]'); f2.ylabel('Stars Found')
    f2.title('Number of Stars Found (Average=' + str(df['NumStarsFound'].mean()) + ')')
    f2.show()

    df = dfutil.colmask(data,['Time','LostStarCount'])
    f3.plotDataFrame(df,xaxis='Time',color='purple');
    f3.xlabel('Time [s]'); f3.ylabel('Stars Lost')
    f3.title('Lost Star Count (Total=' + str(df['LostStarCount'].max()) + "/" + str(df['LostStarCount'].shape[0]) + " = "+ str(float(df['LostStarCount'].max())/float(df['LostStarCount'].shape[0])*100.0)+"%)")
    f3.show()

    