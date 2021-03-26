#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 13:47:47 2018

@author: courtneyklein
"""

import matplotlib.pyplot as plt
import seaborn as sns
import csv
import numpy as np
import pandas as pd
import tables as pt
from FINDflare import FINDflare as ff
import os

sns.set_style("whitegrid")
sns.set_context("talk")

def loaddata(matchfile):
    mtfl = pt.open_file(matchfile, 'r', root_uep='/', filters='lzo')
    sourcedata = pd.read_hdf(matchfile, key="matches/sourcedata")
    #transientdata = pd.read_hdf(matchfile, key="matches/transientdata")
    sourcesObject = mtfl.get_node("/matches", "sources")
    #transientsObject = mtfl.get_node("/matches", "transients")
    sources_columns = pd.DataFrame.from_records(sourcesObject.read(0,0))
    sources_columns_list = sources_columns.columns.tolist()
    sources = pd.DataFrame.from_records(sourcesObject.cols, columns=sources_columns_list)
    #transients_columns = pd.DataFrame.from_records(transientsObject.read(0,0))
    #transients_columns_list = transients_columns.columns.tolist()
    #transients = pd.DataFrame.from_records(transientsObject.cols, columns=transients_columns_list)
    sourcedata.sort_values('mjd', inplace=True)
    #transientdata.sort_values('mjd', inplace=True)
    mtfl.close()
    return sources, sourcedata#, transients, transientdata


def plotflare(file, outdur, idf, lightcurve, typecurve, indx, show=True):
    '''
    Creates a plot of the light curve and a zones in plot of the flare
    file: the match file name
    outdur: is a string
    idf: the id of the light curve
    '''
    fig = plt.figure(figsize=(10, 4.5))
    grid = plt.GridSpec(1, 3, hspace=1, wspace=.5)
    ax = fig.add_subplot(grid[0:1, 0:2])
    ax.errorbar(lightcurve['mjd'], lightcurve['mag'], lightcurve['magerr'],
                marker='o', markersize=3, c="black", alpha=0.15)
    ax.plot(lightcurve['mjd'].values[indx], lightcurve['mag'].values[indx],
            linestyle="none", marker='o', markersize=4, c="red")
    ax.set_title(f'Light Curver for ID = {idf}')
    ax.set_ylabel('Magnitude', size=12)
    ax.set_ylim(np.nanmax(lightcurve['mag'] + 0.2), np.nanmin(lightcurve['mag'] - 0.2))
    ax.set_xlabel('Date (MJD)', size=12)
    '''
    Gives a zoomed in view of the flare
    '''
    ax = fig.add_subplot(grid[0:1, 2:3])
    ax.errorbar(lightcurve['mjd'], lightcurve['mag'], lightcurve['magerr'],
                marker='o', markersize=3, c="black", alpha=0.15)
    ax.plot(lightcurve['mjd'].values[indx], lightcurve['mag'].values[indx],
            linestyle="none", marker='o', markersize=4, c="red")
    ax.set_ylabel('Magnitude', size=12)
    ax.set_ylim(np.nanmax(lightcurve['mag'] + 0.2), np.nanmin(lightcurve['mag'] - 0.2))
    dur = (np.nanmax(lightcurve['mjd'].values[indx]) -
           np.nanmin(lightcurve['mjd'].values[indx]))
    ax.set_xlim(np.nanmin(lightcurve['mjd'].values[indx]) - 3*dur,
                np.nanmax(lightcurve['mjd'].values[indx]) + 10*dur)
    fig.savefig(outdur + '/' + file.split('/')[-1][:-8] + '_id_' + str(idf) +
                '_type_' + str(typecurve) + '.png')
    if show:
        plt.show()
    else:
        plt.close()


def EquivDur(time, flux, flux_error):
    '''
    Compute the Equivalent Duration of an event. This is simply the area
    under the flare, in relative flux units.
    NOTE: sums up all the data given, no start/stop times input. Only
        pass the flare!
    Flux must be array in units of zero-centered RELATIVE FLUX
    Time must be array in units of DAYS
    Fluxerror must be array in units RELATIVE FLUX
    Output has units of SECONDS
    
    note: error comes from the Spenser's code
    '''
    dtime = np.diff(time)

    ED = np.trapz(flux, x=(time * 60.0 * 60.0 * 24.0))
    ED_err = np.sqrt(np.sum((dtime*flux_error[:-1])**2))

    return ED, ED_err


def measureED(x, y, yerr, tpeak, fwhm, num_fwhm=10):
    '''
    Measure the equivalent duration of a flare in a smoothed light
    curve. FINDflare typically doesnt identify the entire flare, so
    integrate num_fwhm/2*fwhm away from the peak. As long as the light 
    curve is flat away from the flare, the region around the flare should
    not significantly contribute.
    Parameters
    ----------
    x : numpy array
        time values from the entire light curve
    y : numpy array
        flux values from the entire light curve
    yerr : numpy array
        error in the flux values
    tpeak : float
        Peak time of the flare detection
    fwhm : float
        Full-width half maximum of the flare
    num_fwhm : float, optional
        Size of the integration window in units of fwhm
    Returns
    -------
        ED - Equivalent duration of the flare
        ED_err - The uncertainty in the equivalent duration
    '''

    try:
        # I will only give the 
        #width = fwhm*num_fwhm
        istart = np.argwhere(x > tpeak - fwhm * 1)[0]
        ipeak = np.argwhere(x > tpeak)[0]
        istop = np.argwhere(x > tpeak + fwhm * 5)[0]
    
        dx = np.diff(x)
        x = x[:-1]
        y = y[:-1]
        yerr = yerr[:-1]
        mask = (x > x[istart]) & (x < x[istop])
        ED = np.trapz(y[mask], x[mask])
        ED_err = np.sqrt(np.sum((dx[mask]*yerr[mask])**2))

    except IndexError:
        return -1, -1
    
    return ED, ED_err




def writedata(file, outdur, idf, lightcurve, typecurve, nflare, equivdur):
    '''
    file: which match file
    idf: is the light curve is
    typecurve: either sources or tansients
    equivdur: use EquivDur in findflare to save duration
    '''
    outfilename = outdur + '/' + file.split('/')[-1][:-8] + '.csv'
    if not os.path.exists(outdur):
        os.makedirs(outdur)
    if not os.path.isfile(outfilename):
        with open(outfilename, 'a') as csv_file:
            write = csv.writer(csv_file)
            write.writerow(['match_file', 'id', 'type', 'mintime', 'maxtime', 'ra',
                           'dec', 'equivdur'])
    
    with open(outfilename, 'a') as csv_file:
        write = csv.writer(csv_file)
        write.writerow([file.split('/')[-1], idf, typecurve,
                        np.nanmin(lightcurve['mjd'].values[nflare]), np.nanmax(lightcurve['mjd'].values[nflare]), 
                        lightcurve['ra'].values[0],
                        lightcurve['dec'].values[0],
                        equivdur])
# start and stop time and index, amplitude peak, range (peak - average) (flux or magnitude),
# average before, average after, equivelant duration,match file, lightcurve image file name


def findflare(file, outdur, idf, lightcurve, typecurve, N1=3, N2=1, N3=3):
    '''
    Need file, typecurve for writedata
    typecurve needs to be a sting
    '''
    dt = np.diff(lightcurve['mjd'])
    bpoint = np.where((dt > 0.1))[0]
    bpoint += 1
    edges = np.concatenate([[0], bpoint, [len(lightcurve)]])
    indx = np.array([], dtype=np.int)
    for j in range(len(edges) - 1):
        flare = ff(lightcurve['psfflux'][edges[j]:edges[j+1]],
                   lightcurve['psffluxerr'].values[edges[j]:edges[j+1]],
                   N1=N1, N2=N2, N3=N3)
        for i in range(len(flare[0, :])):
            nflare = np.arange(edges[j] + flare[0, i],
                               edges[j] + flare[1, i]+1, 1,
                               dtype=np.int)
            if len(nflare) > 0:
                dur = (np.nanmax(lightcurve['mjd'].values[nflare]) - np.nanmin(lightcurve['mjd'].values[nflare]))
                isflare = np.where((lightcurve['mjd'] >= np.nanmin(lightcurve['mjd'].values[nflare]) - 3*dur) &
                                   (lightcurve['mjd'] <= np.nanmax(lightcurve['mjd'].values[nflare]) + 10*dur))[0]
                p = EquivDur(lightcurve['mjd'].values[isflare], lightcurve['psfflux'].values[isflare] /
                             np.nanmedian(lightcurve['psfflux']) - 1)
#                write must come first because it creates the directory
                writedata(file, outdur, idf, lightcurve, typecurve, nflare, p)
                #plotflare(file, outdur, idf, lightcurve, typecurve, nflare, show=False)
                indx = np.append(indx, nflare)            
            
def rundata(matchfile, outdur):
    sources, sourcedata = loaddata(matchfile)
    ids = sources.loc[(sources["bestmedianmag"] <= 21.5) &
                      (sources["bestmedianmag"] > 0) &
                      (sources["nobs"] > 100), "matchid"]  
    #idt = transients.loc[(transients["bestmedianmag"] <= 21.5) & 
    #                     (transients["bestmedianmag"] > 0) &
    #                     (transients["nobs"] > 100), "matchid"]
    for i in ids:
        s = sourcedata[sourcedata["matchid"] == i]
        findflare(matchfile, outdur, i, s, "sources")
#    s = sourcedata[sourcedata["matchid"] == 21400]
#    findflare(matchfile, outdur, 21400, s, 'sources')
    #for i in idt:
    #    t = transientdata[transientdata["matchid"] == i]
    #    findflare(matchfile, outdur, i, t, "transients")


    
def runbatch(mfile='592.txt', nrun=0, outdur=''):
    batch = open(mfile, 'r')
    files = batch.readlines()
    if nrun == 0:
        nrun = len(files)
        
    print(mfile+': running '+str(nrun)+' matchfiles')
    for i in range(nrun):
        outfilename = outdur + '/' + files[i][1:-1].split('/')[-1][:-8] + '.csv'
        if not os.path.isfile(outfilename):
            rundata('/epyc/data/ztf_matchfiles' + files[i][1:-1], outdur)
    
#create a csv for each match file
#4:05 - 4:24

#if __name__ == "__main__":
#    fl = '../ztf_000592_zr_c01_q1_match_programid2.pytable'
#    out = 'flares_found_full'
#    rundata(fl, out)
#    print('finished')

     