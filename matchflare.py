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
    transientdata = pd.read_hdf(matchfile, key="matches/transientdata")
    sourcesObject = mtfl.get_node("/matches", "sources")
    transientsObject = mtfl.get_node("/matches", "transients")
    sources_columns = pd.DataFrame.from_records(sourcesObject.read(0,0))
    sources_columns_list = sources_columns.columns.tolist()
    sources = pd.DataFrame.from_records(sourcesObject.cols, columns=sources_columns_list)
    transients_columns = pd.DataFrame.from_records(transientsObject.read(0,0))
    transients_columns_list = transients_columns.columns.tolist()
    transients = pd.DataFrame.from_records(transientsObject.cols, columns=transients_columns_list)
    sourcedata.sort_values('mjd', inplace=True)
    transientdata.sort_values('mjd', inplace=True)
    return sources, sourcedata, transients, transientdata


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
    ax.set_xlabel('Date (MJD)', size=12)
    ax.invert_yaxis()
    '''
    Gives a zoomed in view of the flare
    '''
    ax = fig.add_subplot(grid[0:1, 2:3])
    ax.errorbar(lightcurve['mjd'], lightcurve['mag'], lightcurve['magerr'],
                marker='o', markersize=3, c="black", alpha=0.15)
    ax.plot(lightcurve['mjd'].values[indx], lightcurve['mag'].values[indx],
            linestyle="none", marker='o', markersize=4, c="red")
    ax.set_ylabel('Magnitude', size=12)
#    ax.set_xlabel('Date (MJD)', size=12)
    ax.invert_yaxis()
    dur = (np.nanmax(lightcurve['mjd'].values[indx]) -
           np.nanmin(lightcurve['mjd'].values[indx]))
    ax.set_xlim(np.nanmin(lightcurve['mjd'].values[indx]) - 3*dur,
                np.nanmax(lightcurve['mjd'].values[indx]) + 10*dur)
    fig.savefig(outdur + '/' + file.split('/')[-1][:-8] + '_id_' + str(idf) +
                '_type_' + str(typecurve) + '.png')


def EquivDur(time, flux):
    '''
    Compute the Equivalent Duration of an event. This is simply the area
    under the flare, in relative flux units.
    NOTE: sums up all the data given, no start/stop times input. Only
        pass the flare!
    Flux must be array in units of zero-centered RELATIVE FLUX
    Time must be array in units of DAYS
    Output has units of SECONDS
    '''
    p = np.trapz(flux, x=(time * 60.0 * 60.0 * 24.0))
    return p


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
            write.writerow(['match_file', 'id', 'type', 'date_of_flare_points',
                           'dec', 'equivdur'])
    
    with open(outfilename, 'a') as csv_file:
        write = csv.writer(csv_file)
        write.writerow([file.split('/')[-1], idf, typecurve,
                        lightcurve['mjd'].values[nflare],
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
        flare = ff(lightcurve['mag'][edges[j]:edges[j+1]],
                   lightcurve['magerr'].values[edges[j]:edges[j+1]],
                   N1=N1, N2=N2, N3=N3)
        for i in range(len(flare[0, :])):
            nflare = np.arange(edges[j] + flare[0, i],
                               edges[j] + flare[1, i]+1, 1,
                               dtype=np.int)
            if len(nflare) > 0:
                p = EquivDur(lightcurve['mjd'], lightcurve['psfflux'])
#                write must come first because it creates the directory
                writedata(file, outdur, idf, lightcurve, typecurve, nflare, p)
                plotflare(file, outdur, idf, lightcurve, typecurve, nflare)
            indx = np.append(indx, nflare)


def rundata(matchfile, outdur):
    sources, sourcedata, transients, transientdata = loaddata(matchfile)
    ids = sources.loc[(sources["bestmedianmag"] <= 21.5) &
                      (sources["bestmedianmag"] > 0) &
                      (sources["nobs"] > 100), "matchid"]  
    idt = transients.loc[(transients["bestmedianmag"] <= 21.5) & 
                         (transients["bestmedianmag"] > 0) &
                         (transients["nobs"] > 100), "matchid"]
    for i in ids:
        s = sourcedata[sourcedata["matchid"] == i]
        findflare(matchfile, outdur, i, s, "sources")
#    s = sourcedata[sourcedata["matchid"] == 21400]
#    findflare(matchfile, outdur, 21400, s, 'sources')
    for i in idt:
        t = transientdata[transientdata["matchid"] == i]
        findflare(matchfile, outdur, i, t, "transients")



def runbatch(mfile='592.txt', nrun=0, outdur=''):
    files = csv.reader(mfile)
    if nrun == 0:
        nrun = len(files)
    for i in range(nrun):
        rundata(files[i], outdur)

#create a csv for each match file
#4:05 - 4:24

if __name__ == "__main__":
    fl = '../ztf_000592_zr_c01_q1_match_programid2.pytable'
    out = 'flares_found_full'
    rundata(fl, out)
    print('finished')

     