#!/usr/bin/env python3

import sys
import numpy as np
import csv
import os
import glob
import tables as pt
from IPython.display import Image, display
import pandas as pd
import matplotlib.pyplot as plt
import matchflare
from FINDflare import FINDflare as ff
import morestuff_flux as ms
from scipy.optimize import curve_fit
from scipy.stats import chisquare as chi


def loaddata(matchfile):
    mtfl = pt.open_file(matchfile, 'r', root_uep='/', filters='lzo')
    sourcedata = pd.read_hdf(matchfile, key="matches/sourcedata")
    sourcesObject = mtfl.get_node("/matches", "sources")
    sources_columns = pd.DataFrame.from_records(sourcesObject.read(0, 0))
    sources_columns_list = sources_columns.columns.tolist()
    sources = pd.DataFrame.from_records(sourcesObject.cols, columns=sources_columns_list)
    sourcedata.sort_values('mjd', inplace=True)
    mtfl.close()
    return sources, sourcedata

def findtransient(lightcurve, N1=3, N2=1, N3=3):
    '''
    Uses the Findflare function to identify spikes in brightness in ZTF lightcurves
    Parameters
    ----------
    lightcurve : pandas dataframe
        A single row from either the sourcedata or transietdata dataframes
        from a ZTF match file.
    N1 : int, optional
        Coefficient from original paper (Default is 3)
        How many times above the stddev is required.
    N2 : int, optional
        Coefficient from original paper (Default is 1)
        How many times above the stddev and uncertainty is required
    N3 : int, optional
        Coefficient from original paper (Default is 3)
        The number of consecutive points required to flag as a flare
    Returns
    -------
    The indices of the times when the magnitude decrease/the source brightness increases
    '''
    dt = np.diff(lightcurve['mjd'])
    #index for boundary points of gaps in the lightcurve
    bpoint = np.where((dt > 0.1))[0] 
    #shifts the boundary point from the beginning of gap to the end of gap
    bpoint += 1
    ''' check in about changing it to -1 '''
    edges = np.concatenate([[0], bpoint, [len(lightcurve)-1]])
    #create a dataframe to put in each set of indx and equivdur
    df = pd.DataFrame(columns=['fl_indx','equivdur'])
    for j in range(len(edges) - 1):
        #searches for a transient in each section of the lightcurve
        trans = ff(lightcurve['psfflux'][edges[j]:edges[j+1]],
                   lightcurve['psffluxerr'].values[edges[j]:edges[j+1]],
                   N1=N1, N2=N2, N3=N3)
        #if no flare is found len = 0, otherwise is loops over nflares
        for i in range(len(trans[0, :])):
            #trans gives the index in the section, this converts to index in the total lightcurve 
            ntrans_indx = np.arange(edges[j] + trans[0, i],
                                    edges[j] + trans[1, i]+1, 1,
                                    dtype=np.int)
            #if a transient is identified
            if len(ntrans_indx) > 0:
                dur = (np.nanmax(lightcurve['mjd'].values[ntrans_indx]) - np.nanmin(lightcurve['mjd'].values[ntrans_indx]))
                isflare = np.where((lightcurve['mjd'] >= np.nanmin(lightcurve['mjd'].values[ntrans_indx]) - 3*dur) &
                                   (lightcurve['mjd'] <= np.nanmax(lightcurve['mjd'].values[ntrans_indx]) + 15*dur))[0]
                equiv = matchflare.EquivDur(lightcurve['mjd'].values[isflare],
                                            lightcurve['psfflux'].values[isflare] / np.nanmedian(lightcurve['psfflux']) - 1)
                df = df.append({'fl_indx':ntrans_indx, 'equivdur':equiv}, ignore_index=True)
                
    return df

def parameters_guess(lightcurve, fl_indx):
    sigma = np.nanmax(lightcurve['mjd'].values[fl_indx]) - np.nanmin(lightcurve['mjd'].values[fl_indx])
    isflare = np.where((lightcurve['mjd'] >= np.nanmin(lightcurve['mjd'].values[fl_indx]) - 5*sigma) &
                       (lightcurve['mjd'] <= np.nanmax(lightcurve['mjd'].values[fl_indx]) + 10*sigma))[0]
    offset = np.nanmedian(lightcurve['psfflux'].values[isflare])
    amp =  np.nanmin(lightcurve['psfflux'].values[fl_indx]) - offset
    timepeak = lightcurve['mjd'].values[fl_indx[0]]
    return offset, amp, sigma, timepeak, isflare


def fit_gauss(lightcurve, fl_indx):
    offset, amp, sigma, timepeak, isflare = parameters_guess(lightcurve, fl_indx)
    err = lightcurve['psffluxerr'].values[isflare]
    para_gauss, cov_gauss = curve_fit(ms.gaus, lightcurve['mjd'].values[isflare],lightcurve['psfflux'].values[isflare], 
                                      p0=[amp, offset, timepeak, sigma], sigma=err)
    perr_gauss = np.sqrt(np.diag(cov_gauss))
    x_gauss = lightcurve['mjd'].values[isflare]
    y_gauss = ms.gaus(x_gauss,para_gauss[0],para_gauss[1],para_gauss[2],para_gauss[3])
    return x_gauss, y_gauss,  para_gauss, perr_gauss


def fit_flaremodel(lightcurve, fl_indx):
    offset, amp, sigma, timepeak, isflare = parameters_guess(lightcurve, fl_indx)
    err = lightcurve['psffluxerr'].values[isflare]
    para_flare, cov_flare = curve_fit(ms.aflare1, lightcurve['mjd'].values[isflare],lightcurve['psfflux'].values[isflare], 
                                      p0=[amp, offset, timepeak, sigma], sigma=err)
    perr_flare = np.sqrt(np.diag(cov_flare))
    x_flare = lightcurve['mjd'].values[isflare]
    y_flare = ms.aflare1(x_flare,para_flare[0],para_flare[1],para_flare[2],para_flare[3])
    return x_flare, y_flare, para_flare, perr_flare


def chisquared(flux, fluxerr, fitflux, para):
    '''
    Input the mag and magerr column from either a 
    mag: 
    '''
    chisquared = np.sum(((flux - fitflux) / fluxerr) ** 2)
    dof = len(flux) - len(para)
    reduced = chisquared / dof
    return chisquared, dof, reduced


def AIC(flux, fluxerr, fitflux, para):
    k = len(para)
    n = len(flux)
    loglike = -1 * np.sum((flux - fitflux) ** 2 / (2 * fluxerr ** 2))
    AIC = (2 * k) - (2 * loglike)
    return AIC


def test_model(lightcurve, fl_indx):
    offset, amp, sigma, timepeak, isflare = parameters_guess(lightcurve, fl_indx)
    y = lightcurve['psfflux'].values[isflare]
    err = lightcurve['psffluxerr'].values[isflare]
    x_f,y_f,para_f,err_f = fit_flaremodel(lightcurve,fl_indx)
    x_g,y_g,para_g,err_g = fit_gauss(lightcurve,fl_indx)
    AIC_f = AIC(y, err, y_f, para_f)
    AIC_g = AIC(y, err, y_g, para_g)        
    chisq_f, dof_f, red_f = chisquared(y, err, y_f, para_f)
    chisq_g, dof_g, red_g = chisquared(y, err, y_g, para_g)    
    compare_exp =  np.exp((AIC_g - AIC_f) / 2)
    compare_per = (AIC_g - AIC_f) #/ np.absolute(AIC_f) * 100
    return AIC_f, chisq_f, dof_f, red_f, AIC_g, chisq_g, dof_g, red_g, compare_exp, compare_per


def checkendflare(lightcurve,indx):
    last_flare = lightcurve['mjd'].values[indx[-1]]
    last_date = lightcurve['mjd'].values[-1]
    return last_date - last_flare


def plot_models(file, lightcurve, fl_indx, ids, initout_csv, residuals=False, stats=None, time=None, full_lc=False, save=False):
    '''
    file: name of the 
    time: time between end of flare and end of light curve
    '''
    offset, amp, sigma, timepeak, isflare = parameters_guess(lightcurve, fl_indx)
    y = lightcurve['psfflux'].values[isflare]
    err = lightcurve['psffluxerr'].values[isflare]
    
    fig = plt.figure(figsize=(15, 7))
    grid = plt.GridSpec(4, 4, hspace=1, wspace=.5)
    ax = fig.add_subplot(grid[0:3, 0:2])
    x_f,y_f,para_f,err_f = fit_flaremodel(lightcurve,fl_indx)
    x_g,y_g,para_g,err_g = fit_gauss(lightcurve,fl_indx)
    ax.errorbar(lightcurve['mjd'].values[isflare],y,err, color='blue',linestyle='none', marker='.', label='Data')
    ax.errorbar(lightcurve['mjd'].values[fl_indx],lightcurve['psfflux'].values[fl_indx],lightcurve['psffluxerr'].values[fl_indx], 
                color='red',linestyle='none', marker='.', label='Data')

    ax.plot(x_f,y_f,color='red', linestyle='-.',label='Flare')
    ax.plot(x_g,y_g,color='green', linestyle='-.',label='Gauss')
    ax.set_title(f'Light Curve for ID = {ids}')
    ax.set_ylabel('Flux')
    ax.legend() 
    
    if residuals:
        ax_res = fig.add_subplot(grid[3:4, 0:2])
        ax_res.errorbar(lightcurve['mjd'].values[isflare], y - y_f, err, color='red',linestyle='none', marker='.',label='Flare', alpha=.5)
        ax_res.errorbar(lightcurve['mjd'].values[isflare], y - y_g, err, color='green',linestyle='none', marker='.',label='Gauss', alpha=.5)
        ax_res.hlines(0, np.min(lightcurve['mjd'].values[isflare]), np.max(lightcurve['mjd'].values[isflare]))
        maxresid = np.max([np.abs(y - y_f), np.abs(y - y_g)])
        ax_res.set_ylim(maxresid, -1 * maxresid)
    if stats:
        fig.text(0.5, 0.89, '{:} = file'.format(file.split('/')[-1][0:37]), ha='left', fontsize=12)        
        fig.text(0.5, 0.85, '{:.5} = AIC Flare Model'.format(stats[0]), ha='left', fontsize=12)
        fig.text(0.5, 0.81, '{:.5} = Chi-Squared Flare'.format(stats[1]), ha='left', fontsize=12)
        fig.text(0.5, 0.77, '{:} = DoF Flare'.format(stats[2]), ha='left', fontsize=12)
        fig.text(0.5, 0.73, '{:.5} = Reduced Flare'.format(stats[3]), ha='left', fontsize=12)
        fig.text(0.5, 0.69, '{:.5} = AIC Gaussian Model'.format(stats[4]), ha='left', fontsize=12)
        fig.text(0.5, 0.65, '{:.5} = Chi-Squared Gauss'.format(stats[5]), ha='left', fontsize=12)
        fig.text(0.5, 0.61, '{:} = DoF Gauss'.format(stats[6]), ha='left', fontsize=12)
        fig.text(0.5, 0.57, '{:.5} = Reduced Gauss'.format(stats[7]), ha='left', fontsize=12)
        fig.text(0.5, 0.53, '{:.5} = EXP comparision of AIC, >1 favors Flare & <1 favors Gauss'.format(stats[8]), ha='left', fontsize=12)
        fig.text(0.5, 0.49, '{:.5} = % comparision of AIC, >0 favors Flare & <0 favors Gauss'.format(stats[9]), ha='left', fontsize=12)
    if time:
        fig.text(0.5, 0.45, '{:.5} = Difference between end of light curve and flare in MJD'.format(time), ha='left', fontsize=12)
    if full_lc:
        ax_full = fig.add_subplot(grid[2:4, 2:4])
        twomjd = lightcurve.loc[lightcurve['mjd'] < lightcurve['mjd'].values[0] + 5]
        ax_full.plot(lightcurve['mjd'],lightcurve['mag'], color='blue', marker='.', label='Data', markersize=5, alpha=.5)
        ax_full.plot(twomjd['mjd'].values[isflare],twomjd['mag'].values[isflare], color='red', marker='.', label='Data', markersize=5, alpha=.5)     
        ax_full.set_ylabel('Mag')
        ax_full.invert_yaxis()
    plt.show()
    if save:
        fig.savefig(os.path.dirname(initout_csv) + '/' + file.split('/')[-1][:-9] +
                    '_id_' + str(ids) + '_indx_' + str(fl_indx[0]) + '_stats.png')
    
    
def writedata(file, initout_csv, ids, lightcurve, stats, trans_df, time_until_end, cat_len):
    '''
    file: which match file
    idf: is the light curve is
    typecurve: either sources or tansients
    equivdur: use EquivDur in findflare to save duration
    time: difference in mjd between flare and end of lightcurve
    '''
    indx = trans_df['fl_indx']
    outfilename = os.path.dirname(initout_csv) + '/stats2_' + file.split('/')[-1][:10] + '.csv'
    if not os.path.isfile(outfilename):
        with open(outfilename, 'a') as csv_file:
            write = csv.writer(csv_file)
            write.writerow(['match_file', 'id', 'aic_flare', 'chi_flare', 'dof_flare', 'red_flare',
                            'aic_gauss', 'chi_gauss', 'dof_gauss', 'red_gauss', 'exp_comp', 'perc_comp',
                            'fl_indx', 'equivdur',
                            'ra', 'dec', 
                            'xpos', 'ypos', 
                            'flare_time', 'end_time',
                            'flare_end_diff', 'n_catflags'])
    
    with open(outfilename, 'a') as csv_file:
        write = csv.writer(csv_file)
        write.writerow([file.split('/')[-1][:-9], ids, stats[0], stats[1], stats[2], stats[3],
                        stats[4], stats[5], stats[6], stats[7], stats[8], stats[9],
                        indx,trans_df['equivdur'],
                        lightcurve['ra'].values[indx[0]], lightcurve['dec'].values[indx[0]], 
                        lightcurve['xpos'].values[indx[0]],lightcurve['ypos'].values[indx[0]],
                        lightcurve['mjd'].values[indx[0]],lightcurve['mjd'].values[-1],
                        time_until_end, cat_len])


def testlightcurve(lightcurve, ids, file, initout_csv):
    cat = lightcurve[lightcurve["catflags"] > 0]
    lightcurve = lightcurve[lightcurve["catflags"] == 0]
    trans_df = findtransient(lightcurve)
    #repeat for each transient detected
    for index, row in trans_df.iterrows():
        
        twomjd = lightcurve.loc[lightcurve['mjd'] < lightcurve['mjd'].values[0] + 5]
        time_until_end = checkendflare(twomjd, row['fl_indx'])
        stats = test_model(twomjd, row['fl_indx'])
        writedata(file, initout_csv, ids, twomjd, stats, row, time_until_end, len(cat))
        plot_models(file, twomjd, row['fl_indx'], ids, initout_csv, residuals=True, 
                    stats=stats, time=time_until_end, full_lc=True, save=True)
        
        return
        


def findmatchpath(path_csv, initout_csv, mypath='/epyc/data/ztf_matchfiles'):
    '''
    Runs testlightcurve function on the objects identified by matchflare.py

    Parameters
    ----------
    path_csv: str
        csv file that contains the paths to all of the matchfiles in a given
        field of view. The csv should contain 64 paths.
    initout_csv: str
        the output csv file from matchflare.py produced from the matchfile are placed
    mypath: str
        path from home directory to directory of the path listed in path_csv.
        If path listed path_csv goes from home directory then input ''.

    Returns
    -------
    The file path to the matchfile that was used to create initout_csv
    '''
    
    path = pd.read_csv(path_csv, header=None) 
    file = initout_csv.split('/')[-1] #takes the file name, not the whole path 
    file = file[0:37] # takes part of the name: ztf_000436_zr_c01_q1_match_programid2
    # looks through the path_csv to find the path that matches name of the initout_csv
    for index, row in path.iterrows():
        name = row[0].split('/')[-1]
        name = name[0:37]
        #when the name of the initout_csv matches the name of the matchfile then return the path
        if file == name:           
            matchfile = mypath + row[0][1:]
            return matchfile

        
def runmatch(matchfile, initout_csv):
    '''
    Runs testlightcurve function on the objects identified by matchflare.py

    Parameters
    ----------
    matchfile: str
        path to a ZTF matchfile
    initout_csv: str
        the output csv file from matchflare.py produced from the matchfile

    Returns
    -------
    testlightcurve for each of the sources in identified in the initout_csv
    '''

    initout = pd.read_csv(initout_csv)
    sources, sourcedata = loaddata(matchfile)
    match_sources = initout.loc[initout['type'] == 'sources']
    for index, row in match_sources.iterrows():
            lightcurve = sourcedata[sourcedata["matchid"] == match_sources['id'][index]]
            try:
                testlightcurve(lightcurve, match_sources['id'][index], matchfile, initout_csv)
            except RuntimeError:
                file = initout_csv.split('/')[-1]
                file = file[0:37] 
                outfilename = os.path.dirname(initout_csv) + '/error2_' + file.split('/')[-1][:10] + '.csv'
                if not os.path.isfile(outfilename):
                    with open(outfilename, 'a') as csv_file:
                        write = csv.writer(csv_file)
                        write.writerow(['match_file', 'id'])
                        write.writerow([file.split('/')[-1][:-9], lightcurve['matchid'].values[0]])

                with open(outfilename, 'a') as csv_file:
                    write = csv.writer(csv_file)
                    write.writerow([file.split('/')[-1][:-9], lightcurve['matchid'].values[0]])

def runfield(path_csv, initout_dir):
    '''
    Runs the runmatch funtion on each of the ZTF matchfiles in a given field of
    view. Must run the matchflare.py script on each matchfile prior to using
    runfield. runfield runs faster than the matchflare.py because it uses the
    csv output from matchflare.py which already contains the lightcurves with
    transients identified.

    Parameters
    ----------
    path_csv: str
        csv file that contains the paths to all of the matchfiles in a given
        field of view. The csv should contain 64 paths.
    initout_dir: str
        directory where the csv files are held from initial run of the
        matchfiles with matchflare.py. This directory should only contain the
        csv files from the matchflare.py output. If nessesary, manually change
        the ending of the file names in glob.glob below to create a unique to
        the output files ending so that it only retrieves the csv files output
        from the matchflare.py.

    Returns
    -------
    Produces a single csv file containing the output from each runmatch
    '''
   
    # identifies all of the csv files in the input directory
    initout_all = glob.glob(initout_dir + '/*programid2.csv')

    for initout_csv in initout_all:
        matchfile = findmatchpath(path_csv, initout_csv)
        print(matchfile)
        runmatch(matchfile, initout_csv)


#if __name__ == "__main__":
#    path_csv = sys.argv[1]
#    directory_to_matchcsv = sys.argv[2]
#    runfield(path_csv, directory_to_matchcsv)
