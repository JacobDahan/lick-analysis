from read_ardulines import read
import matplotlib.pyplot as plt
from itertools import chain
import read_ardulines
import pandas as pd
import numpy as np
import matplotlib
import os

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
    ax.legend(loc='upper right', frameon=False) 

def listclamp(minn, maxn, nlist): 
    """Returns sorted list of values between minn and maxn"""
    return sorted([x for x in nlist if (minn<=x and x<=maxn)])

def flatten_dict_values(dictionary):
    """Returns flattened list of values from dict of lists"""
    return list(chain.from_iterable(dictionary.values()))

def sortTM(tm,direction,*args):
    """Sorts TM by *args"""
    cols      = [a for a in args]
    tm_sorted = tm.sort_values(by=cols, ascending=direction)
    return tm_sorted

def find_ndarr_row(ndarrl, ndarrl_row, ndarrr, ndarrr_row):
    """Finds index of row in unsorted nd array"""
    for i,(rowl,rowr) in enumerate(zip(ndarrl,ndarrr)):
        if np.all(rowl == ndarrl_row) and np.all(rowr == ndarrr_row):
            return i

def TrialMatrix(fi):
    """Create sorted trial matrix from raw ardulines data"""
    st        = [s for s in fi.split('/') if 'saved' in s][0]
    date      = '-'.join([s for s in st.split('-')][0:3])
    mouse     = ''.join([s for s in st.split('-')][6])
    tm        = read(fi,date,mouse)
    sloc      = os.getcwd()
    fname     = '{}_{}_raster_data.csv'.format(mouse,date)
    savefi    = os.path.join(sloc, fname)
    direction = [True, True, True]
    tm        = sortTM(tm,direction,'stim','outcome','trial')
    tm.to_csv(savefi, index=False)
    raster(tm,date,mouse,sloc)
    histo(tm,date,mouse,sloc,only_corr=True)
    histo(tm,date,mouse,sloc,only_corr=False)
    return tm

def raster(tm,date,mouse,sloc,limr=-2000,liml=5000):
    """Plot trial matrix."""
    # transform trial matrix into dictionaries with
    # (trial,[times]) k,v pairs
    dl       = tm.groupby('trial')['left'].apply(list).to_dict()
    dr       = tm.groupby('trial')['right'].apply(list).to_dict()
    # form dataframe from dict
    pdl      = pd.DataFrame.from_dict(dl, orient='index')
    pdr      = pd.DataFrame.from_dict(dr, orient='index')  
    # turn dataframe into list of numpy arrays and remove nans
    ndarrl   = pdl.to_numpy()
    ndarrl   = [row[~np.isnan(row)] for row in ndarrl] 
    ndarrr   = pdr.to_numpy()
    ndarrr   = [row[~np.isnan(row)] for row in ndarrr] 
    # sort list of arrays to match original dataframe 
    # this will reverse default Python sorting and restore sorting by stim, outcome, etc.
    loffs                      = tm.trial.unique()
    inds_unsort                = np.argsort(loffs)
    ndarrl_unsort              = np.zeros_like(ndarrl)
    ndarrl_unsort[inds_unsort] = ndarrl
    ndarrr_unsort              = np.zeros_like(ndarrr)
    ndarrr_unsort[inds_unsort] = ndarrr
    # gather stim change and outcome change indices and unsort them
    stim_chg        = tm.loc[tm.ne(tm.shift()).apply(lambda x: x.index[x].tolist())['stim'][-1]].trial
    stim_chg_unsort = np.where(loffs == stim_chg)[0][0]
    outc_chg        = tm.loc[tm.ne(tm.shift()).apply(lambda x: x.index[x].tolist())['outcome']]
    outc_chg_unsort = [np.where(loffs == oc.trial)[0][0] for i, oc in outc_chg.iterrows()]
    # plot
    fig, ax = plt.subplots()
    ax.eventplot(ndarrl_unsort,colors='blue',label='Left')  
    ax.eventplot(ndarrr_unsort,colors='red',label='Right') 
    legend_without_duplicate_labels(ax)
    ax.autoscale()
    ax.set_ylim(bottom=ax.get_ylim()[0]+7.5,top=ax.get_ylim()[1]-7.5)
    #ax.set_xlim(left=ax.get_xlim()[0],right=ax.get_xlim()[1])
    ax.set_xlim(left=liml,right=limr)
    ax.get_yaxis().set_ticks([])
    for i, oc in enumerate(outc_chg_unsort):
        if i % 2:
            plt.hlines(oc,xmin=ax.get_xlim()[0],xmax=ax.get_xlim()[1],linestyles='dashed')
    plt.xlabel('Time [ms]')
    plt.ylabel('Trial')
    plt.axhspan(ax.get_ylim()[0], stim_chg_unsort, facecolor='yellow', alpha=0.3)
    plt.axhspan(stim_chg_unsort, ax.get_ylim()[1], facecolor='chartreuse', alpha=0.3)
    plt.title("{} ({})".format(mouse,date))
    plt.savefig(os.path.join(sloc,"{}_{}_raster.svg".format(mouse,date)))
    #plt.show()
    plt.close('all')
    return None

def merge_dols(dol1,dol2):
    """Merge two dictionaries of lists (DOLs)."""
    if len(dol2) > 0:
        keys = set(dol1).union(dol2)
        no   = []
        return dict((k, sorted(list(pd.Series(dol1.get(k, no)).dropna()) + list(pd.Series(dol2.get(k, no)).dropna()))) for k in keys)
    else:
        keys = set(dol1)
        no   = []
        return dict((k, sorted(list(pd.Series(dol1.get(k, no)).dropna().dropna()))) for k in keys)

def histo(tm,date,mouse,sloc,only_corr=False):
    # correct rough trials only
    corrr = tm.groupby(['stim','outcome']).get_group(('ROUGH',1.0))
    # dictionary of correct rough trials and lick times
    # with (trial, [times]) k,v pairs for right and left licks
    corrr_r = corrr.groupby('trial')['right'].apply(list).to_dict()
    corrr_l = corrr.groupby('trial')['left'].apply(list).to_dict()
    # merge dictionaries if all licks for correct trials
    # else only save correct licks on correct trials
    corrr_a = merge_dols(corrr_r,corrr_l) if not only_corr else merge_dols(corrr_l,{})
    # correct smooth trials only
    corrs = tm.groupby(['stim','outcome']).get_group(('SMOOTH',1.0))
    # dictionary of correct rough trials and lick times
    # with (trial, [times]) k,v pairs for right and left licks
    corrs_r = corrs.groupby('trial')['right'].apply(list).to_dict()
    corrs_l = corrs.groupby('trial')['left'].apply(list).to_dict()
    # merge dictionaries if all licks for correct trials
    # else only save correct licks on correct trials
    corrs_a = merge_dols(corrs_r,corrs_l) if not only_corr else merge_dols(corrs_r,{})
    # flatten dictionaries for plotting
    corrr_flat = flatten_dict_values(corrr_a)
    corrs_flat = flatten_dict_values(corrs_a)
    # slice to reasonable boundaries
    minn, maxn = -2000, 5000
    corrr_flat = listclamp(minn,maxn,corrr_flat)
    corrs_flat = listclamp(minn,maxn,corrs_flat)
    # assign bins and plot
    bins    = np.linspace(minn,maxn,140)
    fig, ax = plt.subplots()
    oc      = 'OC' if only_corr else 'NOC'
    ax.hist(corrr_flat, bins, color='blue', alpha=0.7, label='ROUGH', weights=np.zeros_like(np.array(corrr_flat)) + 1. / np.array(corrr_flat).size) 
    ax.hist(corrs_flat, bins, color='red', alpha=0.7, label='SMOOTH', weights=np.zeros_like(np.array(corrs_flat)) + 1. / np.array(corrs_flat).size)
    ax.legend(loc='upper right', frameon=False) 
    ax.get_yaxis().set_ticks(np.linspace(0,0.05,5))
    plt.xlabel('Time [ms]')
    plt.ylabel('Relative Frequency')
    plt.title("{} ({})".format(mouse,date))
    plt.tight_layout()
    plt.savefig(os.path.join(sloc,"{}_{}_{}_histo.svg".format(mouse,date,oc),bbox_inches='tight'))
    #plt.show()
    plt.close('all')

def histo2(tm,date,mouse,sloc,only_corr=False):
    # correct rough trials only
    corrr = tm.groupby(['stim','outcome']).get_group(('ROUGH',1.0))
    # dictionary of correct rough trials and lick times
    # with (trial, [times]) k,v pairs for right and left licks
    corrr_r = corrr.groupby('trial')['right'].apply(list).to_dict()
    corrr_l = corrr.groupby('trial')['left'].apply(list).to_dict()
    # merge dictionaries if all licks for correct trials
    # else only save correct licks on correct trials
    corrr_a = merge_dols(corrr_r,corrr_l) if not only_corr else merge_dols(corrr_l,{})
    # correct smooth trials only
    corrs = tm.groupby(['stim','outcome']).get_group(('SMOOTH',1.0))
    # dictionary of correct rough trials and lick times
    # with (trial, [times]) k,v pairs for right and left licks
    corrs_r = corrs.groupby('trial')['right'].apply(list).to_dict()
    corrs_l = corrs.groupby('trial')['left'].apply(list).to_dict()
    # merge dictionaries if all licks for correct trials
    # else only save correct licks on correct trials
    corrs_a = merge_dols(corrs_r,corrs_l) if not only_corr else merge_dols(corrs_r,{})
    # slice to reasonable boundaries
    minn, maxn = -1000, 1000
    corrr_slice = {k: listclamp(minn,maxn,v) for k, v in corrr_a.items() if len(v) > 0}
    corrs_slice = {k: listclamp(minn,maxn,v) for k, v in corrs_a.items() if len(v) > 0}
    # extract first lick
    first_lir  = [v[0] for k,v in corrr_slice.items() if len(v) > 0]
    first_lis  = [v[0] for k,v in corrs_slice.items() if len(v) > 0]
    # assign bins and plot
    bins    = np.linspace(minn,maxn,50)
    fig, ax = plt.subplots()
    oc      = 'OC' if only_corr else 'NOC'
    ax2     = ax.twinx() 
    ax.hist(first_lir, bins, color='blue', alpha=0.7, label='ROUGH', weights=np.zeros_like(np.array(first_lir)) + 1. / np.array(first_lir).size) 
    ax.hist(first_lis, bins, color='red', alpha=0.7, label='SMOOTH', weights=np.zeros_like(np.array(first_lis)) + 1. / np.array(first_lis).size)
    ax2.hist(first_lir, bins, color='blue', alpha=0.7, label='ROUGH', weights=np.zeros_like(np.array(first_lir)) + 1. / np.array(first_lir).size, cumulative=True, histtype='step') 
    ax2.hist(first_lis, bins, color='red', alpha=0.7, label='SMOOTH', weights=np.zeros_like(np.array(first_lis)) + 1. / np.array(first_lis).size, cumulative=True, histtype='step')
    ax.set_yticks(np.linspace(0,0.25,5))
    ax.set_ylabel('Density')
    ax.set_zorder(0)
    ax.set_facecolor('none')
    ax2.set_ylabel('Cumulative Density')
    ax2.set_yticks(np.linspace(0,1,5))
    ax2.set_zorder(-1)
    ax2.axvline(x = maxn, color = 'white', linewidth = 2)
    ax.legend(loc='upper right', fancybox=False, edgecolor='k', framealpha=1) 
    ax.set_ylim(0,0.2625)
    ax2.set_ylim(0,1.05)
    plt.xlabel('Time [ms]')
    plt.title("{} ({})".format(mouse,date))
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(sloc,"{}_{}_{}_first_lick_histo.svg".format(mouse,date,oc),bbox_inches='tight'))
    #plt.show()
    plt.close('all')

def line(tm,date,mouse,only_corr=False):
    # correct rough trials only
    corrr = tm.groupby(['stim','outcome']).get_group(('ROUGH',1.0))
    # dictionary of correct rough trials and lick times
    # with (trial, [times]) k,v pairs for right and left licks
    corrr_r = corrr.groupby('trial')['right'].apply(list).to_dict()
    corrr_l = corrr.groupby('trial')['left'].apply(list).to_dict()
    # merge dictionaries if all licks for correct trials
    # else only save correct licks on correct trials
    corrr_a = merge_dols(corrr_r,corrr_l) if not only_corr else merge_dols(corrr_l,{})
    # correct smooth trials only
    corrs = tm.groupby(['stim','outcome']).get_group(('SMOOTH',1.0))
    # dictionary of correct rough trials and lick times
    # with (trial, [times]) k,v pairs for right and left licks
    corrs_r = corrs.groupby('trial')['right'].apply(list).to_dict()
    corrs_l = corrs.groupby('trial')['left'].apply(list).to_dict()
    # merge dictionaries if all licks for correct trials
    # else only save correct licks on correct trials
    corrs_a = merge_dols(corrs_r,corrs_l) if not only_corr else merge_dols(corrs_r,{})
    # slice to reasonable boundaries
    minn, maxn  = -2000, 5000
    corrr_slice = {k: pd.Series(listclamp(minn,maxn,v)) for k, v in corrr_a.items() if len(v) > 0}
    corrs_slice = {k: pd.Series(listclamp(minn,maxn,v)) for k, v in corrs_a.items() if len(v) > 0}
    # flatten dictionaries
    corrr_slice_flat = pd.Series(flatten_dict_values(corrr_slice))
    corrs_slice_flat = pd.Series(flatten_dict_values(corrs_slice))        
    # find mean time of first lick
    corrr_mean = np.mean(corrr_slice)
    corrs_mean = np.mean(corrs_slice)
    