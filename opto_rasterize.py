from matplotlib.ticker import FormatStrFormatter
from collections import defaultdict
from read_ardulines import read
import matplotlib.pyplot as plt
from itertools import chain
from itertools import cycle
import read_ardulines
import pandas as pd
import numpy as np
import matplotlib
import os

def reorderLegend(ax=None,order=None,unique=False):
    """Returns tuple of handles, labels for axis ax, 
    after reordering them to conform to the label order `order`, 
    and if unique is True, after removing entries with duplicate labels."""
    if ax is None: ax=plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0])) # sort both labels and handles by labels
    if order is not None: # Sort according to a given list (not necessarily complete)
        keys=dict(zip(order,range(len(order))))
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t,keys=keys: keys.get(t[0],np.inf)))
    if unique:  labels, handles= zip(*unique_everseen(zip(labels,handles), key = labels)) # Keep only the first of each handle
    ax.legend(handles, labels, loc='lower right', fancybox=False, edgecolor='k', framealpha=1)
    return(handles, labels)

def unique_everseen(seq, key=None):
    seen = set()
    seen_add = seen.add
    return [x for x,k in zip(seq,key) if not (k in seen or seen_add(k))]

def simpleaxis(ax):
    """Reduces axes to left and right"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def set_box_color(bp, color, alpha):
    """
    Set color for boxplot using arguments color and alpha.
    Color must be converted to RGBA due to Pythonic nonsense.
    """
    bp_color = matplotlib.colors.to_rgba(color,alpha)
    plt.setp(bp['boxes'], facecolor=bp_color)
    plt.setp(bp['fliers'], markeredgecolor='black')
    plt.setp(bp['boxes'], edgecolor='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['caps'], color='black')
    plt.setp(bp['medians'], color='black')

def legend_without_duplicate_labels(ax,pos='lower right',title=None):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    if title is None:
        ax.legend(*zip(*unique), loc=pos, fancybox=False, edgecolor='k', framealpha=1)
    else:
        ax.legend(*zip(*unique), loc=pos, fancybox=False, edgecolor='k', framealpha=1, title=title)

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

def find_ndarr_row(lNdarray, lNdarray_row, rNdarray, rNdarray_row):
    """Finds index of row in unsorted nd array"""
    for i,(rowl,rowr) in enumerate(zip(lNdarray,rNdarray)):
        if np.all(rowl == lNdarray_row) and np.all(rowr == rNdarray_row):
            return i

def generate_decision_column(tm):
    """Generate decision column from stimulus and outcome columns"""
    # initiate column
    decisions = []
    # gather outcomes and actual stimuli
    outcomes = tm.outcome
    stimuli  = tm.stim
    for outc, stim in zip(outcomes, stimuli):
        if outc:
            decisions.extend(['left' if stim == 'ROUGH' else 'right'])
        else:
            decisions.extend(['right' if stim == 'ROUGH' else 'left'])
    return decisions

def bin_by_time_and_binary(tm,time_col,bin_col,liml=-2000,limr=2000,bins=41):
    """Filters unreasonable times. Bins tm into n bins.
    Calculates proportions using binary column bin_col"""
    binsList        = np.linspace(liml,limr,bins)
    clampedList     = [(time,binary) for time,binary in zip(tm[time_col],tm[bin_col]) if (liml<=time and time<=limr)]
    clampedTimeList = [time for time,binary in clampedList]
    clampedBinList  = [binary for time,binary in clampedList]
    tempDf          = pd.DataFrame({'time':clampedTimeList,'bin':clampedBinList})
    binnedSeries    = pd.cut(tempDf.time,binsList,include_lowest=True)
    statsDf         = tempDf.groupby(binnedSeries)['bin'].agg(['count', 'mean', 'sem'])
    # turn single-trial bins into nans for more accurate data analysis
    statsDf.at[statsDf[statsDf['count']==1].index.values,'mean'] = np.nan 
    # calculate statistics
    proportions  = statsDf['mean']
    errors       = statsDf['sem']
    plot_centers = [interval.mid for interval in statsDf.index]
    return proportions, plot_centers, errors

def TrialMatrix(fi,LT=False,FA=False,opto=np.inf):
    """Create sorted trial matrix from raw ardulines data
    LT: lick train session (bool)
    FA: forced alternation session (bool)
    opto: time threshold within which first lick must occur (for opto sessions, use 3; else inf)"""
    st        = [s for s in fi.split('/') if 'saved' in s][0]
    date      = '-'.join([s for s in st.split('-')][0:3])
    mouse     = ''.join([s for s in st.split('-')][6])
    tm        = read(fi,date,mouse,LT,FA,opto)
    return tm

def makeTrialMatrix(fis,LT=False,FA=False,opto=np.inf):
    """Create TrialMatrix for each session and concatenate.
    LT: lick train session (bool)
    FA: forced alternation session (bool)
    opto: time threshold within which first lick must occur (for opto sessions, use 3; else inf)"""
    # set ascending/descending sorting order
    tmList    = []
    max_trial = 0
    for i, fi in enumerate(fis):
        tm_temp       = TrialMatrix(fi,LT,FA,opto)
        # add trials to previous maximum trial number to ensure no trial number overlap
        tm_temp.trial = tm_temp.trial.apply(lambda x: x + max_trial)
        max_trial     = tm_temp.trial.max() + 1
        tmList.append(tm_temp)
    if not LT:
        direction = [False, True, True, True]
        tm        = sortTM(pd.concat(tmList).reset_index(drop=True),direction,'opto','stim','outcome','trial').reset_index(drop=True)
    else:
        direction = [True]
        tm        = sortTM(pd.concat(tmList).reset_index(drop=True),direction,'raw_time').reset_index(drop=True)
    return tm

def CongruentLine(tm,beh,genotype,per_mouse=True,liml=-2000,limr=2000,bins=41):
    """Plots proportion of licks congruent with recorded choice as function of time"""
    # identify recorded choice
    decisions  = generate_decision_column(tm)
    tm         = tm.copy().assign(decision=decisions)
    congruents = np.where(tm.left.notna() & tm.decision.isin(['left']) | tm.right.notna() & tm.decision.isin(['right']),1,0)
    tm         = tm.assign(congruent=congruents)
    titleStr   = ''.join(tm.mouse.unique()) if per_mouse else genotype
    titleStr2  = "Congruent Licks - {} : {} ({})".format(genotype,titleStr,beh) if per_mouse else "Congruent Licks - {} ({})".format(titleStr,beh)
    # group tm by laser status
    onCongruentDf  = tm.groupby('opto').get_group('ON')
    offCongruentDf = tm.groupby('opto').get_group('OFF')
    # merge left and right lick times
    tempOnDf       = pd.DataFrame({'left':onCongruentDf.left, 'right':onCongruentDf.right})   
    tempOffDf      = pd.DataFrame({'left':offCongruentDf.left, 'right':offCongruentDf.right})  
    licksOnList    = tempOnDf.stack().groupby(level=0).sum() 
    licksOffList   = tempOffDf.stack().groupby(level=0).sum() 
    onCongruentDf  = onCongruentDf.assign(licks=licksOnList)
    offCongruentDf = offCongruentDf.assign(licks=licksOffList)  
    # calculate proportions binned by time for laser on and off
    proportionsOn, plotCentersOn, errorOn    = bin_by_time_and_binary(onCongruentDf,time_col='licks',bin_col='congruent',liml=liml,limr=limr,bins=bins)
    proportionsOff, plotCentersOff, errorOff = bin_by_time_and_binary(offCongruentDf,time_col='licks',bin_col='congruent',liml=liml,limr=limr,bins=bins)
    # average centers of bins to correct for numpy rounding errors
    plotCentersList = [np.mean([p1,p2]) for p1,p2 in zip(plotCentersOn,plotCentersOff)]
    # plot
    fig, ax = plt.subplots()
    ax.plot(plotCentersList,proportionsOn,linestyle='-',color='#ffb82b',label='Laser ON')  
    ax.plot(plotCentersList,proportionsOff,linestyle='-',color='black',label='Laser OFF') 
    legend_without_duplicate_labels(ax)
    ax.autoscale()
    ax.set_ylim(bottom=0,top=1.05)
    ax.get_yaxis().set_ticks(np.linspace(0.25,1.0,4))
    ax.set_xlim(left=liml,right=limr)
    simpleaxis(ax)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.fill_between(plotCentersList, proportionsOn-errorOn, proportionsOn+errorOn, alpha=0.3, edgecolor='none', facecolor='#ffb82b')
    plt.fill_between(plotCentersList, proportionsOff-errorOff, proportionsOff+errorOff, alpha=0.6, edgecolor='none', facecolor='silver')
    plt.xlabel('Time [ms]')
    plt.ylabel('Lick Probability')
    plt.title("{}".format(titleStr2))
    plt.savefig(os.path.join(os.getcwd(),"{}_concat_congruent_{}.svg".format(titleStr,beh)))
    plt.close('all')
    return None

def CongruentLineByStim(tm,beh,genotype,per_mouse=True,liml=-2000,limr=2000,bins=41):
    """Plots proportion of licks congruent with recorded choice as function of time, separated by stimulus type"""
    # identify recorded choice
    decisions  = generate_decision_column(tm)
    tm         = tm.copy().assign(decision=decisions)
    congruents = np.where(tm.left.notna() & tm.decision.isin(['left']) | tm.right.notna() & tm.decision.isin(['right']),1,0)
    tm         = tm.assign(congruent=congruents)
    titleStr   = ''.join(tm.mouse.unique()) if per_mouse else genotype
    titleStr2  = "Congruent Licks (by stim) - {} : {} ({})".format(genotype,titleStr,beh) if per_mouse else "Congruent Licks (by stim) - {} ({})".format(titleStr,beh)
    # group tm by laser status
    onCongruentDf  = tm.groupby('opto').get_group('ON')
    offCongruentDf = tm.groupby('opto').get_group('OFF')
    # grup by stimulus
    onRoughDf   = onCongruentDf.groupby('stim').get_group('ROUGH')
    onSmoothDf  = onCongruentDf.groupby('stim').get_group('SMOOTH')
    offRoughDf  = offCongruentDf.groupby('stim').get_group('ROUGH')
    offSmoothDf = offCongruentDf.groupby('stim').get_group('SMOOTH')
    # merge left and right lick times
    tempOnRoughDf      = pd.DataFrame({'left':onRoughDf.left, 'right':onRoughDf.right})  
    tempOnSmoothDf     = pd.DataFrame({'left':onSmoothDf.left, 'right':onSmoothDf.right})  
    tempOffRoughDf     = pd.DataFrame({'left':offRoughDf.left, 'right':offRoughDf.right})  
    tempOffSmoothDf    = pd.DataFrame({'left':offSmoothDf.left, 'right':offSmoothDf.right})
    licksOnRoughList   = tempOnRoughDf.stack().groupby(level=0).sum()
    licksOnSmoothList  = tempOnSmoothDf.stack().groupby(level=0).sum()
    licksOffRoughList  = tempOffRoughDf.stack().groupby(level=0).sum()
    licksOffSmoothList = tempOffSmoothDf.stack().groupby(level=0).sum()
    onRoughDf          = onRoughDf.assign(licks=licksOnRoughList)
    onSmoothDf         = onSmoothDf.assign(licks=licksOnSmoothList)
    offRoughDf         = offRoughDf.assign(licks=licksOffRoughList)
    offSmoothDf        = offSmoothDf.assign(licks=licksOffSmoothList)  
    # calculate proportions binned by time for laser on and off for each stimulus
    proportionsOnRough, plotCentersOnRough, errorOnRough       = bin_by_time_and_binary(onRoughDf,time_col='licks',bin_col='congruent',liml=liml,limr=limr,bins=bins)
    proportionsOnSmooth, plotCentersOnSmooth, errorOnSmooth    = bin_by_time_and_binary(onSmoothDf,time_col='licks',bin_col='congruent',liml=liml,limr=limr,bins=bins)
    proportionsOffRough, plotCentersOffRough, errorOffRough    = bin_by_time_and_binary(offRoughDf,time_col='licks',bin_col='congruent',liml=liml,limr=limr,bins=bins)
    proportionsOffSmooth, plotCentersOffSmooth, errorOffSmooth = bin_by_time_and_binary(offSmoothDf,time_col='licks',bin_col='congruent',liml=liml,limr=limr,bins=bins)
    # average centers of bins to correct for numpy rounding errors
    plotCentersList = [np.mean([p1,p2,p3,p4]) for p1,p2,p3,p4 in zip(plotCentersOnRough,plotCentersOnSmooth,plotCentersOffRough,plotCentersOffSmooth)]
    # plot
    fig, ax = plt.subplots()
    ax.plot(plotCentersList,proportionsOnRough,linestyle='--',color='#ffb82b',label='Laser ON (rough)') 
    ax.plot(plotCentersList,proportionsOnSmooth,linestyle='-',color='#ffb82b',label='Laser ON (smooth)') 
    ax.plot(plotCentersList,proportionsOffRough,linestyle='--',color='black',label='Laser OFF (rough)')  
    ax.plot(plotCentersList,proportionsOffSmooth,linestyle='-',color='black',label='Laser OFF (smooth)') 
    legend_without_duplicate_labels(ax)
    ax.autoscale()
    ax.set_ylim(bottom=0,top=1.05)
    ax.get_yaxis().set_ticks(np.linspace(0.25,1.0,4))
    ax.set_xlim(left=liml,right=limr)
    simpleaxis(ax)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.fill_between(plotCentersList, proportionsOnRough-errorOnRough, proportionsOnRough+errorOnRough, alpha=0.3, edgecolor='none', facecolor='#ffb82b')
    plt.fill_between(plotCentersList, proportionsOnSmooth-errorOnSmooth, proportionsOnSmooth+errorOnSmooth, alpha=0.3, edgecolor='none', facecolor='#ffb82b')
    plt.fill_between(plotCentersList, proportionsOffRough-errorOffRough, proportionsOffRough+errorOffRough, alpha=0.6, edgecolor='none', facecolor='silver')
    plt.fill_between(plotCentersList, proportionsOffSmooth-errorOffSmooth, proportionsOffSmooth+errorOffSmooth, alpha=0.6, edgecolor='none', facecolor='silver')
    plt.xlabel('Time [ms]')
    plt.ylabel('Lick Probability')
    #plt.show()
    plt.title("{}".format(titleStr2))
    plt.savefig(os.path.join(os.getcwd(),"{}_concat_congruent_STIM_{}.svg".format(titleStr,beh)))
    plt.close('all')
    return None

def makeCongruentLine(tmsDict,beh,per_mouse=True,liml=-2000,limr=2000):
    """Call CongruentLine function for each mouse (per_mouse=True)
    or for each genotype (per_mouse=False)"""
    genotypes = list(tmsDict.keys())
    for g in genotypes:
        plots = [CongruentLine(tm,beh,g,per_mouse,bins=51) for tm in tmsDict[g]] if per_mouse else CongruentLine(tmsDict[g],beh,g,per_mouse,bins=41)
        plots = [CongruentLineByStim(tm,beh,g,per_mouse,bins=21) for tm in tmsDict[g]] if per_mouse else CongruentLineByStim(tmsDict[g],beh,g,per_mouse,bins=41)
    return None

def CorrectLine(tm,beh,genotype,per_mouse=True,liml=-2000,limr=2000,bins=41):
    """Plots proportion of licks congruent with recorded choice as function of time"""
    # identify correct licks
    corrects  = np.where(tm.left.notna() & tm.stim.isin(['ROUGH']) | tm.right.notna() & tm.stim.isin(['SMOOTH']),1,0) 
    tm        = tm.copy().assign(correct=corrects)
    titleStr  = ''.join(tm.mouse.unique()) if per_mouse else genotype
    titleStr2 = "Correct Licks - {} : {} ({})".format(genotype,titleStr,beh) if per_mouse else "Correct Licks - {} ({})".format(titleStr,beh)
    # group tm by laser status
    onCorrectDf  = tm.groupby('opto').get_group('ON')
    offCorrectDf = tm.groupby('opto').get_group('OFF')
    # merge left and right lick times
    tempOnDf     = pd.DataFrame({'left':onCorrectDf.left, 'right':onCorrectDf.right})   
    tempOffDf    = pd.DataFrame({'left':offCorrectDf.left, 'right':offCorrectDf.right})  
    licksOnList  = tempOnDf.stack().groupby(level=0).sum() 
    licksOffList = tempOffDf.stack().groupby(level=0).sum() 
    onCorrectDf  = onCorrectDf.assign(licks=licksOnList)
    offCorrectDf = offCorrectDf.assign(licks=licksOffList)  
    # calculate proportions binned by time for laser on and off
    proportionsOn, plotCentersOn, errorOn    = bin_by_time_and_binary(onCorrectDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
    proportionsOff, plotCentersOff, errorOff = bin_by_time_and_binary(offCorrectDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
    # average centers of bins to correct for numpy rounding errors
    plotCentersList = [np.mean([p1,p2]) for p1,p2 in zip(plotCentersOn,plotCentersOff)]
    # plot
    fig, ax = plt.subplots()
    ax.plot(plotCentersList,proportionsOn,linestyle='-',color='#ffb82b',label='Laser ON')  
    ax.plot(plotCentersList,proportionsOff,linestyle='-',color='black',label='Laser OFF') 
    legend_without_duplicate_labels(ax)
    ax.autoscale()
    ax.set_ylim(bottom=0,top=1.05)
    ax.get_yaxis().set_ticks(np.linspace(0.25,1.0,4))
    ax.set_xlim(left=liml,right=limr)
    simpleaxis(ax)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.fill_between(plotCentersList, proportionsOn-errorOn, proportionsOn+errorOn, alpha=0.3, edgecolor='none', facecolor='#ffb82b')
    plt.fill_between(plotCentersList, proportionsOff-errorOff, proportionsOff+errorOff, alpha=0.6, edgecolor='none', facecolor='silver')
    plt.xlabel('Time [ms]')
    plt.ylabel('Lick Probability')
    plt.title("{}".format(titleStr2))
    plt.savefig(os.path.join(os.getcwd(),"{}_concat_correct_{}.svg".format(titleStr,beh)))
    plt.close('all')
    return None

def CorrectLineByStim(tm,beh,genotype,per_mouse=True,liml=-2000,limr=2000,bins=41):
    """Plots proportion of licks congruent with recorded choice as function of time"""
    # identify correct licks
    corrects  = np.where(tm.left.notna() & tm.stim.isin(['ROUGH']) | tm.right.notna() & tm.stim.isin(['SMOOTH']),1,0) 
    tm        = tm.copy().assign(correct=corrects)
    titleStr  = ''.join(tm.mouse.unique()) if per_mouse else genotype
    titleStr2 = "Correct Licks (by stim) - {} : {} ({})".format(genotype,titleStr,beh) if per_mouse else "Correct Licks (by stim) - {} ({})".format(titleStr,beh)
    # group tm by laser status
    onCorrectDf  = tm.groupby('opto').get_group('ON')
    offCorrectDf = tm.groupby('opto').get_group('OFF')
    # grup by stimulus
    onRoughDf   = onCorrectDf.groupby('stim').get_group('ROUGH')
    onSmoothDf  = onCorrectDf.groupby('stim').get_group('SMOOTH')
    offRoughDf  = offCorrectDf.groupby('stim').get_group('ROUGH')
    offSmoothDf = offCorrectDf.groupby('stim').get_group('SMOOTH')
    # merge left and right lick times
    tempOnRoughDf      = pd.DataFrame({'left':onRoughDf.left, 'right':onRoughDf.right})  
    tempOnSmoothDf     = pd.DataFrame({'left':onSmoothDf.left, 'right':onSmoothDf.right})  
    tempOffRoughDf     = pd.DataFrame({'left':offRoughDf.left, 'right':offRoughDf.right})  
    tempOffSmoothDf    = pd.DataFrame({'left':offSmoothDf.left, 'right':offSmoothDf.right})
    licksOnRoughList   = tempOnRoughDf.stack().groupby(level=0).sum()
    licksOnSmoothList  = tempOnSmoothDf.stack().groupby(level=0).sum()
    licksOffRoughList  = tempOffRoughDf.stack().groupby(level=0).sum()
    licksOffSmoothList = tempOffSmoothDf.stack().groupby(level=0).sum()
    onRoughDf          = onRoughDf.assign(licks=licksOnRoughList)
    onSmoothDf         = onSmoothDf.assign(licks=licksOnSmoothList)
    offRoughDf         = offRoughDf.assign(licks=licksOffRoughList)
    offSmoothDf        = offSmoothDf.assign(licks=licksOffSmoothList)  
    # calculate proportions binned by time for laser on and off for each stimulus
    proportionsOnRough, plotCentersOnRough, errorOnRough       = bin_by_time_and_binary(onRoughDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
    proportionsOnSmooth, plotCentersOnSmooth, errorOnSmooth    = bin_by_time_and_binary(onSmoothDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
    proportionsOffRough, plotCentersOffRough, errorOffRough    = bin_by_time_and_binary(offRoughDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
    proportionsOffSmooth, plotCentersOffSmooth, errorOffSmooth = bin_by_time_and_binary(offSmoothDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
    # average centers of bins to correct for numpy rounding errors
    plotCentersList = [np.mean([p1,p2,p3,p4]) for p1,p2,p3,p4 in zip(plotCentersOnRough,plotCentersOnSmooth,plotCentersOffRough,plotCentersOffSmooth)]
    # plot
    fig, ax = plt.subplots()
    ax.plot(plotCentersList,proportionsOnRough,linestyle='--',color='#ffb82b',label='Laser ON (rough)') 
    ax.plot(plotCentersList,proportionsOnSmooth,linestyle='-',color='#ffb82b',label='Laser ON (smooth)') 
    ax.plot(plotCentersList,proportionsOffRough,linestyle='--',color='black',label='Laser OFF (rough)')  
    ax.plot(plotCentersList,proportionsOffSmooth,linestyle='-',color='black',label='Laser OFF (smooth)') 
    legend_without_duplicate_labels(ax)
    ax.autoscale()
    ax.set_ylim(bottom=0,top=1.05)
    ax.get_yaxis().set_ticks(np.linspace(0.25,1.0,4))
    ax.set_xlim(left=liml,right=limr)
    simpleaxis(ax)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.fill_between(plotCentersList, proportionsOnRough-errorOnRough, proportionsOnRough+errorOnRough, alpha=0.3, edgecolor='none', facecolor='#ffb82b')
    plt.fill_between(plotCentersList, proportionsOnSmooth-errorOnSmooth, proportionsOnSmooth+errorOnSmooth, alpha=0.3, edgecolor='none', facecolor='#ffb82b')
    plt.fill_between(plotCentersList, proportionsOffRough-errorOffRough, proportionsOffRough+errorOffRough, alpha=0.6, edgecolor='none', facecolor='silver')
    plt.fill_between(plotCentersList, proportionsOffSmooth-errorOffSmooth, proportionsOffSmooth+errorOffSmooth, alpha=0.6, edgecolor='none', facecolor='silver')
    plt.xlabel('Time [ms]')
    plt.ylabel('Lick Probability')
    plt.title("{}".format(titleStr2))
    plt.savefig(os.path.join(os.getcwd(),"{}_concat_correct_STIM_{}.svg".format(titleStr,beh)))
    plt.close('all')
    return None

def makeCorrectLine(tmsDict,beh,per_mouse=True,liml=-2000,limr=2000):
    """Call CorrectLine function for each mouse (per_mouse=True)
    or for each genotype (per_mouse=False)"""
    genotypes = list(tmsDict.keys())
    for g in genotypes:
        plots = [CorrectLine(tm,beh,g,per_mouse,bins=51) for tm in tmsDict[g]] if per_mouse else CorrectLine(tmsDict[g],beh,g,per_mouse,bins=41)
        plots = [CorrectLineByStim(tm,beh,g,per_mouse,bins=21) for tm in tmsDict[g]] if per_mouse else CorrectLineByStim(tmsDict[g],beh,g,per_mouse,bins=41)
    return None

def CongruentLine2(tmList,beh,genotype,liml=-2000,limr=2000,bins=41):
    """Plots proportion of licks congruent with recorded choice as function of time"""
    proportionsOnDict  = defaultdict(list)
    proportionsOffDict = defaultdict(list)
    plotCentersDict    = defaultdict(list)
    titleStr           = genotype
    titleStr2          = "Congruent Licks - {} ({})".format(titleStr,beh)
    # for each mouse
    for tm in tmList:
        # gather mouse name
        mouse = ''.join(tm.mouse.unique())
        # identify correct licks
        decisions  = generate_decision_column(tm)
        tm         = tm.copy().assign(decision=decisions)
        congruents = np.where(tm.left.notna() & tm.decision.isin(['left']) | tm.right.notna() & tm.decision.isin(['right']),1,0)
        tm         = tm.assign(congruent=congruents)
        # group tm by laser status
        onCongruentDf  = tm.groupby('opto').get_group('ON')
        offCongruentDf = tm.groupby('opto').get_group('OFF')
        # merge left and right lick times
        tempOnDf       = pd.DataFrame({'left':onCongruentDf.left, 'right':onCongruentDf.right})   
        tempOffDf      = pd.DataFrame({'left':offCongruentDf.left, 'right':offCongruentDf.right})  
        licksOnList    = tempOnDf.stack().groupby(level=0).sum() 
        licksOffList   = tempOffDf.stack().groupby(level=0).sum() 
        onCongruentDf  = onCongruentDf.assign(licks=licksOnList)
        offCongruentDf = offCongruentDf.assign(licks=licksOffList)  
        # calculate proportions binned by time for laser on and off
        proportionsOn, plotCentersOn, errorOn    = bin_by_time_and_binary(onCongruentDf,time_col='licks',bin_col='congruent',liml=liml,limr=limr,bins=bins)
        proportionsOff, plotCentersOff, errorOff = bin_by_time_and_binary(offCongruentDf,time_col='licks',bin_col='congruent',liml=liml,limr=limr,bins=bins)
        # average centers of bins to correct for numpy rounding errors
        plotCentersList = [np.mean([p1,p2]) for p1,p2 in zip(plotCentersOn,plotCentersOff)]
        # add to dictionaries
        proportionsOnDict[mouse]  = proportionsOn.values
        proportionsOffDict[mouse] = proportionsOff.values
        plotCentersDict[mouse]    = plotCentersList
    # generate dataframes
    proportionsOnDf  = pd.DataFrame(proportionsOnDict)
    proportionsOffDf = pd.DataFrame(proportionsOffDict)
    plotCentersDf    = pd.DataFrame(plotCentersDict)
    # calculate means and errors
    proportionsOnMeanList  = proportionsOnDf.mean(axis=1).values
    proportionsOnSEMList   = proportionsOnDf.sem(axis=1).values
    proportionsOffMeanList = proportionsOffDf.mean(axis=1).values
    proportionsOffSEMList  = proportionsOffDf.sem(axis=1).values
    # average centers of bins again to correct for numpy rounding errors
    plotCentersMeanList = plotCentersDf.mean(axis=1).values
    # plot
    fig, ax = plt.subplots()
    ax.plot(plotCentersMeanList,proportionsOnMeanList,linestyle='-',color='#ffb82b',label='Laser ON')  
    ax.plot(plotCentersMeanList,proportionsOffMeanList,linestyle='-',color='black',label='Laser OFF') 
    legend_without_duplicate_labels(ax)
    ax.autoscale()
    ax.set_ylim(bottom=0,top=1.05)
    ax.get_yaxis().set_ticks(np.linspace(0.25,1.0,4))
    ax.set_xlim(left=liml,right=limr)
    simpleaxis(ax)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.fill_between(plotCentersMeanList, proportionsOnMeanList-proportionsOnSEMList, proportionsOnMeanList+proportionsOnSEMList, alpha=0.3, edgecolor='none', facecolor='#ffb82b')
    plt.fill_between(plotCentersMeanList, proportionsOffMeanList-proportionsOffSEMList, proportionsOffMeanList+proportionsOffSEMList, alpha=0.6, edgecolor='none', facecolor='silver')
    plt.xlabel('Time [ms]')
    plt.ylabel('Lick Probability')
    plt.title("{}".format(titleStr2))
    #plt.show()
    plt.savefig(os.path.join(os.getcwd(),"{}_concat_congruent_{}.svg".format(titleStr,beh)))
    plt.close('all')
    return None

def CongruentLineByStim2(tmList,beh,genotype,liml=-2000,limr=2000,bins=41):
    """Plots proportion of licks licks congruent with recorded choice as function of time"""
    proportionsOnRoughDict   = defaultdict(list)
    proportionsOnSmoothDict  = defaultdict(list)
    proportionsOffRoughDict  = defaultdict(list)
    proportionsOffSmoothDict = defaultdict(list)
    plotCentersDict          = defaultdict(list)
    titleStr                 = genotype
    titleStr2                = "Congruent Licks (by stim) - {} ({})".format(titleStr,beh)
    # for each mouse
    for tm in tmList:
        # gather mouse name
        mouse = ''.join(tm.mouse.unique())
        # identify congruent licks
        decisions  = generate_decision_column(tm)
        tm         = tm.copy().assign(decision=decisions)
        congruents = np.where((tm.left.notna() & tm.decision.isin(['left'])) | (tm.right.notna() & tm.decision.isin(['right'])),1,0)
        tm         = tm.assign(congruent=congruents)
        # group tm by laser status
        onCongruentDf  = tm.groupby('opto').get_group('ON')
        offCongruentDf = tm.groupby('opto').get_group('OFF')
        # grup by stimulus
        onRoughDf   = onCongruentDf.groupby('stim').get_group('ROUGH')
        onSmoothDf  = onCongruentDf.groupby('stim').get_group('SMOOTH')
        offRoughDf  = offCongruentDf.groupby('stim').get_group('ROUGH')
        offSmoothDf = offCongruentDf.groupby('stim').get_group('SMOOTH')
        # merge left and right lick times
        tempOnRoughDf      = pd.DataFrame({'left':onRoughDf.left, 'right':onRoughDf.right})  
        tempOnSmoothDf     = pd.DataFrame({'left':onSmoothDf.left, 'right':onSmoothDf.right})  
        tempOffRoughDf     = pd.DataFrame({'left':offRoughDf.left, 'right':offRoughDf.right})  
        tempOffSmoothDf    = pd.DataFrame({'left':offSmoothDf.left, 'right':offSmoothDf.right})
        licksOnRoughList   = tempOnRoughDf.stack().groupby(level=0).sum()
        licksOnSmoothList  = tempOnSmoothDf.stack().groupby(level=0).sum()
        licksOffRoughList  = tempOffRoughDf.stack().groupby(level=0).sum()
        licksOffSmoothList = tempOffSmoothDf.stack().groupby(level=0).sum()
        onRoughDf          = onRoughDf.assign(licks=licksOnRoughList)
        onSmoothDf         = onSmoothDf.assign(licks=licksOnSmoothList)
        offRoughDf         = offRoughDf.assign(licks=licksOffRoughList)
        offSmoothDf        = offSmoothDf.assign(licks=licksOffSmoothList)  
        # calculate proportions binned by time for laser on and off for each stimulus
        proportionsOnRough, plotCentersOnRough, errorOnRough       = bin_by_time_and_binary(onRoughDf,time_col='licks',bin_col='congruent',liml=liml,limr=limr,bins=bins)
        proportionsOnSmooth, plotCentersOnSmooth, errorOnSmooth    = bin_by_time_and_binary(onSmoothDf,time_col='licks',bin_col='congruent',liml=liml,limr=limr,bins=bins)
        proportionsOffRough, plotCentersOffRough, errorOffRough    = bin_by_time_and_binary(offRoughDf,time_col='licks',bin_col='congruent',liml=liml,limr=limr,bins=bins)
        proportionsOffSmooth, plotCentersOffSmooth, errorOffSmooth = bin_by_time_and_binary(offSmoothDf,time_col='licks',bin_col='congruent',liml=liml,limr=limr,bins=bins)
        # average centers of bins to correct for numpy rounding errors
        plotCentersList = [np.mean([p1,p2,p3,p4]) for p1,p2,p3,p4 in zip(plotCentersOnRough,plotCentersOnSmooth,plotCentersOffRough,plotCentersOffSmooth)]
        # add to dictionaries
        proportionsOnRoughDict[mouse]   = proportionsOnRough.values
        proportionsOnSmoothDict[mouse]  = proportionsOnSmooth.values
        proportionsOffRoughDict[mouse]  = proportionsOffRough.values
        proportionsOffSmoothDict[mouse] = proportionsOffSmooth.values
        plotCentersDict[mouse]          = plotCentersList
    # generate dataframes
    proportionsOnRoughDf   = pd.DataFrame(proportionsOnRoughDict)
    proportionsOnSmoothDf  = pd.DataFrame(proportionsOnSmoothDict)
    proportionsOffRoughDf  = pd.DataFrame(proportionsOffRoughDict)
    proportionsOffSmoothDf = pd.DataFrame(proportionsOffSmoothDict)
    plotCentersDf          = pd.DataFrame(plotCentersDict)
    # calculate means and errors
    proportionsOnRoughMeanList   = proportionsOnRoughDf.mean(axis=1).values
    proportionsOnRoughSEMList    = proportionsOnRoughDf.sem(axis=1).values
    proportionsOnSmoothMeanList  = proportionsOnSmoothDf.mean(axis=1).values
    proportionsOnSmoothSEMList   = proportionsOnSmoothDf.sem(axis=1).values
    proportionsOffRoughMeanList  = proportionsOffRoughDf.mean(axis=1).values
    proportionsOffRoughSEMList   = proportionsOffRoughDf.sem(axis=1).values
    proportionsOffSmoothMeanList = proportionsOffSmoothDf.mean(axis=1).values
    proportionsOffSmoothSEMList  = proportionsOffSmoothDf.sem(axis=1).values
    # average centers of bins again to correct for numpy rounding errors
    plotCentersMeanList = plotCentersDf.mean(axis=1).values
    # plot
    fig, ax = plt.subplots()
    ax.plot(plotCentersMeanList,proportionsOnRoughMeanList,linestyle='--',color='#ffb82b',label='Laser ON (rough)') 
    ax.plot(plotCentersMeanList,proportionsOnSmoothMeanList,linestyle='-',color='#ffb82b',label='Laser ON (smooth)') 
    ax.plot(plotCentersMeanList,proportionsOffRoughMeanList,linestyle='--',color='black',label='Laser OFF (rough)')  
    ax.plot(plotCentersMeanList,proportionsOffSmoothMeanList,linestyle='-',color='black',label='Laser OFF (smooth)') 
    legend_without_duplicate_labels(ax)
    ax.autoscale()
    ax.set_ylim(bottom=0,top=1.05)
    ax.get_yaxis().set_ticks(np.linspace(0.25,1.0,4))
    ax.set_xlim(left=liml,right=limr)
    simpleaxis(ax)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.fill_between(plotCentersMeanList, proportionsOnRoughMeanList-proportionsOnRoughSEMList, proportionsOnRoughMeanList+proportionsOnRoughSEMList, alpha=0.3, edgecolor='none', facecolor='#ffb82b')
    plt.fill_between(plotCentersMeanList, proportionsOnSmoothMeanList-proportionsOnSmoothSEMList, proportionsOnSmoothMeanList+proportionsOnSmoothSEMList, alpha=0.3, edgecolor='none', facecolor='#ffb82b')
    plt.fill_between(plotCentersMeanList, proportionsOffRoughMeanList-proportionsOffRoughSEMList, proportionsOffRoughMeanList+proportionsOffRoughSEMList, alpha=0.6, edgecolor='none', facecolor='silver')
    plt.fill_between(plotCentersMeanList, proportionsOffSmoothMeanList-proportionsOffSmoothSEMList, proportionsOffSmoothMeanList+proportionsOffSmoothSEMList, alpha=0.6, edgecolor='none', facecolor='silver')
    plt.xlabel('Time [ms]')
    plt.ylabel('Lick Probability')
    plt.title("{}".format(titleStr2))
    plt.savefig(os.path.join(os.getcwd(),"{}_concat_congruent_STIM_{}.svg".format(titleStr,beh)))
    plt.close('all')
    return None

def makeCongruentLine2(tmsDict,beh,liml=-2000,limr=2000):
    """Call CongruentLine2 function for for each genotype with adjusted statistics"""
    genotypes = list(tmsDict.keys())
    for g in genotypes:
        plots = CongruentLine2(tmsDict[g],beh,g,bins=41)
        plots = CongruentLineByStim2(tmsDict[g],beh,g,bins=21)
    return None

def CongruentLineByStim2Lesion(tmList,beh,day,liml=-2000,limr=2000,bins=41):
    """Plots proportion of licks licks congruent with recorded choice as function of time.
    Used for lesion data."""
    # identify correct licks
    proportionsRoughDict  = defaultdict(list)
    proportionsSmoothDict = defaultdict(list)
    plotCentersDict       = defaultdict(list)
    titleStr              = day
    titleStr2             = "Congruent Licks (by stim) - day {} ({})".format(titleStr,beh)
    # for each mouse
    for tm in tmList:
        # gather mouse name
        mouse = ''.join(tm.mouse.unique())
        # identify congruent licks
        decisions  = generate_decision_column(tm)
        tm         = tm.copy().assign(decision=decisions)
        congruents = np.where((tm.left.notna() & tm.decision.isin(['left'])) | (tm.right.notna() & tm.decision.isin(['right'])),1,0)
        tm         = tm.assign(congruent=congruents)
        # grup by stimulus
        roughDf  = tm.groupby('stim').get_group('ROUGH')
        smoothDf = tm.groupby('stim').get_group('SMOOTH')
        # merge left and right lick times 
        tempRoughDf     = pd.DataFrame({'left':roughDf.left, 'right':roughDf.right})  
        tempSmoothDf    = pd.DataFrame({'left':smoothDf.left, 'right':smoothDf.right})
        licksRoughList  = tempRoughDf.stack().groupby(level=0).sum()
        licksSmoothList = tempSmoothDf.stack().groupby(level=0).sum()
        roughDf         = roughDf.assign(licks=licksRoughList)
        smoothDf        = smoothDf.assign(licks=licksSmoothList)
        # calculate proportions binned by time for laser on and off for each stimulus
        proportionsRough, plotCentersRough, errorRough    = bin_by_time_and_binary(roughDf,time_col='licks',bin_col='congruent',liml=liml,limr=limr,bins=bins)
        proportionsSmooth, plotCentersSmooth, errorSmooth = bin_by_time_and_binary(smoothDf,time_col='licks',bin_col='congruent',liml=liml,limr=limr,bins=bins)
        # average centers of bins to correct for numpy rounding errors
        plotCentersList = [np.mean([p1,p2]) for p1,p2 in zip(plotCentersRough,plotCentersSmooth)]
        # add to dictionaries
        proportionsRoughDict[mouse]   = proportionsRough.values
        proportionsSmoothDict[mouse]  = proportionsSmooth.values
        plotCentersDict[mouse]        = plotCentersList
    # generate dataframes
    proportionsRoughDf   = pd.DataFrame(proportionsRoughDict)
    proportionsSmoothDf  = pd.DataFrame(proportionsSmoothDict)
    plotCentersDf        = pd.DataFrame(plotCentersDict)
    # calculate means and errors
    proportionsRoughMeanList  = proportionsRoughDf.mean(axis=1).values
    proportionsRoughSEMList   = proportionsRoughDf.sem(axis=1).values
    proportionsSmoothMeanList = proportionsSmoothDf.mean(axis=1).values
    proportionsSmoothSEMList  = proportionsSmoothDf.sem(axis=1).values
    # average centers of bins again to correct for numpy rounding errors
    plotCentersMeanList = plotCentersDf.mean(axis=1).values
    # plot
    fig, ax = plt.subplots()
    ax.plot(plotCentersMeanList,proportionsRoughMeanList,linestyle='--',color='black',label='Rough')  
    ax.plot(plotCentersMeanList,proportionsSmoothMeanList,linestyle='-',color='black',label='Smooth') 
    legend_without_duplicate_labels(ax)
    ax.autoscale()
    ax.set_ylim(bottom=0,top=1.05)
    ax.get_yaxis().set_ticks(np.linspace(0.25,1.0,4))
    ax.set_xlim(left=liml,right=limr)
    simpleaxis(ax)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.fill_between(plotCentersMeanList, proportionsRoughMeanList-proportionsRoughSEMList, proportionsRoughMeanList+proportionsRoughSEMList, alpha=0.6, edgecolor='none', facecolor='silver')
    plt.fill_between(plotCentersMeanList, proportionsSmoothMeanList-proportionsSmoothSEMList, proportionsSmoothMeanList+proportionsSmoothSEMList, alpha=0.6, edgecolor='none', facecolor='silver')
    plt.xlabel('Time [ms]')
    plt.ylabel('Lick Probability')
    plt.title("{}".format(titleStr2))
    plt.savefig(os.path.join(os.getcwd(),"lesion_{}_concat_congruent_STIM_{}.svg".format(titleStr,beh)))
    plt.close('all')
    return None

def makeCongruentLine2Lesion(tmsDict,beh,liml=-2000,limr=2000):
    """Call CongruentLine2 function for for each genotype with adjusted statistics.
    Used for lesion data."""
    days = list(tmsDict.keys())
    for d in days:
        plots = CongruentLineByStim2Lesion(tmsDict[d],beh,d,bins=21)
    return None

def CorrectLine2(tmList,beh,genotype,liml=-2000,limr=2000,bins=41):
    """Plots proportion of licks correct with respect to stimulus as function of time"""
    proportionsOnDict  = defaultdict(list)
    proportionsOffDict = defaultdict(list)
    plotCentersDict    = defaultdict(list)
    titleStr           = genotype
    titleStr2          = "Correct Licks - {} ({})".format(titleStr,beh)
    # for each mouse
    for tm in tmList:
        # gather mouse name
        mouse = ''.join(tm.mouse.unique())
        # identify correct licks
        corrects  = np.where(tm.left.notna() & tm.stim.isin(['ROUGH']) | tm.right.notna() & tm.stim.isin(['SMOOTH']),1,0) 
        tm        = tm.copy().assign(correct=corrects)
        # group tm by laser status
        onCorrectDf  = tm.groupby('opto').get_group('ON')
        offCorrectDf = tm.groupby('opto').get_group('OFF')
        # merge left and right lick times
        tempOnDf     = pd.DataFrame({'left':onCorrectDf.left, 'right':onCorrectDf.right})   
        tempOffDf    = pd.DataFrame({'left':offCorrectDf.left, 'right':offCorrectDf.right})  
        licksOnList  = tempOnDf.stack().groupby(level=0).sum() 
        licksOffList = tempOffDf.stack().groupby(level=0).sum() 
        onCorrectDf  = onCorrectDf.assign(licks=licksOnList)
        offCorrectDf = offCorrectDf.assign(licks=licksOffList)  
        # calculate proportions binned by time for laser on and off
        proportionsOn, plotCentersOn, errorOn    = bin_by_time_and_binary(onCorrectDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
        proportionsOff, plotCentersOff, errorOff = bin_by_time_and_binary(offCorrectDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
        # average centers of bins to correct for numpy rounding errors
        plotCentersList = [np.mean([p1,p2]) for p1,p2 in zip(plotCentersOn,plotCentersOff)]
        # add to dictionaries
        proportionsOnDict[mouse]  = proportionsOn.values
        proportionsOffDict[mouse] = proportionsOff.values
        plotCentersDict[mouse]    = plotCentersList
    # generate dataframes
    proportionsOnDf  = pd.DataFrame(proportionsOnDict)
    proportionsOffDf = pd.DataFrame(proportionsOffDict)
    plotCentersDf    = pd.DataFrame(plotCentersDict)
    # calculate means and errors
    proportionsOnMeanList  = proportionsOnDf.mean(axis=1).values
    proportionsOnSEMList   = proportionsOnDf.sem(axis=1).values
    proportionsOffMeanList = proportionsOffDf.mean(axis=1).values
    proportionsOffSEMList  = proportionsOffDf.sem(axis=1).values
    # average centers of bins again to correct for numpy rounding errors
    plotCentersMeanList = plotCentersDf.mean(axis=1).values
    # plot
    fig, ax = plt.subplots()
    ax.plot(plotCentersMeanList,proportionsOnMeanList,linestyle='-',color='#ffb82b',label='Laser ON')  
    ax.plot(plotCentersMeanList,proportionsOffMeanList,linestyle='-',color='black',label='Laser OFF') 
    legend_without_duplicate_labels(ax)
    ax.autoscale()
    ax.set_ylim(bottom=0,top=1.05)
    ax.get_yaxis().set_ticks(np.linspace(0.25,1.0,4))
    ax.set_xlim(left=liml,right=limr)
    simpleaxis(ax)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.fill_between(plotCentersMeanList, proportionsOnMeanList-proportionsOnSEMList, proportionsOnMeanList+proportionsOnSEMList, alpha=0.3, edgecolor='none', facecolor='#ffb82b')
    plt.fill_between(plotCentersMeanList, proportionsOffMeanList-proportionsOffSEMList, proportionsOffMeanList+proportionsOffSEMList, alpha=0.6, edgecolor='none', facecolor='silver')
    plt.xlabel('Time [ms]')
    plt.ylabel('Lick Probability')
    plt.title("{}".format(titleStr2))
    #plt.show()
    plt.savefig(os.path.join(os.getcwd(),"{}_concat_correct_{}.svg".format(titleStr,beh)))
    plt.close('all')
    return None

def CorrectLineByStim2(tmList,beh,genotype,liml=-2000,limr=2000,bins=41):
    """Plots proportion of licks correct with respect to stimulus as function of time"""
    # identify correct licks
    proportionsOnRoughDict   = defaultdict(list)
    proportionsOnSmoothDict  = defaultdict(list)
    proportionsOffRoughDict  = defaultdict(list)
    proportionsOffSmoothDict = defaultdict(list)
    plotCentersDict          = defaultdict(list)
    titleStr                 = genotype
    titleStr2                = "Correct Licks (by stim) - {} ({})".format(titleStr,beh)
    # for each mouse
    for tm in tmList:
        # gather mouse name
        mouse = ''.join(tm.mouse.unique())
        # identify correct licks
        corrects = np.where((tm.left.notna() & tm.stim.isin(['ROUGH'])) | (tm.right.notna() & tm.stim.isin(['SMOOTH'])),1,0) 
        tm       = tm.copy().assign(correct=corrects)
        # group tm by laser status
        onCorrectDf  = tm.groupby('opto').get_group('ON')
        offCorrectDf = tm.groupby('opto').get_group('OFF')
        # grup by stimulus
        onRoughDf   = onCorrectDf.groupby('stim').get_group('ROUGH')
        onSmoothDf  = onCorrectDf.groupby('stim').get_group('SMOOTH')
        offRoughDf  = offCorrectDf.groupby('stim').get_group('ROUGH')
        offSmoothDf = offCorrectDf.groupby('stim').get_group('SMOOTH')
        # merge left and right lick times
        tempOnRoughDf      = pd.DataFrame({'left':onRoughDf.left, 'right':onRoughDf.right})  
        tempOnSmoothDf     = pd.DataFrame({'left':onSmoothDf.left, 'right':onSmoothDf.right})  
        tempOffRoughDf     = pd.DataFrame({'left':offRoughDf.left, 'right':offRoughDf.right})  
        tempOffSmoothDf    = pd.DataFrame({'left':offSmoothDf.left, 'right':offSmoothDf.right})
        licksOnRoughList   = tempOnRoughDf.stack().groupby(level=0).sum()
        licksOnSmoothList  = tempOnSmoothDf.stack().groupby(level=0).sum()
        licksOffRoughList  = tempOffRoughDf.stack().groupby(level=0).sum()
        licksOffSmoothList = tempOffSmoothDf.stack().groupby(level=0).sum()
        onRoughDf          = onRoughDf.assign(licks=licksOnRoughList)
        onSmoothDf         = onSmoothDf.assign(licks=licksOnSmoothList)
        offRoughDf         = offRoughDf.assign(licks=licksOffRoughList)
        offSmoothDf        = offSmoothDf.assign(licks=licksOffSmoothList)  
        # calculate proportions binned by time for laser on and off for each stimulus
        proportionsOnRough, plotCentersOnRough, errorOnRough       = bin_by_time_and_binary(onRoughDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
        proportionsOnSmooth, plotCentersOnSmooth, errorOnSmooth    = bin_by_time_and_binary(onSmoothDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
        proportionsOffRough, plotCentersOffRough, errorOffRough    = bin_by_time_and_binary(offRoughDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
        proportionsOffSmooth, plotCentersOffSmooth, errorOffSmooth = bin_by_time_and_binary(offSmoothDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
        # average centers of bins to correct for numpy rounding errors
        plotCentersList = [np.mean([p1,p2,p3,p4]) for p1,p2,p3,p4 in zip(plotCentersOnRough,plotCentersOnSmooth,plotCentersOffRough,plotCentersOffSmooth)]
        # add to dictionaries
        proportionsOnRoughDict[mouse]   = proportionsOnRough.values
        proportionsOnSmoothDict[mouse]  = proportionsOnSmooth.values
        proportionsOffRoughDict[mouse]  = proportionsOffRough.values
        proportionsOffSmoothDict[mouse] = proportionsOffSmooth.values
        plotCentersDict[mouse]          = plotCentersList
    # generate dataframes
    proportionsOnRoughDf   = pd.DataFrame(proportionsOnRoughDict)
    proportionsOnSmoothDf  = pd.DataFrame(proportionsOnSmoothDict)
    proportionsOffRoughDf  = pd.DataFrame(proportionsOffRoughDict)
    proportionsOffSmoothDf = pd.DataFrame(proportionsOffSmoothDict)
    plotCentersDf          = pd.DataFrame(plotCentersDict)
    # calculate means and errors
    proportionsOnRoughMeanList   = proportionsOnRoughDf.mean(axis=1).values
    proportionsOnRoughSEMList    = proportionsOnRoughDf.sem(axis=1).values
    proportionsOnSmoothMeanList  = proportionsOnSmoothDf.mean(axis=1).values
    proportionsOnSmoothSEMList   = proportionsOnSmoothDf.sem(axis=1).values
    proportionsOffRoughMeanList  = proportionsOffRoughDf.mean(axis=1).values
    proportionsOffRoughSEMList   = proportionsOffRoughDf.sem(axis=1).values
    proportionsOffSmoothMeanList = proportionsOffSmoothDf.mean(axis=1).values
    proportionsOffSmoothSEMList  = proportionsOffSmoothDf.sem(axis=1).values
    # average centers of bins again to correct for numpy rounding errors
    plotCentersMeanList = plotCentersDf.mean(axis=1).values
    # save to csv
    proportionsOnRoughDf.to_csv("correct_{}_ON_ROUGH.csv".format(beh),index=False)
    proportionsOnSmoothDf.to_csv("correct_{}_ON_SMOOTH.csv".format(beh),index=False)
    proportionsOffRoughDf.to_csv("correct_{}_OFF_ROUGH.csv".format(beh),index=False)
    proportionsOffSmoothDf.to_csv("correct_{}_OFF_SMOOTH.csv".format(beh),index=False)
    # plot
    fig, ax = plt.subplots()
    ax.plot(plotCentersMeanList,proportionsOnRoughMeanList,linestyle='--',color='#ffb82b',label='Laser ON (rough)') 
    ax.plot(plotCentersMeanList,proportionsOnSmoothMeanList,linestyle='-',color='#ffb82b',label='Laser ON (smooth)') 
    ax.plot(plotCentersMeanList,proportionsOffRoughMeanList,linestyle='--',color='black',label='Laser OFF (rough)')  
    ax.plot(plotCentersMeanList,proportionsOffSmoothMeanList,linestyle='-',color='black',label='Laser OFF (smooth)') 
    legend_without_duplicate_labels(ax)
    ax.autoscale()
    ax.set_ylim(bottom=0,top=1.05)
    ax.get_yaxis().set_ticks(np.linspace(0.25,1.0,4))
    ax.set_xlim(left=liml,right=limr)
    simpleaxis(ax)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.fill_between(plotCentersMeanList, proportionsOnRoughMeanList-proportionsOnRoughSEMList, proportionsOnRoughMeanList+proportionsOnRoughSEMList, alpha=0.3, edgecolor='none', facecolor='#ffb82b')
    plt.fill_between(plotCentersMeanList, proportionsOnSmoothMeanList-proportionsOnSmoothSEMList, proportionsOnSmoothMeanList+proportionsOnSmoothSEMList, alpha=0.3, edgecolor='none', facecolor='#ffb82b')
    plt.fill_between(plotCentersMeanList, proportionsOffRoughMeanList-proportionsOffRoughSEMList, proportionsOffRoughMeanList+proportionsOffRoughSEMList, alpha=0.6, edgecolor='none', facecolor='silver')
    plt.fill_between(plotCentersMeanList, proportionsOffSmoothMeanList-proportionsOffSmoothSEMList, proportionsOffSmoothMeanList+proportionsOffSmoothSEMList, alpha=0.6, edgecolor='none', facecolor='silver')
    plt.xlabel('Time [ms]')
    plt.ylabel('Lick Probability')
    plt.title("{}".format(titleStr2))
    plt.savefig(os.path.join(os.getcwd(),"{}_concat_correct_STIM_{}.svg".format(titleStr,beh)))
    plt.close('all')
    return None

def makeCorrectLine2(tmsDict,beh,liml=-2000,limr=2000):
    """Call CorrectLine2 function for for each genotype with adjusted statistics"""
    genotypes = list(tmsDict.keys())
    for g in genotypes:
        plots = CorrectLine2(tmsDict[g],beh,g,bins=41)
        plots = CorrectLineByStim2(tmsDict[g],beh,g,bins=21)
    return None

def CorrectLineByStim2Lesion(tmList,beh,day,liml=-2000,limr=2000,bins=41):
    """Plots proportion of licks correct with respect to stimulus as function of time.
    Used for lesion data."""
    # identify correct licks
    proportionsRoughDict  = defaultdict(list)
    proportionsSmoothDict = defaultdict(list)
    plotCentersDict       = defaultdict(list)
    titleStr              = day
    titleStr2             = "Correct Licks (by stim) - day {} ({})".format(titleStr,beh)
    # for each mouse
    for tm in tmList:
        # gather mouse name
        mouse = ''.join(tm.mouse.unique())
        # identify correct licks
        corrects = np.where((tm.left.notna() & tm.stim.isin(['ROUGH'])) | (tm.right.notna() & tm.stim.isin(['SMOOTH'])),1,0) 
        tm       = tm.copy().assign(correct=corrects)
        # grup by stimulus
        roughDf  = tm.groupby('stim').get_group('ROUGH')
        smoothDf = tm.groupby('stim').get_group('SMOOTH')
        # merge left and right lick times 
        tempRoughDf     = pd.DataFrame({'left':roughDf.left, 'right':roughDf.right})  
        tempSmoothDf    = pd.DataFrame({'left':smoothDf.left, 'right':smoothDf.right})
        licksRoughList  = tempRoughDf.stack().groupby(level=0).sum()
        licksSmoothList = tempSmoothDf.stack().groupby(level=0).sum()
        roughDf         = roughDf.assign(licks=licksRoughList)
        smoothDf        = smoothDf.assign(licks=licksSmoothList)
        # calculate proportions binned by time for laser on and off for each stimulus
        proportionsRough, plotCentersRough, errorRough    = bin_by_time_and_binary(roughDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
        proportionsSmooth, plotCentersSmooth, errorSmooth = bin_by_time_and_binary(smoothDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
        # average centers of bins to correct for numpy rounding errors
        plotCentersList = [np.mean([p1,p2]) for p1,p2 in zip(plotCentersRough,plotCentersSmooth)]
        # add to dictionaries
        proportionsRoughDict[mouse]   = proportionsRough.values
        proportionsSmoothDict[mouse]  = proportionsSmooth.values
        plotCentersDict[mouse]        = plotCentersList
    # generate dataframes
    proportionsRoughDf   = pd.DataFrame(proportionsRoughDict)
    proportionsSmoothDf  = pd.DataFrame(proportionsSmoothDict)
    plotCentersDf        = pd.DataFrame(plotCentersDict)
    # calculate means and errors
    proportionsRoughMeanList  = proportionsRoughDf.mean(axis=1).values
    proportionsRoughSEMList   = proportionsRoughDf.sem(axis=1).values
    proportionsSmoothMeanList = proportionsSmoothDf.mean(axis=1).values
    proportionsSmoothSEMList  = proportionsSmoothDf.sem(axis=1).values
    # average centers of bins again to correct for numpy rounding errors
    plotCentersMeanList = plotCentersDf.mean(axis=1).values
    # plot
    fig, ax = plt.subplots()
    ax.plot(plotCentersMeanList,proportionsRoughMeanList,linestyle='--',color='black',label='Rough')  
    ax.plot(plotCentersMeanList,proportionsSmoothMeanList,linestyle='-',color='black',label='Smooth') 
    legend_without_duplicate_labels(ax)
    ax.autoscale()
    ax.set_ylim(bottom=0,top=1.05)
    ax.get_yaxis().set_ticks(np.linspace(0.25,1.0,4))
    ax.set_xlim(left=liml,right=limr)
    simpleaxis(ax)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.fill_between(plotCentersMeanList, proportionsRoughMeanList-proportionsRoughSEMList, proportionsRoughMeanList+proportionsRoughSEMList, alpha=0.6, edgecolor='none', facecolor='silver')
    plt.fill_between(plotCentersMeanList, proportionsSmoothMeanList-proportionsSmoothSEMList, proportionsSmoothMeanList+proportionsSmoothSEMList, alpha=0.6, edgecolor='none', facecolor='silver')
    plt.xlabel('Time [ms]')
    plt.ylabel('Lick Probability')
    plt.title("{}".format(titleStr2))
    plt.savefig(os.path.join(os.getcwd(),"lesion_{}_concat_correct_STIM_{}.svg".format(titleStr,beh)))
    plt.close('all')
    return None

def makeCorrectLine2Lesion(tmsDict,beh,liml=-2000,limr=2000):
    """Call CorrectLine2 function for for each day with adjusted statistics.
    Used for lesion data."""
    days = list(tmsDict.keys())
    for d in days:
        plots = CorrectLineByStim2Lesion(tmsDict[d],beh,d,bins=21)
    return None

def CorrectLineByStimLikeOpto(tmsDict,beh,liml=-2000,limr=2000,bins=41):
    """Plots proportion of licks correct with respect to stimulus as function of time.
    Plots like opto, with averaged days -2 and -1 for 'laser off' and day 1 as 'laser on'.
    Used for lesion data."""
    # identify correct licks
    proportionsPreRoughDict   = defaultdict(list)
    proportionsPreSmoothDict  = defaultdict(list)
    proportionsPostRoughDict  = defaultdict(list)
    proportionsPostSmoothDict = defaultdict(list)
    plotCentersDict           = defaultdict(list)
    titleStr                  = 'lesion'
    titleStr2                 = "Correct Licks (by stim) - {} ({})".format(titleStr,beh)
    # for each mouse
    for preTM,postTM in zip(tmsDict['pre'],tmsDict['post']):
        # gather mouse name
        mouse = ''.join(preTM.mouse.unique())
        # identify correct licks
        preCorrects  = np.where((preTM.left.notna() & preTM.stim.isin(['ROUGH'])) | (preTM.right.notna() & preTM.stim.isin(['SMOOTH'])),1,0) 
        preDf        = preTM.copy().assign(correct=preCorrects)
        postCorrects = np.where((postTM.left.notna() & postTM.stim.isin(['ROUGH'])) | (postTM.right.notna() & postTM.stim.isin(['SMOOTH'])),1,0) 
        postDf       = postTM.copy().assign(correct=postCorrects)
        # grup by stimulus
        preRoughDf   = preDf.groupby('stim').get_group('ROUGH')
        preSmoothDf  = preDf.groupby('stim').get_group('SMOOTH')
        postRoughDf  = postDf.groupby('stim').get_group('ROUGH')
        postSmoothDf = postDf.groupby('stim').get_group('SMOOTH')
        # merge left and right lick times
        tempPreRoughDf      = pd.DataFrame({'left':preRoughDf.left, 'right':preRoughDf.right})  
        tempPreSmoothDf     = pd.DataFrame({'left':preSmoothDf.left, 'right':preSmoothDf.right})  
        tempPostRoughDf     = pd.DataFrame({'left':postRoughDf.left, 'right':postRoughDf.right})  
        tempPostSmoothDf    = pd.DataFrame({'left':postSmoothDf.left, 'right':postSmoothDf.right})
        licksPreRoughList   = tempPreRoughDf.stack().groupby(level=0).sum()
        licksPreSmoothList  = tempPreSmoothDf.stack().groupby(level=0).sum()
        licksPostRoughList  = tempPostRoughDf.stack().groupby(level=0).sum()
        licksPostSmoothList = tempPostSmoothDf.stack().groupby(level=0).sum()
        preRoughDf          = preRoughDf.assign(licks=licksPreRoughList)
        preSmoothDf         = preSmoothDf.assign(licks=licksPreSmoothList)
        postRoughDf         = postRoughDf.assign(licks=licksPostRoughList)
        postSmoothDf        = postSmoothDf.assign(licks=licksPostSmoothList)  
        # calculate proportions binned by time for laser on and off for each stimulus
        proportionsPreRough, plotCentersPreRough, errorPreRough       = bin_by_time_and_binary(preRoughDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
        proportionsPreSmooth, plotCentersPreSmooth, errorPreSmooth    = bin_by_time_and_binary(preSmoothDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
        proportionsPostRough, plotCentersPostRough, errorPostRough    = bin_by_time_and_binary(postRoughDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
        proportionsPostSmooth, plotCentersPostSmooth, errorPostSmooth = bin_by_time_and_binary(postSmoothDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
        # average centers of bins to correct for numpy rounding errors
        plotCentersList = [np.mean([p1,p2,p3,p4]) for p1,p2,p3,p4 in zip(plotCentersPreRough,plotCentersPreSmooth,plotCentersPostRough,plotCentersPostSmooth)]
        # add to dictionaries
        proportionsPreRoughDict[mouse]   = proportionsPreRough.values
        proportionsPreSmoothDict[mouse]  = proportionsPreSmooth.values
        proportionsPostRoughDict[mouse]  = proportionsPostRough.values
        proportionsPostSmoothDict[mouse] = proportionsPostSmooth.values
        plotCentersDict[mouse]           = plotCentersList
    # generate dataframes
    proportionsPreRoughDf   = pd.DataFrame(proportionsPreRoughDict)
    proportionsPreSmoothDf  = pd.DataFrame(proportionsPreSmoothDict)
    proportionsPostRoughDf  = pd.DataFrame(proportionsPostRoughDict)
    proportionsPostSmoothDf = pd.DataFrame(proportionsPostSmoothDict)
    plotCentersDf           = pd.DataFrame(plotCentersDict)
    # calculate means and errors
    proportionsPreRoughMeanList   = proportionsPreRoughDf.mean(axis=1).values
    proportionsPreRoughSEMList    = proportionsPreRoughDf.sem(axis=1).values
    proportionsPreSmoothMeanList  = proportionsPreSmoothDf.mean(axis=1).values
    proportionsPreSmoothSEMList   = proportionsPreSmoothDf.sem(axis=1).values
    proportionsPostRoughMeanList  = proportionsPostRoughDf.mean(axis=1).values
    proportionsPostRoughSEMList   = proportionsPostRoughDf.sem(axis=1).values
    proportionsPostSmoothMeanList = proportionsPostSmoothDf.mean(axis=1).values
    proportionsPostSmoothSEMList  = proportionsPostSmoothDf.sem(axis=1).values
    # average centers of bins again to correct for numpy rounding errors
    plotCentersMeanList = plotCentersDf.mean(axis=1).values
    # plot
    fig, ax = plt.subplots()
    ax.plot(plotCentersMeanList,proportionsPreRoughMeanList,linestyle='--',color='black',label='Pre-lesion (rough)') 
    ax.plot(plotCentersMeanList,proportionsPreSmoothMeanList,linestyle='-',color='black',label='Pre-lesion (smooth)') 
    ax.plot(plotCentersMeanList,proportionsPostRoughMeanList,linestyle='--',color='#ffb82b',label='Post-lesion (rough)')  
    ax.plot(plotCentersMeanList,proportionsPostSmoothMeanList,linestyle='-',color='#ffb82b',label='Post-lesion (smooth)') 
    legend_without_duplicate_labels(ax)
    ax.autoscale()
    ax.set_ylim(bottom=0,top=1.05)
    ax.get_yaxis().set_ticks(np.linspace(0.25,1.0,4))
    ax.set_xlim(left=liml,right=limr)
    simpleaxis(ax)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.fill_between(plotCentersMeanList, proportionsPreRoughMeanList-proportionsPreRoughSEMList, proportionsPreRoughMeanList+proportionsPreRoughSEMList, alpha=0.6, edgecolor='none', facecolor='silver')
    plt.fill_between(plotCentersMeanList, proportionsPreSmoothMeanList-proportionsPreSmoothSEMList, proportionsPreSmoothMeanList+proportionsPreSmoothSEMList, alpha=0.6, edgecolor='none', facecolor='silver')
    plt.fill_between(plotCentersMeanList, proportionsPostRoughMeanList-proportionsPostRoughSEMList, proportionsPostRoughMeanList+proportionsPostRoughSEMList, alpha=0.3, edgecolor='none', facecolor='#ffb82b')
    plt.fill_between(plotCentersMeanList, proportionsPostSmoothMeanList-proportionsPostSmoothSEMList, proportionsPostSmoothMeanList+proportionsPostSmoothSEMList, alpha=0.3, edgecolor='none', facecolor='#ffb82b')
    plt.xlabel('Time [ms]')
    plt.ylabel('Lick Probability')
    plt.title("{}".format(titleStr2))
    plt.savefig(os.path.join(os.getcwd(),"like_opto_{}_concat_correct_STIM_{}.svg".format(titleStr,beh)))
    plt.close('all')
    return None

def makeCorrectLineLikeOpto(tmsDict,beh,liml=-2000,limr=2000):
    """Call CorrectLineByStimLikeOpto function for each day with adjusted statistics.
    Used for lesion data."""
    plots = CorrectLineByStimLikeOpto(tmsDict,beh,bins=21)
    return None

def FirstLicksScatter(firstLicksDf,beh):
    """Visualize strategy change between laser on and off trials"""
    genotypesList  = firstLicksDf.genotype.unique()
    onLatencyList  = firstLicksDf.diffOn
    offLatencyList = firstLicksDf.diffOff
    # set order for legend
    orderedGenotypesList = ['Flex-Halo', 'Emx1-Halo', 'Cux2-Halo', 'Nr5a1-Halo', 'Fezf2-Halo', 'Rbp4-Halo']
    # calculate laser on and off latency mean and SEM
    averageOnLatencyDict  = {''.join(group[1].genotype.unique()):group[1].diffOn.mean() for group in firstLicksDf.groupby('genotype')}
    semOnLatencyDict      = {''.join(group[1].genotype.unique()):group[1].diffOn.sem() for group in firstLicksDf.groupby('genotype')}
    averageOffLatencyDict = {''.join(group[1].genotype.unique()):group[1].diffOff.mean() for group in firstLicksDf.groupby('genotype')}
    semOffLatencyDict     = {''.join(group[1].genotype.unique()):group[1].diffOff.sem() for group in firstLicksDf.groupby('genotype')}
    # plot
    fig, ax = plt.subplots()
    cmap    = {'Emx1-Halo':'#0eb554','Nr5a1-Halo':'#F26522','Flex-Halo':'black','Cux2-Halo':'#f0028f','Fezf2-Halo':'#7F3E98','Rbp4-Halo':'#213F9A'}
    colors  = [cmap[transgene] for transgene in genotypesList]
    for i,g in enumerate(genotypesList):
        xi = [offLatencyList[j] for j in range(len(offLatencyList)) if firstLicksDf.genotype[j] == g]
        yi = [onLatencyList[j] for j in range(len(onLatencyList)) if firstLicksDf.genotype[j] == g]
        ax.scatter(xi, yi, c=colors[i], label=g)
    reorderLegend(ax,orderedGenotypesList)
    ax.set_title('Laser-induced Behavioral Alterations ({})'.format(beh))                      
    ax.set_xlim(-1000,1000)  
    ax.set_ylim(-1000,1000) 
    # plot identity line
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])] 
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_yticks(np.linspace(-1000,1000,5))
    ax.set_xticks(np.linspace(-1000,1000,5))
    ax.set_aspect('equal', adjustable='box')
    ax.set_ylabel('Lick Latency (Laser ON) [ms]')
    ax.set_xlabel('Lick Latency (Laser OFF) [ms]')
    #plt.show()
    plt.savefig(os.path.join(os.getcwd(),"strategies_scatter_{}.svg".format(beh)))
    # save to csv
    firstLicksDf.to_csv("strategies_scatter_{}.csv".format(beh),index=False)
    # plot concatenated scatter
    fig, ax = plt.subplots()
    for i,g in enumerate(genotypesList):
        xi      = averageOffLatencyDict[g]
        yi      = averageOnLatencyDict[g]
        xiError = semOffLatencyDict[g]
        yiError = semOnLatencyDict[g]
        ax.scatter(xi, yi, c=colors[i], label=g)
        ax.errorbar(xi, yi, xerr=xiError, yerr=yiError, fmt='none', c=colors[i])
    reorderLegend(ax,orderedGenotypesList)
    ax.set_title('Laser-induced Behavioral Alterations ({})'.format(beh))                      
    ax.set_xlim(-1000,1000)  
    ax.set_ylim(-1000,1000) 
    # plot identity line
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])] 
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_yticks(np.linspace(-1000,1000,5))
    ax.set_xticks(np.linspace(-1000,1000,5))
    ax.set_aspect('equal', adjustable='box')
    ax.set_ylabel('Lick Latency (Laser ON) [ms]')
    ax.set_xlabel('Lick Latency (Laser OFF) [ms]')
    #plt.show()
    plt.savefig(os.path.join(os.getcwd(),"strategies_concat_scatter_{}.svg".format(beh)))

def FirstLicksArrows(firstLicksDf,beh):
    """Visualize strategy change between laser on and off trials"""
    # sort data by genotype
    orderedGenotypesList  = ['Flex-Halo', 'Emx1-Halo', 'Cux2-Halo', 'Nr5a1-Halo', 'Fezf2-Halo', 'Rbp4-Halo']
    categoryGenotypes     = pd.api.types.CategoricalDtype(categories=orderedGenotypesList, ordered=True)
    firstLicksDf.genotype = firstLicksDf.genotype.astype(categoryGenotypes)
    firstLicksDf          = firstLicksDf.sort_values(by='genotype',ascending=False)
    # plot
    fig, ax = plt.subplots()
    n       = len(firstLicksDf.index)
    ax.set_xlim(-1,1)                        
    ax.set_ylim(-1,n)                        
    ax.set_yticks(range(n))                  
    ax.set_yticklabels(firstLicksDf.genotype.astype(str))  
    ax.axvline(x=0, color='0.9', ls='--', lw=2, zorder=0)
    ax.grid(axis='y', color='0.9')  
    # define arrows
    arrowStarts       = firstLicksDf.diffOff
    arrowEnds         = firstLicksDf.diffOn
    #absmax            = max(firstLicksDf.diffOn.abs().max(),firstLicksDf.diffOff.abs().max(),firstLicksDf.diffs.abs().max()) 
    #arrowStartsScaled = arrowStarts/absmax
    #arrowEndsScaled   = arrowEnds/absmax
    arrowLengths      = arrowEnds - arrowStarts
    #add arrows to plot
    for i in range(n):
        if arrowLengths[i] > 0:
            arrowColor = 'red'
        elif arrowLengths[i] < 0:
            arrowColor = 'blue'
        else:
            arrowColor = 'black'
        ax.arrow(arrowStarts[i],i,arrowLengths[i],0,head_width=0.6,head_length=25,width=0.2,facecolor=arrowColor,edgecolor=arrowColor,length_includes_head=True)
    ax.set_title('Laser-induced Behavioral Alterations ({})'.format(beh))                      
    ax.set_xlim(-1000,1000)                                     
    ax.set_xlabel('Lick Latency [ms]')                                               
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(),"strategies_arrows_{}.svg".format(beh)))
    plt.close('all')
    return None

def FirstLicksHisto(firstLicksDf,beh):
    """Plots histogram of average time difference between first left and right lick for each mouse.
    First, separates laser on and off trials. Second, plot all concatenated"""
    diffOn        = firstLicksDf.diffOn
    diffOff       = firstLicksDf.diffOff
    diffs         = firstLicksDf.diffs
    #absmax        = max(diffOn.abs().max(),diffOff.abs().max(),diffs.abs().max()) 
    #diffOnScaled  = diffOn/absmax
    #diffOffScaled = diffOff/absmax
    #diffScaled    = diffs/absmax
    bins          = np.linspace(-1000,1000,10)
    fig, ax       = plt.subplots()
    # plot
    ax.hist(diffOff, bins, color='silver', alpha=0.7, label='Laser OFF', weights=np.zeros_like(np.array(diffOff)) + 1. / np.array(diffOff).size)
    ax.hist(diffOn, bins, color='#ffb82b', alpha=0.7, label='Laser ON', weights=np.zeros_like(np.array(diffOn)) + 1. / np.array(diffOn).size) 
    legend_without_duplicate_labels(ax)
    ax.set_ylim(bottom=0,top=0.6)
    ax.set_ylabel('Density')
    ax.set_xlabel('Lick Latency [ms]')
    ax.get_yaxis().set_ticks(np.linspace(0.1,0.6,6))
    ax.legend(loc='upper right', fancybox=False, edgecolor='k', framealpha=1) 
    plt.title("Lick Strategies - {}".format(beh))
    plt.savefig(os.path.join(os.getcwd(),"strategies_laserONOFF_{}.svg".format(beh)))
    plt.close('all')
    # plot 2 (no consideration for laser)
    bins    = np.linspace(-1000,1000,10)
    fig, ax = plt.subplots()
    ax.hist(diffs, bins, color='fuchsia', alpha=0.7, label='ALL TRIALS', weights=np.zeros_like(np.array(diffs)) + 1. / np.array(diffs).size)  
    legend_without_duplicate_labels(ax)
    ax.set_ylim(bottom=0,top=0.6)
    ax.set_ylabel('Density')
    ax.set_xlabel('Lick Latency [ms]')
    ax.get_yaxis().set_ticks(np.linspace(0.1,0.6,6)) 
    ax.legend(loc='upper right', fancybox=False, edgecolor='k', framealpha=1)  
    plt.title("Lick Strategies - {}".format(beh))  
    plt.savefig(os.path.join(os.getcwd(),"strategies_concat_{}.svg".format(beh)))
    plt.close('all')

def makeFirstLickPlots(tmList,beh,genesDict):
    """Plots histogram of average time difference between first left and right lick for each mouse.
    Zero indicates mice with no obvious behavioral lick-left or -right strategy.
    Negative indicates mice with a lick-left default strategy.
    Same values represented on arrow and scatter plots."""
    # select only correct trials
    #tmListCorrect = [tm.groupby('outcome').get_group(1) for tm in tmList] # probably not necessary
    tmListCorrect = tmList
    # clamp to reasonable values
    tmListCorrClamped = [tm.iloc[np.where((tm.left.notna() & (tm.left > -1000) & (tm.left < 1000)) |
                                          (tm.right.notna() & (tm.right > -1000) & (tm.right < 1000)))] for tm in tmListCorrect]
    # determine average first lick for left and right for each mouse
    firstLicksDict = {}
    for tm in tmListCorrClamped:
        # gather mouse name
        mouse = ''.join(tm.mouse.unique())
        # gather genotype
        genotype = genesDict[mouse]
        # group tm by laser status
        onDf  = tm.groupby('opto').get_group('ON')
        offDf = tm.groupby('opto').get_group('OFF')
        # split by trial
        trialListOn  = [group[1] for group in onDf.groupby('trial')]  
        trialListOff = [group[1] for group in offDf.groupby('trial')] 
        # first left lick per trial
        #firstLeftListOn  = [trial.loc[trial.left.first_valid_index()].left if trial.left.first_valid_index() is not None else np.nan for trial in trialListOn] 
        #firstLeftListOff = [trial.loc[trial.left.first_valid_index()].left if trial.left.first_valid_index() is not None else np.nan for trial in trialListOff] 
        # first left lick per ROUGH trial
        firstLeftListOn  = [trial.loc[trial.left.first_valid_index()].left if (trial.left.first_valid_index() is not None and trial.stim.unique() == 'ROUGH') else np.nan for trial in trialListOn] 
        firstLeftListOff = [trial.loc[trial.left.first_valid_index()].left if (trial.left.first_valid_index() is not None and trial.stim.unique() == 'ROUGH') else np.nan for trial in trialListOff] 
        # first right lick per trial
        #firstRightListOn  = [trial.loc[trial.right.first_valid_index()].right if trial.right.first_valid_index() is not None else np.nan for trial in trialListOn] 
        #firstRightListOff = [trial.loc[trial.right.first_valid_index()].right if trial.right.first_valid_index() is not None else np.nan for trial in trialListOff]
        # first right lick per SMOOTH trial
        firstRightListOn  = [trial.loc[trial.right.first_valid_index()].right if (trial.right.first_valid_index() is not None and trial.stim.unique() == 'SMOOTH') else np.nan for trial in trialListOn] 
        firstRightListOff = [trial.loc[trial.right.first_valid_index()].right if (trial.right.first_valid_index() is not None and trial.stim.unique() == 'SMOOTH') else np.nan for trial in trialListOff]         
        # calculate means (ignoring nans) 
        meanLeftOn   = np.nanmean(firstLeftListOn)
        meanRightOn  = np.nanmean(firstRightListOn)
        meanLeftOff  = np.nanmean(firstLeftListOff)
        meanRightOff = np.nanmean(firstRightListOff)
        # calculate difference
        diffOn  = meanLeftOn - meanRightOn
        diffOff = meanLeftOff - meanRightOff
        # group by trials (no laser status consideration)
        trialList = [group[1] for group in tm.groupby('trial')] 
        # first left and right lick per trial (no laser status consideration)
        firstLeftList  = [trial.loc[trial.left.first_valid_index()].left if trial.left.first_valid_index() is not None else np.nan for trial in trialList]
        firstRightList = [trial.loc[trial.right.first_valid_index()].right if trial.right.first_valid_index() is not None else np.nan for trial in trialList]
        # means
        meanLeft  = np.nanmean(firstLeftList)
        meanRight = np.nanmean(firstRightList)
        # difference
        diffs = meanLeft - meanRight
        # add to dictionary
        firstLicksDict[mouse] = [mouse,genotype,meanLeftOn,meanRightOn,meanLeftOff,meanRightOff,diffOn,diffOff,meanLeft,meanRight,diffs]
    cols         = ['mouse','genotype','leftOn','rightOn','leftOff','rightOff','diffOn','diffOff','meanLeft','meanRight','diffs']
    firstLicksDf = pd.DataFrame.from_dict(firstLicksDict,orient='index',columns=cols)
    plot         = FirstLicksHisto(firstLicksDf,beh)
    plot2        = FirstLicksArrows(firstLicksDf,beh)
    plot3        = FirstLicksScatter(firstLicksDf,beh)
    return None

def FirstCongruentLicksScatter(firstLicksDf,beh):
    """Visualize lick latency change between laser on and off trials"""
    genotypesList  = firstLicksDf.genotype.unique()
    onLatencyList  = firstLicksDf.meanOn
    offLatencyList = firstLicksDf.meanOff
    # set order for legend
    orderedGenotypesList = ['Flex-Halo', 'Emx1-Halo', 'Cux2-Halo', 'Nr5a1-Halo', 'Fezf2-Halo', 'Rbp4-Halo']
    # calculate laser on and off latency mean and SEM
    averageOnLatencyDict  = {''.join(group[1].genotype.unique()):group[1].meanOn.mean() for group in firstLicksDf.groupby('genotype')}
    semOnLatencyDict      = {''.join(group[1].genotype.unique()):group[1].meanOn.sem() for group in firstLicksDf.groupby('genotype')}
    averageOffLatencyDict = {''.join(group[1].genotype.unique()):group[1].meanOff.mean() for group in firstLicksDf.groupby('genotype')}
    semOffLatencyDict     = {''.join(group[1].genotype.unique()):group[1].meanOff.sem() for group in firstLicksDf.groupby('genotype')}
    # plot
    fig, ax = plt.subplots()
    cmap    = {'Emx1-Halo':'#0eb554','Nr5a1-Halo':'#F26522','Flex-Halo':'black','Cux2-Halo':'#f0028f','Fezf2-Halo':'#7F3E98','Rbp4-Halo':'#213F9A'}
    colors  = [cmap[transgene] for transgene in genotypesList]
    for i,g in enumerate(genotypesList):
        xi = [offLatencyList[j] for j in range(len(offLatencyList)) if firstLicksDf.genotype[j] == g]
        yi = [onLatencyList[j] for j in range(len(onLatencyList)) if firstLicksDf.genotype[j] == g]
        ax.scatter(xi, yi, c=colors[i], label=g)
    reorderLegend(ax,orderedGenotypesList)
    ax.set_title('Laser-induced Decision Latency ({})'.format(beh))                      
    ax.set_xlim(-1000,1000)  
    ax.set_ylim(-1000,1000) 
    # plot identity line
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])] 
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_yticks(np.linspace(-1000,1000,5))
    ax.set_xticks(np.linspace(-1000,1000,5))
    ax.set_aspect('equal', adjustable='box')
    ax.set_ylabel('Lick Latency (Laser ON) [ms]')
    ax.set_xlabel('Lick Latency (Laser OFF) [ms]')
    #plt.show()
    plt.savefig(os.path.join(os.getcwd(),"lick_latency_scatter_{}.svg".format(beh)))
    # save to csv
    firstLicksDf.to_csv("lick_latency_scatter_{}.csv".format(beh),index=False)
    # plot concatenated scatter
    fig, ax = plt.subplots()
    for i,g in enumerate(genotypesList):
        xi      = averageOffLatencyDict[g]
        yi      = averageOnLatencyDict[g]
        xiError = semOffLatencyDict[g]
        yiError = semOnLatencyDict[g]
        ax.scatter(xi, yi, c=colors[i], label=g)
        ax.errorbar(xi, yi, xerr=xiError, yerr=yiError, fmt='none', c=colors[i])
    reorderLegend(ax,orderedGenotypesList)
    ax.set_title('Laser-induced Decision Latency ({})'.format(beh))                      
    ax.set_xlim(-1000,1000)  
    ax.set_ylim(-1000,1000) 
    # plot identity line
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])] 
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_yticks(np.linspace(-1000,1000,5))
    ax.set_xticks(np.linspace(-1000,1000,5))
    ax.set_aspect('equal', adjustable='box')
    ax.set_ylabel('Lick Latency (Laser ON) [ms]')
    ax.set_xlabel('Lick Latency (Laser OFF) [ms]')
    #plt.show()
    plt.savefig(os.path.join(os.getcwd(),"lick_latency_concat_scatter_{}.svg".format(beh)))

def makeFirstCongruentLickLatencyPlots(tmList,beh,genesDict):
    """Plots average time for first congruent lick on laser ON versus laser OFF trials.
    """
    # clamp to reasonable values (-1000 < lick < 1000)
    tmListClamped = [tm.iloc[np.where((tm.left.notna() & (tm.left > -1000) & (tm.left < 1000)) |
                                          (tm.right.notna() & (tm.right > -1000) & (tm.right < 1000)))] for tm in tmList]
    # determine average first lick for left and right for each mouse
    firstCongruentLicksDict = {}
    for tm in tmListClamped:
        # gather mouse name
        mouse        = ''.join(tm.mouse.unique())
        # gather genotype
        genotype     = genesDict[mouse]
        # identify congruent licks
        decisions    = generate_decision_column(tm)
        tm           = tm.copy().assign(decision=decisions)
        congruents   = np.where((tm.left.notna() & tm.decision.isin(['left'])) | (tm.right.notna() & tm.decision.isin(['right'])),1,0)
        tm           = tm.assign(congruent=congruents)
        # group tm by laser status
        onDf         = tm.groupby('opto').get_group('ON')
        offDf        = tm.groupby('opto').get_group('OFF')
        # split by trial
        trialListOn  = [group[1] for group in onDf.groupby('trial')]  
        trialListOff = [group[1] for group in offDf.groupby('trial')] 
        # first congruent lick per ON trial
        firstCongruentLickOn  = [trial.loc[trial[trial.congruent==True].first_valid_index()].licks if trial[trial.congruent==True].first_valid_index() is not None else np.nan for trial in trialListOn]
        # first congruent lick per OFF trial
        firstCongruentLickOff = [trial.loc[trial[trial.congruent==True].first_valid_index()].licks if trial[trial.congruent==True].first_valid_index() is not None else np.nan for trial in trialListOff]           
        # calculate means (ignoring nans) 
        meanOn  = np.nanmean(firstCongruentLickOn)
        meanOff = np.nanmean(firstCongruentLickOff)
        # add to dictionary
        firstCongruentLicksDict[mouse] = [mouse,genotype,meanOn,meanOff]
    cols                  = ['mouse','genotype','meanOn','meanOff']
    firstCongruentLicksDf = pd.DataFrame.from_dict(firstCongruentLicksDict,orient='index',columns=cols)
    plot                  = FirstCongruentLicksScatter(firstCongruentLicksDf,beh)
    return None

def LicksAfterLaserLine(tmList,beh,genotype,liml=-2000,limr=2000,bins=41):
    """Plot proportion of correct licks after laser-on trial 
    for each genotype as a function of time."""
    proportionsNPlusOneDict = defaultdict(list)
    proportionsOffDict      = defaultdict(list)
    proportionsOnDict       = defaultdict(list)
    plotCentersDict         = defaultdict(list)
    titleStr                = genotype
    titleStr2               = "Post-Laser Correct Licks - {} ({})".format(titleStr,beh)
    # for each mouse
    for tm in tmList:
        # gather mouse name
        mouse = ''.join(tm.mouse.unique())
        # identify correct licks
        corrects = np.where(tm.left.notna() & tm.stim.isin(['ROUGH']) | tm.right.notna() & tm.stim.isin(['SMOOTH']),1,0) 
        tm       = tm.copy().assign(correct=corrects)
        # group tm by laser status and outcome
        onDf        = tm.groupby('opto').get_group('ON')
        onCorrectDf = tm.groupby(['opto','outcome']).get_group(('ON',1.0)) 
        # get array of laser-on trial numbers
        # exclude any laser-on trial indices n where trial n+1 is also laser-on (regardless of correct/incorrect)
        laserOnTrialList = [trial for trial in onCorrectDf.trial.unique() if trial+1 in tm.groupby('opto').get_group('OFF').trial.unique()]
        nPlusOneList     = [trial+1 for trial in laserOnTrialList]
        # get group of laser-off n+1 trials
        nPlusOneDf = tm[tm.trial.isin(nPlusOneList)]  
        # get group of all laser-off trials
        allOffDf = tm.groupby('opto').get_group('OFF')
        # drop all n+1 trials
        allOffDf = allOffDf[~tm.trial.isin(nPlusOneList)]
        # merge left and right lick times
        tempNPlusOneDf    = pd.DataFrame({'left':nPlusOneDf.left, 'right':nPlusOneDf.right})
        tempOffDf         = pd.DataFrame({'left':allOffDf.left, 'right':allOffDf.right})
        tempOnDf          = pd.DataFrame({'left':onDf.left, 'right':onDf.right}) 
        licksNPlusOneList = tempNPlusOneDf.stack().groupby(level=0).sum()
        licksOffList      = tempOffDf.stack().groupby(level=0).sum() 
        licksOnList       = tempOnDf.stack().groupby(level=0).sum()
        nPlusOneDf        = nPlusOneDf.assign(licks=licksNPlusOneList)
        allOffDf          = allOffDf.assign(licks=licksOffList) 
        onDf              = onDf.assign(licks=licksOnList)
        # calculate proportions binned by time for n+1 and non-n+1 trials
        proportionsNPlusOne, plotCentersNPlusOne, errorNPlusOne = bin_by_time_and_binary(nPlusOneDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
        proportionsOff, plotCentersOff, errorOff                = bin_by_time_and_binary(allOffDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
        proportionsOn, plotCentersOn, errorOn                   = bin_by_time_and_binary(onDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
        # average centers of bins to correct for numpy rounding errors
        plotCentersList = [np.mean([p1,p2,p3]) for p1,p2,p3 in zip(plotCentersNPlusOne,plotCentersOff,plotCentersOn)]
        # add to dictionaries
        proportionsNPlusOneDict[mouse] = proportionsNPlusOne.values
        proportionsOffDict[mouse]      = proportionsOff.values
        proportionsOnDict[mouse]       = proportionsOn.values
        plotCentersDict[mouse]         = plotCentersList
    # generate dataframes
    proportionsNPlusOneDf = pd.DataFrame(proportionsNPlusOneDict)
    proportionsOffDf      = pd.DataFrame(proportionsOffDict)
    proportionsOnDf       = pd.DataFrame(proportionsOnDict)
    plotCentersDf         = pd.DataFrame(plotCentersDict)
    # calculate means and errors
    proportionsNPlusOneMeanList = proportionsNPlusOneDf.mean(axis=1).values
    proportionsNPlusOneSEMList  = proportionsNPlusOneDf.sem(axis=1).values
    proportionsOffMeanList      = proportionsOffDf.mean(axis=1).values
    proportionsOffSEMList       = proportionsOffDf.sem(axis=1).values
    proportionsOnMeanList       = proportionsOnDf.mean(axis=1).values
    proportionsOnSEMList        = proportionsOnDf.sem(axis=1).values
    # average centers of bins again to correct for numpy rounding errors
    plotCentersMeanList = plotCentersDf.mean(axis=1).values
    # plot
    fig, ax = plt.subplots() 
    ax.plot(plotCentersMeanList,proportionsOffMeanList,linestyle='-',color='black',label='Laser OFF') 
    ax.plot(plotCentersMeanList,proportionsOnMeanList,linestyle='-',color='#ffb82b',label='Laser ON')
    ax.plot(plotCentersMeanList,proportionsNPlusOneMeanList,linestyle='-',color='#0eb554',label='Post-Laser ON') 
    legend_without_duplicate_labels(ax)
    ax.autoscale()
    ax.set_ylim(bottom=0,top=1.05)
    ax.get_yaxis().set_ticks(np.linspace(0.25,1.0,4))
    ax.set_xlim(left=liml,right=limr)
    simpleaxis(ax)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.fill_between(plotCentersMeanList, proportionsNPlusOneMeanList-proportionsNPlusOneSEMList, proportionsNPlusOneMeanList+proportionsNPlusOneSEMList, alpha=0.3, edgecolor='none', facecolor='#0eb554')
    plt.fill_between(plotCentersMeanList, proportionsOffMeanList-proportionsOffSEMList, proportionsOffMeanList+proportionsOffSEMList, alpha=0.6, edgecolor='none', facecolor='silver')
    plt.fill_between(plotCentersMeanList, proportionsOnMeanList-proportionsOnSEMList, proportionsOnMeanList+proportionsOnSEMList, alpha=0.3, edgecolor='none', facecolor='#ffb82b')
    plt.xlabel('Time [ms]')
    plt.ylabel('Lick Probability')
    #plt.show()
    plt.title("{}".format(titleStr2))
    plt.savefig(os.path.join(os.getcwd(),"{}_post_laser_correct_{}.svg".format(titleStr,beh)))
    plt.close('all')
    return None

def LicksAfterLaserScatter(tmList,beh,genotype):
    """Plot proportion of correct licks after for laser ON, OFF, and n+1 trials
    for each genotype as a scatter with axes proportion ROUGH and SMOOTH."""
    proportionsRoughOnList        = []
    proportionsRoughNPlusOneList  = []
    proportionsRoughOffList       = []
    proportionsSmoothOnList       = []
    proportionsSmoothNPlusOneList = []
    proportionsSmoothOffList      = []
    # title
    titleStr  = genotype
    titleStr2 = "Post-Laser Licks - {} ({})".format(titleStr,beh)
    # for each mouse
    for tm in tmList:
        # gather mouse name
        mouse = ''.join(tm.mouse.unique())
        # identify correct licks
        corrects = np.where(tm.left.notna() & tm.stim.isin(['ROUGH']) | tm.right.notna() & tm.stim.isin(['SMOOTH']),1,0) 
        tm       = tm.copy().assign(correct=corrects)
        # group tm by laser status and outcome
        onDf        = tm.groupby('opto').get_group('ON')
        onCorrectDf = tm.groupby(['opto','outcome']).get_group(('ON',1.0)) 
        # get array of laser-on trial numbers
        # exclude any laser-on trial indices n where trial n+1 is also laser-on (regardless of correct/incorrect)
        laserOnTrialList = [trial for trial in onCorrectDf.trial.unique() if trial+1 in tm.groupby('opto').get_group('OFF').trial.unique()]
        nPlusOneList     = [trial+1 for trial in laserOnTrialList]
        # get group of laser-off n+1 trials
        nPlusOneDf = tm[tm.trial.isin(nPlusOneList)]  
        # get group of all laser-off trials
        allOffDf = tm.groupby('opto').get_group('OFF')
        # drop all n+1 trials
        allOffDf = allOffDf[~tm.trial.isin(nPlusOneList)]
        # group by stimulus
        roughOnDf        = onDf.groupby('stim').get_group('ROUGH')
        roughNPlusOneDf  = nPlusOneDf.groupby('stim').get_group('ROUGH')
        roughOffDf       = allOffDf.groupby('stim').get_group('ROUGH')
        smoothOnDf       = onDf.groupby('stim').get_group('SMOOTH')
        smoothNPlusOneDf = nPlusOneDf.groupby('stim').get_group('SMOOTH')
        smoothOffDf      = allOffDf.groupby('stim').get_group('SMOOTH')
        # one result per trial
        roughOnDf        = roughOnDf.drop_duplicates('trial')
        roughNPlusOneDf  = roughNPlusOneDf.drop_duplicates('trial')
        roughOffDf       = roughOffDf.drop_duplicates('trial')
        smoothOnDf       = smoothOnDf.drop_duplicates('trial')
        smoothNPlusOneDf = smoothNPlusOneDf.drop_duplicates('trial')
        smoothOffDf      = smoothOffDf.drop_duplicates('trial')
        # add to dictionaries
        proportionsRoughOnList.append(roughOnDf.outcome.mean())
        proportionsRoughNPlusOneList.append(roughNPlusOneDf.outcome.mean())
        proportionsRoughOffList.append(roughOffDf.outcome.mean())
        proportionsSmoothOnList.append(smoothOnDf.outcome.mean())
        proportionsSmoothNPlusOneList.append(smoothNPlusOneDf.outcome.mean())
        proportionsSmoothOffList.append(smoothOffDf.outcome.mean())
    # plot
    fig, ax = plt.subplots() 
    ax.scatter(proportionsSmoothOffList,proportionsRoughOffList,c='black',label='Laser OFF')
    ax.scatter(proportionsSmoothOnList,proportionsRoughOnList,c='#ffb82b',label='Laser ON')
    ax.scatter(proportionsSmoothNPlusOneList,proportionsRoughNPlusOneList,c='#0eb554',label='Post-Laser ON')
    reorderLegend(ax,['Laser OFF','Laser ON','Post-Laser ON'])
    ax.autoscale()
    ax.set_ylim(bottom=0,top=1.0)
    ax.get_yaxis().set_ticks(np.linspace(0.,1.0,5))
    ax.set_xlim(left=0,right=1.0)
    ax.get_xaxis().set_ticks(np.linspace(0.,1.0,5))
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])] 
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel('Reward Probability (SMOOTH)')
    plt.ylabel('Reward Probability (ROUGH)')
    plt.title("{}".format(titleStr2))
    #plt.show()
    plt.savefig(os.path.join(os.getcwd(),"{}_post_laser_correct_scatter_{}.svg".format(titleStr,beh)))
    plt.close('all')
    return None

def makeLicksAfterLaserPlots(tmsDict,beh):
    """Call LicksAfterLaserLine and Scatter function for each genotype"""
    genotypes = list(tmsDict.keys())
    for g in genotypes:
        plots = LicksAfterLaserLine(tmsDict[g],beh,g,bins=41)
        plots = LicksAfterLaserScatter(tmsDict[g],beh,g)
    return None

def LickFrequencyLineByStim(tmList,beh,genotype,liml=-2000,limr=2000,bins=41):
    """Plots frequency of licks as function of time by stimulus"""
    freqOnRoughDict   = defaultdict(list)
    freqOnSmoothDict  = defaultdict(list)
    freqOffRoughDict  = defaultdict(list)
    freqOffSmoothDict = defaultdict(list)
    plotCentersDict   = defaultdict(list)
    titleStr          = genotype
    titleStr2         = "Lick Frequency (by stim) - {} ({})".format(titleStr,beh)
    # for each mouse
    for tm in tmList:
        # gather mouse name
        mouse = ''.join(tm.mouse.unique())
        # group tm by laser status
        onDf  = tm.groupby('opto').get_group('ON')
        offDf = tm.groupby('opto').get_group('OFF')
        # grup by stimulus
        onRoughDf   = onDf.groupby('stim').get_group('ROUGH')
        onSmoothDf  = onDf.groupby('stim').get_group('SMOOTH')
        offRoughDf  = offDf.groupby('stim').get_group('ROUGH')
        offSmoothDf = offDf.groupby('stim').get_group('SMOOTH')
        # merge left and right lick times
        tempOnRoughDf      = pd.DataFrame({'left':onRoughDf.left, 'right':onRoughDf.right})  
        tempOnSmoothDf     = pd.DataFrame({'left':onSmoothDf.left, 'right':onSmoothDf.right})  
        tempOffRoughDf     = pd.DataFrame({'left':offRoughDf.left, 'right':offRoughDf.right})  
        tempOffSmoothDf    = pd.DataFrame({'left':offSmoothDf.left, 'right':offSmoothDf.right})
        licksOnRoughList   = tempOnRoughDf.stack().groupby(level=0).sum()
        licksOnSmoothList  = tempOnSmoothDf.stack().groupby(level=0).sum()
        licksOffRoughList  = tempOffRoughDf.stack().groupby(level=0).sum()
        licksOffSmoothList = tempOffSmoothDf.stack().groupby(level=0).sum()
        onRoughDf          = onRoughDf.assign(licks=licksOnRoughList)
        onSmoothDf         = onSmoothDf.assign(licks=licksOnSmoothList)
        offRoughDf         = offRoughDf.assign(licks=licksOffRoughList)
        offSmoothDf        = offSmoothDf.assign(licks=licksOffSmoothList)  
        # calculate lick counts binned by time for laser on and off for each stimulus
        binsList           = np.linspace(liml,limr,bins)
        onRoughLicksList   = onRoughDf.groupby(pd.cut(onRoughDf.licks,binsList,include_lowest=True)).licks.agg(['count'])
        onSmoothLicksList  = onSmoothDf.groupby(pd.cut(onSmoothDf.licks,binsList,include_lowest=True)).licks.agg(['count'])
        offRoughLicksList  = offRoughDf.groupby(pd.cut(offRoughDf.licks,binsList,include_lowest=True)).licks.agg(['count'])
        offSmoothLicksList = offSmoothDf.groupby(pd.cut(offSmoothDf.licks,binsList,include_lowest=True)).licks.agg(['count'])
        # calculate lick frequency binned by time for laser on and off for each stimulus
        # calculate bin size (ms)
        binSize = (limr - liml) / (bins - 1)
        # convert to frequency in Hz (licks per second) and average across n trials
        freqOnRoughList   = (onRoughLicksList['count'].values / binSize)   * (1000 / 1) / (len(onRoughDf.trial.unique()))
        freqOnSmoothList  = (onSmoothLicksList['count'].values / binSize)  * (1000 / 1) / (len(onSmoothDf.trial.unique()))
        freqOffRoughList  = (offRoughLicksList['count'].values / binSize)  * (1000 / 1) / (len(offRoughDf.trial.unique()))
        freqOffSmoothList = (offSmoothLicksList['count'].values / binSize) * (1000 / 1) / (len(offSmoothDf.trial.unique()))
        # calculate plot centers
        plotCentersList = [interval.mid for interval in onRoughLicksList.index]
        # add to dictionaries
        freqOnRoughDict[mouse]   = freqOnRoughList
        freqOnSmoothDict[mouse]  = freqOnSmoothList
        freqOffRoughDict[mouse]  = freqOffRoughList
        freqOffSmoothDict[mouse] = freqOffSmoothList
        plotCentersDict[mouse]   = plotCentersList
    # generate dataframes
    freqOnRoughDf   = pd.DataFrame(freqOnRoughDict)
    freqOnSmoothDf  = pd.DataFrame(freqOnSmoothDict)
    freqOffRoughDf  = pd.DataFrame(freqOffRoughDict)
    freqOffSmoothDf = pd.DataFrame(freqOffSmoothDict)
    plotCentersDf   = pd.DataFrame(plotCentersDict)
    # calculate means and errors
    freqOnRoughMeanList   = freqOnRoughDf.mean(axis=1).values
    freqOnRoughSEMList    = freqOnRoughDf.sem(axis=1).values
    freqOnSmoothMeanList  = freqOnSmoothDf.mean(axis=1).values
    freqOnSmoothSEMList   = freqOnSmoothDf.sem(axis=1).values
    freqOffRoughMeanList  = freqOffRoughDf.mean(axis=1).values
    freqOffRoughSEMList   = freqOffRoughDf.sem(axis=1).values
    freqOffSmoothMeanList = freqOffSmoothDf.mean(axis=1).values
    freqOffSmoothSEMList  = freqOffSmoothDf.sem(axis=1).values
    # average centers of bins again to correct for numpy rounding errors
    plotCentersMeanList = plotCentersDf.mean(axis=1).values
    # plot
    fig, ax = plt.subplots()
    ax.plot(plotCentersMeanList,freqOnRoughMeanList,linestyle='--',color='#ffb82b',label='Laser ON (rough)') 
    ax.plot(plotCentersMeanList,freqOnSmoothMeanList,linestyle='-',color='#ffb82b',label='Laser ON (smooth)') 
    ax.plot(plotCentersMeanList,freqOffRoughMeanList,linestyle='--',color='black',label='Laser OFF (rough)')  
    ax.plot(plotCentersMeanList,freqOffSmoothMeanList,linestyle='-',color='black',label='Laser OFF (smooth)') 
    legend_without_duplicate_labels(ax,pos='upper right')
    ax.autoscale()
    ax.set_ylim(bottom=0,top=8)
    ax.get_yaxis().set_ticks(np.linspace(2,8,4))
    ax.set_xlim(left=liml,right=limr)
    simpleaxis(ax)
    plt.fill_between(plotCentersMeanList, freqOnRoughMeanList-freqOnRoughSEMList, freqOnRoughMeanList+freqOnRoughSEMList, alpha=0.3, edgecolor='none', facecolor='#ffb82b')
    plt.fill_between(plotCentersMeanList, freqOnSmoothMeanList-freqOnSmoothSEMList, freqOnSmoothMeanList+freqOnSmoothSEMList, alpha=0.3, edgecolor='none', facecolor='#ffb82b')
    plt.fill_between(plotCentersMeanList, freqOffRoughMeanList-freqOffRoughSEMList, freqOffRoughMeanList+freqOffRoughSEMList, alpha=0.6, edgecolor='none', facecolor='silver')
    plt.fill_between(plotCentersMeanList, freqOffSmoothMeanList-freqOffSmoothSEMList, freqOffSmoothMeanList+freqOffSmoothSEMList, alpha=0.6, edgecolor='none', facecolor='silver')
    plt.xlabel('Time [ms]')
    plt.ylabel('Lick Frequency [Hz]')
    plt.title("{}".format(titleStr2))
    #plt.show()
    plt.savefig(os.path.join(os.getcwd(),"{}_concat_freq_STIM_{}.svg".format(titleStr,beh)))
    plt.close('all')
    return None

def LickIntervalHisto(tmList,beh,genotype):
    """Plots inter-lick interval as histogram for each mouse"""
    # for each mouse
    for tm in tmList:
        # gather mouse name
        mouse    = ''.join(tm.mouse.unique())
        titleStr = "Inter-lick interval - {} ({})".format(mouse,beh)
        # merge left and right lick times
        tempDf    = pd.DataFrame({'left':tm.left, 'right':tm.right})  
        licksList = tempDf.stack().groupby(level=0).sum()
        allDf     = tm.assign(licks=licksList)
        # calculate inter-lick intervals
        ILIs = allDf.licks.groupby(allDf.trial).diff().abs()
        # plot
        bins = np.linspace(0,250,51)
        fig, ax = plt.subplots()
        ax.hist(ILIs, bins, color='red', alpha=0.7)
        ax.set_ylabel('Licks')
        ax.set_xlabel('Inter-lick interval [ms]')
        plt.title("{}".format(titleStr))
        #plt.show()
        plt.savefig(os.path.join(os.getcwd(),"{}_ILI.svg".format(mouse)))
        plt.close('all')
    return None

def makeLickFrequencyPlots(tmsDict,beh):
    """Call LickFrequencyLine an Histo function for each genotype"""
    genotypes = list(tmsDict.keys())
    for g in genotypes:
        plots = LickFrequencyLineByStim(tmsDict[g],beh,g,bins=21)
        plots = LickIntervalHisto(tmsDict[g],beh,g)
    return None

def FreqLineByStimLikeOpto(tmsDict,beh,liml=-2000,limr=2000,bins=41):
    """Plots frequency of licks with respect to stimulus as function of time.
    Plots like opto, with averaged days -2 and -1 for 'laser off' (pre) and day 1 as 'laser on' (post).
    Used for lesion data."""
    freqPreRoughDict   = defaultdict(list)
    freqPreSmoothDict  = defaultdict(list)
    freqPostRoughDict  = defaultdict(list)
    freqPostSmoothDict = defaultdict(list)
    plotCentersDict    = defaultdict(list)
    titleStr           = 'lesion'
    titleStr2          = "Lick Frequency (by stim) - {} ({})".format(titleStr,beh)
    # for each mouse
    for preTM,postTM in zip(tmsDict['pre'],tmsDict['post']):
        # gather mouse name
        mouse = ''.join(preTM.mouse.unique())
        # grup by stimulus
        preRoughDf   = preTM.groupby('stim').get_group('ROUGH')
        preSmoothDf  = preTM.groupby('stim').get_group('SMOOTH')
        postRoughDf  = postTM.groupby('stim').get_group('ROUGH')
        postSmoothDf = postTM.groupby('stim').get_group('SMOOTH')
        # merge left and right lick times
        tempPreRoughDf      = pd.DataFrame({'left':preRoughDf.left, 'right':preRoughDf.right})  
        tempPreSmoothDf     = pd.DataFrame({'left':preSmoothDf.left, 'right':preSmoothDf.right})  
        tempPostRoughDf     = pd.DataFrame({'left':postRoughDf.left, 'right':postRoughDf.right})  
        tempPostSmoothDf    = pd.DataFrame({'left':postSmoothDf.left, 'right':postSmoothDf.right})
        licksPreRoughList   = tempPreRoughDf.stack().groupby(level=0).sum()
        licksPreSmoothList  = tempPreSmoothDf.stack().groupby(level=0).sum()
        licksPostRoughList  = tempPostRoughDf.stack().groupby(level=0).sum()
        licksPostSmoothList = tempPostSmoothDf.stack().groupby(level=0).sum()
        preRoughDf          = preRoughDf.assign(licks=licksPreRoughList)
        preSmoothDf         = preSmoothDf.assign(licks=licksPreSmoothList)
        postRoughDf         = postRoughDf.assign(licks=licksPostRoughList)
        postSmoothDf        = postSmoothDf.assign(licks=licksPostSmoothList)  
        # calculate lick counts binned by time for pre and post lesion for each stimulus
        binsList            = np.linspace(liml,limr,bins)
        preRoughLicksList   = preRoughDf.groupby(pd.cut(preRoughDf.licks,binsList,include_lowest=True)).licks.agg(['count'])
        preSmoothLicksList  = preSmoothDf.groupby(pd.cut(preSmoothDf.licks,binsList,include_lowest=True)).licks.agg(['count'])
        postRoughLicksList  = postRoughDf.groupby(pd.cut(postRoughDf.licks,binsList,include_lowest=True)).licks.agg(['count'])
        postSmoothLicksList = postSmoothDf.groupby(pd.cut(postSmoothDf.licks,binsList,include_lowest=True)).licks.agg(['count'])
        # calculate lick frequency binned by time for laser on and off for each stimulus
        # calculate bin size (ms)
        binSize = (limr - liml) / (bins - 1)
        # convert to frequency in Hz (licks per second) and average across n trials
        freqPreRough   = (preRoughLicksList['count'].values / binSize)   * (1000 / 1) / (len(preRoughDf.trial.unique()))
        freqPreSmooth  = (preSmoothLicksList['count'].values / binSize)  * (1000 / 1) / (len(preSmoothDf.trial.unique()))
        freqPostRough  = (postRoughLicksList['count'].values / binSize)  * (1000 / 1) / (len(postRoughDf.trial.unique()))
        freqPostSmooth = (postSmoothLicksList['count'].values / binSize) * (1000 / 1) / (len(postSmoothDf.trial.unique()))
        # calculate plot centers
        plotCentersList = [interval.mid for interval in preRoughLicksList.index]
        # add to dictionaries
        freqPreRoughDict[mouse]   = freqPreRough
        freqPreSmoothDict[mouse]  = freqPreSmooth
        freqPostRoughDict[mouse]  = freqPostRough
        freqPostSmoothDict[mouse] = freqPostSmooth
        plotCentersDict[mouse]    = plotCentersList
    # generate dataframes
    freqPreRoughDf   = pd.DataFrame(freqPreRoughDict)
    freqPreSmoothDf  = pd.DataFrame(freqPreSmoothDict)
    freqPostRoughDf  = pd.DataFrame(freqPostRoughDict)
    freqPostSmoothDf = pd.DataFrame(freqPostSmoothDict)
    plotCentersDf    = pd.DataFrame(plotCentersDict)
    # calculate means and errors
    freqPreRoughMeanList   = freqPreRoughDf.mean(axis=1).values
    freqPreRoughSEMList    = freqPreRoughDf.sem(axis=1).values
    freqPreSmoothMeanList  = freqPreSmoothDf.mean(axis=1).values
    freqPreSmoothSEMList   = freqPreSmoothDf.sem(axis=1).values
    freqPostRoughMeanList  = freqPostRoughDf.mean(axis=1).values
    freqPostRoughSEMList   = freqPostRoughDf.sem(axis=1).values
    freqPostSmoothMeanList = freqPostSmoothDf.mean(axis=1).values
    freqPostSmoothSEMList  = freqPostSmoothDf.sem(axis=1).values
    # average centers of bins again to correct for numpy rounding errors
    plotCentersMeanList = plotCentersDf.mean(axis=1).values
    # plot
    fig, ax = plt.subplots()
    ax.plot(plotCentersMeanList,freqPreRoughMeanList,linestyle='--',color='black',label='Pre-lesion (rough)') 
    ax.plot(plotCentersMeanList,freqPreSmoothMeanList,linestyle='-',color='black',label='Pre-lesion (smooth)') 
    ax.plot(plotCentersMeanList,freqPostRoughMeanList,linestyle='--',color='#ffb82b',label='Post-lesion (rough)')  
    ax.plot(plotCentersMeanList,freqPostSmoothMeanList,linestyle='-',color='#ffb82b',label='Post-lesion (smooth)') 
    legend_without_duplicate_labels(ax,pos='upper right')
    ax.autoscale()
    ax.set_ylim(bottom=0,top=8)
    ax.get_yaxis().set_ticks(np.linspace(2,8,4))
    ax.set_xlim(left=liml,right=limr)
    simpleaxis(ax)
    plt.fill_between(plotCentersMeanList, freqPreRoughMeanList-freqPreRoughSEMList, freqPreRoughMeanList+freqPreRoughSEMList, alpha=0.6, edgecolor='none', facecolor='silver')
    plt.fill_between(plotCentersMeanList, freqPreSmoothMeanList-freqPreSmoothSEMList, freqPreSmoothMeanList+freqPreSmoothSEMList, alpha=0.6, edgecolor='none', facecolor='silver')
    plt.fill_between(plotCentersMeanList, freqPostRoughMeanList-freqPostRoughSEMList, freqPostRoughMeanList+freqPostRoughSEMList, alpha=0.3, edgecolor='none', facecolor='#ffb82b')
    plt.fill_between(plotCentersMeanList, freqPostSmoothMeanList-freqPostSmoothSEMList, freqPostSmoothMeanList+freqPostSmoothSEMList, alpha=0.3, edgecolor='none', facecolor='#ffb82b')
    plt.xlabel('Time [ms]')
    plt.ylabel('Lick Frequency [Hz]')
    plt.title("{}".format(titleStr2))
    #plt.show()
    plt.savefig(os.path.join(os.getcwd(),"like_opto_{}_concat_freq_STIM_{}.svg".format(titleStr,beh)))
    plt.close('all')
    return None

def makeFreqLineLikeOpto(tmsDict,beh,liml=-2000,limr=2000):
    """Call FreqLineByStimLikeOpto function for each day with adjusted statistics.
    Used for lesion data."""
    plots = FreqLineByStimLikeOpto(tmsDict,beh,bins=21)
    return None

def NaiveExpertFrequencyByStim(tmList,status,liml=-2000,limr=2000,bins=41):
    """Plots frequency of licks as function of time by stimulus.
    Used for comparing naive and expert licking data in a non-genotype specific manner."""
    freqSmoothDict  = defaultdict(list)
    freqRoughDict   = defaultdict(list)
    plotCentersDict = defaultdict(list)
    titleStr        = "JP"
    titleStr2       = "Lick Frequency (by stim) - {}".format(status)
    # for each mouse
    for tm in tmList:
        # gather mouse name
        mouse = ''.join(tm.mouse.unique())
        # grup by stimulus
        smoothDf = tm.groupby('stim').get_group('SMOOTH')
        roughDf  = tm.groupby('stim').get_group('ROUGH')
        # merge left and right lick times
        tempSmoothDf    = pd.DataFrame({'left':smoothDf.left, 'right':smoothDf.right})  
        tempRoughDf     = pd.DataFrame({'left':roughDf.left, 'right':roughDf.right})  
        licksSmoothList = tempSmoothDf.stack().groupby(level=0).sum()
        licksRoughList  = tempRoughDf.stack().groupby(level=0).sum()
        smoothDf        = smoothDf.assign(licks=licksSmoothList)
        roughDf         = roughDf.assign(licks=licksRoughList)
        # calculate lick counts binned by time for SRGAP2C and WT for each stimulus
        binsList        = np.linspace(liml,limr,bins)
        smoothLicksList = smoothDf.groupby(pd.cut(smoothDf.licks,binsList,include_lowest=True)).licks.agg(['count'])
        roughLicksList  = roughDf.groupby(pd.cut(roughDf.licks,binsList,include_lowest=True)).licks.agg(['count'])
        # calculate lick frequency binned by time for SRGAP2C and WT for each stimulus
        # calculate bin size (ms)
        binSize = (limr - liml) / (bins - 1)
        # convert to frequency in Hz (licks per second) and average across n trials
        freqSmoothList = (smoothLicksList['count'].values / binSize)   * (1000 / 1) / (len(smoothDf.trial.unique()))
        freqRoughList  = (roughLicksList['count'].values / binSize)  * (1000 / 1) / (len(roughDf.trial.unique()))
        # calculate plot centers
        plotCentersList = [interval.mid for interval in smoothLicksList.index]
        # add to dictionaries
        freqSmoothDict[mouse]  = freqSmoothList
        freqRoughDict[mouse]   = freqRoughList
        plotCentersDict[mouse] = plotCentersList
    # generate dataframes
    freqSmoothDf  = pd.DataFrame(freqSmoothDict)
    freqRoughDf   = pd.DataFrame(freqRoughDict)
    plotCentersDf = pd.DataFrame(plotCentersDict)
    # calculate means and errors
    freqSmoothMeanList = freqSmoothDf.mean(axis=1).values
    freqSmoothSEMList  = freqSmoothDf.sem(axis=1).values
    freqRoughMeanList  = freqRoughDf.mean(axis=1).values
    freqRoughSEMList   = freqRoughDf.sem(axis=1).values
    # average centers of bins again to correct for numpy rounding errors
    plotCentersMeanList = plotCentersDf.mean(axis=1).values
    # plot
    fig, ax = plt.subplots()
    ax.plot(plotCentersMeanList,freqSmoothMeanList,linestyle='-',color='black',label='Smooth')  
    ax.plot(plotCentersMeanList,freqRoughMeanList,linestyle='--',color='black',label='Rough') 
    legend_without_duplicate_labels(ax,pos='upper right')
    ax.autoscale()
    ax.set_ylim(bottom=0,top=8)
    ax.get_yaxis().set_ticks(np.linspace(2,8,4))
    ax.set_xlim(left=liml,right=limr)
    simpleaxis(ax)
    plt.fill_between(plotCentersMeanList, freqSmoothMeanList-freqSmoothSEMList, freqSmoothMeanList+freqSmoothSEMList, alpha=0.6, edgecolor='none', facecolor='silver')
    plt.fill_between(plotCentersMeanList, freqRoughMeanList-freqRoughSEMList, freqRoughMeanList+freqRoughSEMList, alpha=0.6, edgecolor='none', facecolor='silver')
    plt.xlabel('Time [ms]')
    plt.ylabel('Lick Frequency [Hz]')
    plt.title("{}".format(titleStr2))
    #plt.show()
    plt.savefig(os.path.join(os.getcwd(),"{}_freq_STIM_{}.svg".format(titleStr,status)))
    plt.close('all')
    return None

def makeNaiveExpertFrequencyPlots(tmsDict,status):
    """Call NaiveExpertFrequencyByStim function for each learning status.
    Used for comparing naive and expert licking data in a non-genotype specific manner."""
    plots = [NaiveExpertFrequencyByStim(tmsDict[s],s,bins=21) for s in tmsDict.keys()]
    return None

def NaiveExpertCorrectByStim(tmList,status,liml=-2000,limr=2000,bins=41):
    """Plots proportion of correct licks as function of time by stimulus.
    Used for comparing naive and expert licking data in a non-genotype specific manner."""
    proportionsSmoothDict  = defaultdict(list)
    proportionsRoughDict   = defaultdict(list)
    plotCentersDict        = defaultdict(list)
    titleStr               = "JP"
    titleStr2              = "Correct Licks (by stim) - {}".format(status)
    # for each mouse
    for tm in tmList:
        # gather mouse name
        mouse = ''.join(tm.mouse.unique())
        # identify correct licks
        correctLicks = np.where((tm.left.notna() & tm.stim.isin(['ROUGH'])) | (tm.right.notna() & tm.stim.isin(['SMOOTH'])),1,0) 
        tm           = tm.copy().assign(correct=correctLicks)
        # grup by stimulus
        smoothDf = tm.groupby('stim').get_group('SMOOTH')
        roughDf  = tm.groupby('stim').get_group('ROUGH')
        # merge left and right lick times
        tempSmoothDf    = pd.DataFrame({'left':smoothDf.left, 'right':smoothDf.right})  
        tempRoughDf     = pd.DataFrame({'left':roughDf.left, 'right':roughDf.right})  
        licksSmoothList = tempSmoothDf.stack().groupby(level=0).sum()
        licksRoughList  = tempRoughDf.stack().groupby(level=0).sum()
        smoothDf        = smoothDf.assign(licks=licksSmoothList)
        roughDf         = roughDf.assign(licks=licksRoughList)
        # calculate proportions binned by time for each stimulus
        proportionsSmooth, plotCentersSmooth, errorSmooth = bin_by_time_and_binary(smoothDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
        proportionsRough, plotCentersRough, errorRough    = bin_by_time_and_binary(roughDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
        # average centers of bins to correct for numpy rounding errors
        plotCentersList = [np.mean([p1,p2]) for p1,p2 in zip(plotCentersSmooth,plotCentersRough)]
        # add to dictionaries
        proportionsSmoothDict[mouse]  = proportionsSmooth
        proportionsRoughDict[mouse]   = proportionsRough
        plotCentersDict[mouse]        = plotCentersList
    # generate dataframes
    proportionsSmoothDf = pd.DataFrame(proportionsSmoothDict)
    proportionsRoughDf  = pd.DataFrame(proportionsRoughDict)
    plotCentersDf       = pd.DataFrame(plotCentersDict)
    # calculate means and errors
    proportionsSmoothMeanList = proportionsSmoothDf.mean(axis=1).values
    proportionsSmoothSEMList  = proportionsSmoothDf.sem(axis=1).values
    proportionsRoughMeanList  = proportionsRoughDf.mean(axis=1).values
    proportionsRoughSEMList   = proportionsRoughDf.sem(axis=1).values
    # average centers of bins again to correct for numpy rounding errors
    plotCentersMeanList = plotCentersDf.mean(axis=1).values
    # plot
    fig, ax = plt.subplots()
    ax.plot(plotCentersMeanList,proportionsSmoothMeanList,linestyle='-',color='black',label='Smooth')  
    ax.plot(plotCentersMeanList,proportionsRoughMeanList,linestyle='--',color='black',label='Rough') 
    legend_without_duplicate_labels(ax)
    ax.autoscale()
    ax.set_ylim(bottom=0,top=1.05)
    ax.get_yaxis().set_ticks(np.linspace(0.25,1.0,4))
    ax.set_xlim(left=liml,right=limr)
    simpleaxis(ax)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.fill_between(plotCentersMeanList, proportionsSmoothMeanList-proportionsSmoothSEMList, proportionsSmoothMeanList+proportionsSmoothSEMList, alpha=0.6, edgecolor='none', facecolor='silver')
    plt.fill_between(plotCentersMeanList, proportionsRoughMeanList-proportionsRoughSEMList, proportionsRoughMeanList+proportionsRoughSEMList, alpha=0.6, edgecolor='none', facecolor='silver')
    plt.xlabel('Time [ms]')
    plt.ylabel('Lick Probability')
    plt.title("{}".format(titleStr2))
    #plt.show()
    plt.savefig(os.path.join(os.getcwd(),"{}_concat_correct_STIM_{}.svg".format(titleStr,status)))
    plt.close('all')
    return None

def makeNaiveExpertCorrectPlots(tmsDict,status):
    """Call NaiveExpertCorrectByStim function for each learning status.
    Used for comparing naive and expert licking data in a non-genotype specific manner."""
    plots = [NaiveExpertCorrectByStim(tmsDict[s],s,bins=21) for s in tmsDict.keys()]
    return None

def LickFrequencyLineByStim2C(tmsDict,status,liml=-2000,limr=2000,bins=41):
    """Plots frequency of licks as function of time by stimulus.
    Used for SRGAP2C data."""
    freqSRGAP2CRoughDict   = defaultdict(list)
    freqSRGAP2CRougherDict = defaultdict(list)
    freqWTRoughDict        = defaultdict(list)
    freqWTRougherDict      = defaultdict(list)
    plotCentersDict        = defaultdict(list)
    titleStr               = 'SRGAP2C'
    titleStr2              = "Lick Frequency (by stim) - {} ({})".format(titleStr,status)
    # for each mouse
    # cycle the smaller class
    # wasteful computationally, but will simply overwrite previous values with no consequences
    if len(tmsDict['WT']) <= len(tmsDict['SRGAP2C']):
        for WTDf, SRGAP2CDf in zip(cycle(tmsDict['WT']),tmsDict['SRGAP2C']):
            # gather mouse names
            mouseWT      = ''.join(WTDf.mouse.unique())
            mouseSRGAP2C = ''.join(SRGAP2CDf.mouse.unique())
            # grup by stimulus
            SRGAP2CRoughDf   = SRGAP2CDf.groupby('stim').get_group('SMOOTH')
            SRGAP2CRougherDf = SRGAP2CDf.groupby('stim').get_group('ROUGH')
            WTRoughDf        = WTDf.groupby('stim').get_group('SMOOTH')
            WTRougherDf      = WTDf.groupby('stim').get_group('ROUGH')
            # merge left and right lick times
            tempSRGAP2CRoughDf      = pd.DataFrame({'left':SRGAP2CRoughDf.left, 'right':SRGAP2CRoughDf.right})  
            tempSRGAP2CRougherDf    = pd.DataFrame({'left':SRGAP2CRougherDf.left, 'right':SRGAP2CRougherDf.right})  
            tempWTRoughDf           = pd.DataFrame({'left':WTRoughDf.left, 'right':WTRoughDf.right})  
            tempWTRougherDf         = pd.DataFrame({'left':WTRougherDf.left, 'right':WTRougherDf.right})
            licksSRGAP2CRoughList   = tempSRGAP2CRoughDf.stack().groupby(level=0).sum()
            licksSRGAP2CRougherList = tempSRGAP2CRougherDf.stack().groupby(level=0).sum()
            licksWTRoughList        = tempWTRoughDf.stack().groupby(level=0).sum()
            licksWTRougherList      = tempWTRougherDf.stack().groupby(level=0).sum()
            SRGAP2CRoughDf          = SRGAP2CRoughDf.assign(licks=licksSRGAP2CRoughList)
            SRGAP2CRougherDf        = SRGAP2CRougherDf.assign(licks=licksSRGAP2CRougherList)
            WTRoughDf               = WTRoughDf.assign(licks=licksWTRoughList)
            WTRougherDf             = WTRougherDf.assign(licks=licksWTRougherList)  
            # calculate lick counts binned by time for SRGAP2C and WT for each stimulus
            binsList                = np.linspace(liml,limr,bins)
            SRGAP2CRoughLicksList   = SRGAP2CRoughDf.groupby(pd.cut(SRGAP2CRoughDf.licks,binsList,include_lowest=True)).licks.agg(['count'])
            SRGAP2CRougherLicksList = SRGAP2CRougherDf.groupby(pd.cut(SRGAP2CRougherDf.licks,binsList,include_lowest=True)).licks.agg(['count'])
            WTRoughLicksList        = WTRoughDf.groupby(pd.cut(WTRoughDf.licks,binsList,include_lowest=True)).licks.agg(['count'])
            WTRougherLicksList      = WTRougherDf.groupby(pd.cut(WTRougherDf.licks,binsList,include_lowest=True)).licks.agg(['count'])
            # calculate lick frequency binned by time for SRGAP2C and WT for each stimulus
            # calculate bin size (ms)
            binSize = (limr - liml) / (bins - 1)
            # convert to frequency in Hz (licks per second) and average across n trials
            freqSRGAP2CRoughList   = (SRGAP2CRoughLicksList['count'].values / binSize)   * (1000 / 1) / (len(SRGAP2CRoughDf.trial.unique()))
            freqSRGAP2CRougherList = (SRGAP2CRougherLicksList['count'].values / binSize)  * (1000 / 1) / (len(SRGAP2CRougherDf.trial.unique()))
            freqWTRoughList        = (WTRoughLicksList['count'].values / binSize)  * (1000 / 1) / (len(WTRoughDf.trial.unique()))
            freqWTRougherList      = (WTRougherLicksList['count'].values / binSize) * (1000 / 1) / (len(WTRougherDf.trial.unique()))
            # calculate plot centers
            plotCentersListSRGAP2C = [interval.mid for interval in SRGAP2CRoughLicksList.index]
            plotCentersListWT      = [interval.mid for interval in WTRoughLicksList.index]
            # add to dictionaries
            freqSRGAP2CRoughDict[mouseSRGAP2C]   = freqSRGAP2CRoughList
            freqSRGAP2CRougherDict[mouseSRGAP2C] = freqSRGAP2CRougherList
            freqWTRoughDict[mouseWT]             = freqWTRoughList
            freqWTRougherDict[mouseWT]           = freqWTRougherList
            plotCentersDict[mouseSRGAP2C]        = plotCentersListSRGAP2C
            plotCentersDict[mouseWT]             = plotCentersListWT
    else:
        for WTDf, SRGAP2CDf in zip(tmsDict['WT'],cycle(tmsDict['SRGAP2C'])):
            # gather mouse names
            mouseWT      = ''.join(WTDf.mouse.unique())
            mouseSRGAP2C = ''.join(SRGAP2CDf.mouse.unique())
            # grup by stimulus
            SRGAP2CRoughDf   = SRGAP2CDf.groupby('stim').get_group('SMOOTH')
            SRGAP2CRougherDf = SRGAP2CDf.groupby('stim').get_group('ROUGH')
            WTRoughDf        = WTDf.groupby('stim').get_group('SMOOTH')
            WTRougherDf      = WTDf.groupby('stim').get_group('ROUGH')
            # merge left and right lick times
            tempSRGAP2CRoughDf      = pd.DataFrame({'left':SRGAP2CRoughDf.left, 'right':SRGAP2CRoughDf.right})  
            tempSRGAP2CRougherDf    = pd.DataFrame({'left':SRGAP2CRougherDf.left, 'right':SRGAP2CRougherDf.right})  
            tempWTRoughDf           = pd.DataFrame({'left':WTRoughDf.left, 'right':WTRoughDf.right})  
            tempWTRougherDf         = pd.DataFrame({'left':WTRougherDf.left, 'right':WTRougherDf.right})
            licksSRGAP2CRoughList   = tempSRGAP2CRoughDf.stack().groupby(level=0).sum()
            licksSRGAP2CRougherList = tempSRGAP2CRougherDf.stack().groupby(level=0).sum()
            licksWTRoughList        = tempWTRoughDf.stack().groupby(level=0).sum()
            licksWTRougherList      = tempWTRougherDf.stack().groupby(level=0).sum()
            SRGAP2CRoughDf          = SRGAP2CRoughDf.assign(licks=licksSRGAP2CRoughList)
            SRGAP2CRougherDf        = SRGAP2CRougherDf.assign(licks=licksSRGAP2CRougherList)
            WTRoughDf               = WTRoughDf.assign(licks=licksWTRoughList)
            WTRougherDf             = WTRougherDf.assign(licks=licksWTRougherList)  
            # calculate lick counts binned by time for SRGAP2C and WT for each stimulus
            binsList                = np.linspace(liml,limr,bins)
            SRGAP2CRoughLicksList   = SRGAP2CRoughDf.groupby(pd.cut(SRGAP2CRoughDf.licks,binsList,include_lowest=True)).licks.agg(['count'])
            SRGAP2CRougherLicksList = SRGAP2CRougherDf.groupby(pd.cut(SRGAP2CRougherDf.licks,binsList,include_lowest=True)).licks.agg(['count'])
            WTRoughLicksList        = WTRoughDf.groupby(pd.cut(WTRoughDf.licks,binsList,include_lowest=True)).licks.agg(['count'])
            WTRougherLicksList      = WTRougherDf.groupby(pd.cut(WTRougherDf.licks,binsList,include_lowest=True)).licks.agg(['count'])
            # calculate lick frequency binned by time for SRGAP2C and WT for each stimulus
            # calculate bin size (ms)
            binSize = (limr - liml) / (bins - 1)
            # convert to frequency in Hz (licks per second) and average across n trials
            freqSRGAP2CRoughList   = (SRGAP2CRoughLicksList['count'].values / binSize)   * (1000 / 1) / (len(SRGAP2CRoughDf.trial.unique()))
            freqSRGAP2CRougherList = (SRGAP2CRougherLicksList['count'].values / binSize)  * (1000 / 1) / (len(SRGAP2CRougherDf.trial.unique()))
            freqWTRoughList        = (WTRoughLicksList['count'].values / binSize)  * (1000 / 1) / (len(WTRoughDf.trial.unique()))
            freqWTRougherList      = (WTRougherLicksList['count'].values / binSize) * (1000 / 1) / (len(WTRougherDf.trial.unique()))
            # calculate plot centers
            plotCentersListSRGAP2C = [interval.mid for interval in SRGAP2CRoughLicksList.index]
            plotCentersListWT      = [interval.mid for interval in WTRoughLicksList.index]
            # add to dictionaries
            freqSRGAP2CRoughDict[mouseSRGAP2C]   = freqSRGAP2CRoughList
            freqSRGAP2CRougherDict[mouseSRGAP2C] = freqSRGAP2CRougherList
            freqWTRoughDict[mouseWT]             = freqWTRoughList
            freqWTRougherDict[mouseWT]           = freqWTRougherList
            plotCentersDict[mouseSRGAP2C]        = plotCentersListSRGAP2C
            plotCentersDict[mouseWT]             = plotCentersListWT
    # generate dataframes
    freqSRGAP2CRoughDf   = pd.DataFrame(freqSRGAP2CRoughDict)
    freqSRGAP2CRougherDf = pd.DataFrame(freqSRGAP2CRougherDict)
    freqWTRoughDf        = pd.DataFrame(freqWTRoughDict)
    freqWTRougherDf      = pd.DataFrame(freqWTRougherDict)
    plotCentersDf        = pd.DataFrame(plotCentersDict)
    # calculate means and errors
    freqSRGAP2CRoughMeanList   = freqSRGAP2CRoughDf.mean(axis=1).values
    freqSRGAP2CRoughSEMList    = freqSRGAP2CRoughDf.sem(axis=1).values
    freqSRGAP2CRougherMeanList = freqSRGAP2CRougherDf.mean(axis=1).values
    freqSRGAP2CRougherSEMList  = freqSRGAP2CRougherDf.sem(axis=1).values
    freqWTRoughMeanList        = freqWTRoughDf.mean(axis=1).values
    freqWTRoughSEMList         = freqWTRoughDf.sem(axis=1).values
    freqWTRougherMeanList      = freqWTRougherDf.mean(axis=1).values
    freqWTRougherSEMList       = freqWTRougherDf.sem(axis=1).values
    # save to csv
    freqSRGAP2CRoughDf.to_csv("frequency_{}_2C_rough.csv".format(status),index=False)
    freqSRGAP2CRougherDf.to_csv("frequency_{}_2C_rougher.csv".format(status),index=False)
    freqWTRoughDf.to_csv("frequency_{}_WT_rough.csv".format(status),index=False)
    freqWTRougherDf.to_csv("frequency_{}_WT_rougher.csv".format(status),index=False)
    # average centers of bins again to correct for numpy rounding errors
    plotCentersMeanList = plotCentersDf.mean(axis=1).values
    # plot
    fig, ax = plt.subplots()
    ax.plot(plotCentersMeanList,freqSRGAP2CRoughMeanList,linestyle='-',color='#035a70',label='SRGAP2C (rough)') 
    ax.plot(plotCentersMeanList,freqSRGAP2CRougherMeanList,linestyle='--',color='#035a70',label='SRGAP2C (rougher)') 
    ax.plot(plotCentersMeanList,freqWTRoughMeanList,linestyle='-',color='black',label='WT (rough)')  
    ax.plot(plotCentersMeanList,freqWTRougherMeanList,linestyle='--',color='black',label='WT (rougher)') 
    legend_without_duplicate_labels(ax,pos='upper right')
    ax.autoscale()
    ax.set_ylim(bottom=0,top=8)
    ax.get_yaxis().set_ticks(np.linspace(2,8,4))
    ax.set_xlim(left=liml,right=limr)
    simpleaxis(ax)
    plt.fill_between(plotCentersMeanList, freqSRGAP2CRoughMeanList-freqSRGAP2CRoughSEMList, freqSRGAP2CRoughMeanList+freqSRGAP2CRoughSEMList, alpha=0.15, edgecolor='none', facecolor='#035a70')
    plt.fill_between(plotCentersMeanList, freqSRGAP2CRougherMeanList-freqSRGAP2CRougherSEMList, freqSRGAP2CRougherMeanList+freqSRGAP2CRougherSEMList, alpha=0.15, edgecolor='none', facecolor='#035a70')
    plt.fill_between(plotCentersMeanList, freqWTRoughMeanList-freqWTRoughSEMList, freqWTRoughMeanList+freqWTRoughSEMList, alpha=0.3, edgecolor='none', facecolor='silver')
    plt.fill_between(plotCentersMeanList, freqWTRougherMeanList-freqWTRougherSEMList, freqWTRougherMeanList+freqWTRougherSEMList, alpha=0.3, edgecolor='none', facecolor='silver')
    plt.xlabel('Time [ms]')
    plt.ylabel('Lick Frequency [Hz]')
    plt.title("{}".format(titleStr2))
    #plt.show()
    plt.savefig(os.path.join(os.getcwd(),"{}_freq_STIM_{}.svg".format(titleStr,status)))
    plt.close('all')
    return None

def makeLickFrequencyPlotsByStim2C(tmsDict,status):
    """Call LickFrequencyLineByStim function for each genotype"""
    plots = LickFrequencyLineByStim2C(tmsDict,status,bins=21)
    return None

def LickFrequencyLine2C(tmsDict,status,liml=-2000,limr=2000,bins=41):
    """Plots frequency of licks as function of time.
    Used for SRGAP2C data."""
    freqSRGAP2CDict   = defaultdict(list)
    freqWTDict        = defaultdict(list)
    plotCentersDict   = defaultdict(list)
    titleStr          = 'SRGAP2C'
    titleStr2         = "Lick Frequency - {} ({})".format(titleStr,status)
    # for each mouse
    # cycle the smaller class
    # wasteful computationally, but will simply overwrite previous values with no consequences
    if len(tmsDict['WT']) <= len(tmsDict['SRGAP2C']):
        for WTDf, SRGAP2CDf in zip(cycle(tmsDict['WT']),tmsDict['SRGAP2C']):
            # gather mouse names
            mouseWT      = ''.join(WTDf.mouse.unique())
            mouseSRGAP2C = ''.join(SRGAP2CDf.mouse.unique())
            # merge left and right lick times
            tempWTDf         = pd.DataFrame({'left':WTDf.left, 'right':WTDf.right})  
            tempSRGAP2CDf    = pd.DataFrame({'left':SRGAP2CDf.left, 'right':SRGAP2CDf.right})  
            licksWTList      = tempWTDf.stack().groupby(level=0).sum()
            licksSRGAP2CList = tempSRGAP2CDf.stack().groupby(level=0).sum()
            WTDf             = WTDf.assign(licks=licksWTList)
            SRGAP2CDf        = SRGAP2CDf.assign(licks=licksSRGAP2CList) 
            # calculate lick counts binned by time for SRGAP2C and WT 
            binsList         = np.linspace(liml,limr,bins)
            SRGAP2CLicksList = SRGAP2CDf.groupby(pd.cut(SRGAP2CDf.licks,binsList,include_lowest=True)).licks.agg(['count'])
            WTLicksList      = WTDf.groupby(pd.cut(WTDf.licks,binsList,include_lowest=True)).licks.agg(['count'])
            # calculate lick frequency binned by time for SRGAP2C and WT 
            # calculate bin size (ms)
            binSize = (limr - liml) / (bins - 1)
            # convert to frequency in Hz (licks per second) and average across n trials
            freqSRGAP2CList = (SRGAP2CLicksList['count'].values / binSize) * (1000 / 1) / (len(SRGAP2CDf.trial.unique()))
            freqWTList      = (WTLicksList['count'].values / binSize) * (1000 / 1) / (len(WTDf.trial.unique()))
            # calculate plot centers
            plotCentersListSRGAP2C = [interval.mid for interval in SRGAP2CLicksList.index]
            plotCentersListWT      = [interval.mid for interval in WTLicksList.index]
            # add to dictionaries
            freqSRGAP2CDict[mouseSRGAP2C] = freqSRGAP2CList
            freqWTDict[mouseWT]           = freqWTList
            plotCentersDict[mouseSRGAP2C] = plotCentersListSRGAP2C
            plotCentersDict[mouseWT]      = plotCentersListWT
    else:
        for WTDf, SRGAP2CDf in zip(tmsDict['WT'],cycle(tmsDict['SRGAP2C'])):
            # gather mouse names
            mouseWT      = ''.join(WTDf.mouse.unique())
            mouseSRGAP2C = ''.join(SRGAP2CDf.mouse.unique())
            # merge left and right lick times
            tempWTDf         = pd.DataFrame({'left':WTDf.left, 'right':WTDf.right})  
            tempSRGAP2CDf    = pd.DataFrame({'left':SRGAP2CDf.left, 'right':SRGAP2CDf.right})  
            licksWTList      = tempWTDf.stack().groupby(level=0).sum()
            licksSRGAP2CList = tempSRGAP2CDf.stack().groupby(level=0).sum()
            WTDf             = WTDf.assign(licks=licksWTList)
            SRGAP2CDf        = SRGAP2CDf.assign(licks=licksSRGAP2CList) 
            # calculate lick counts binned by time for SRGAP2C and WT 
            binsList         = np.linspace(liml,limr,bins)
            SRGAP2CLicksList = SRGAP2CDf.groupby(pd.cut(SRGAP2CDf.licks,binsList,include_lowest=True)).licks.agg(['count'])
            WTLicksList      = WTDf.groupby(pd.cut(WTDf.licks,binsList,include_lowest=True)).licks.agg(['count'])
            # calculate lick frequency binned by time for SRGAP2C and WT 
            # calculate bin size (ms)
            binSize = (limr - liml) / (bins - 1)
            # convert to frequency in Hz (licks per second) and average across n trials
            freqSRGAP2CList = (SRGAP2CLicksList['count'].values / binSize)   * (1000 / 1) / (len(SRGAP2CDf.trial.unique()))
            freqWTList      = (WTLicksList['count'].values / binSize)  * (1000 / 1) / (len(WTDf.trial.unique()))
            # calculate plot centers
            plotCentersListSRGAP2C = [interval.mid for interval in SRGAP2CLicksList.index]
            plotCentersListWT      = [interval.mid for interval in WTLicksList.index]
            # add to dictionaries
            freqSRGAP2CDict[mouseSRGAP2C] = freqSRGAP2CList
            freqWTDict[mouseWT]           = freqWTList
            plotCentersDict[mouseSRGAP2C] = plotCentersListSRGAP2C
            plotCentersDict[mouseWT]      = plotCentersListWT
    # generate dataframes
    freqSRGAP2CDf   = pd.DataFrame(freqSRGAP2CDict)
    freqWTDf        = pd.DataFrame(freqWTDict)
    plotCentersDf   = pd.DataFrame(plotCentersDict)
    # calculate means and errors
    freqSRGAP2CMeanList = freqSRGAP2CDf.mean(axis=1).values
    freqSRGAP2CSEMList  = freqSRGAP2CDf.sem(axis=1).values
    freqWTMeanList      = freqWTDf.mean(axis=1).values
    freqWTSEMList       = freqWTDf.sem(axis=1).values
    # save to csv
    freqSRGAP2CDf.to_csv("frequency_{}_2C.csv".format(status),index=False)
    freqWTDf.to_csv("frequency_{}_WT.csv".format(status),index=False)
    # average centers of bins again to correct for numpy rounding errors
    plotCentersMeanList = plotCentersDf.mean(axis=1).values
    # plot
    fig, ax = plt.subplots()
    ax.plot(plotCentersMeanList,freqSRGAP2CMeanList,linestyle='-',color='#035a70',label='SRGAP2C') 
    ax.plot(plotCentersMeanList,freqWTMeanList,linestyle='-',color='black',label='WT')  
    legend_without_duplicate_labels(ax,pos='upper right')
    ax.autoscale()
    ax.set_ylim(bottom=0,top=8)
    ax.get_yaxis().set_ticks(np.linspace(2,8,4))
    ax.set_xlim(left=liml,right=limr)
    simpleaxis(ax)
    plt.fill_between(plotCentersMeanList, freqSRGAP2CMeanList-freqSRGAP2CSEMList, freqSRGAP2CMeanList+freqSRGAP2CSEMList, alpha=0.15, edgecolor='none', facecolor='#035a70')
    plt.fill_between(plotCentersMeanList, freqWTMeanList-freqWTSEMList, freqWTMeanList+freqWTSEMList, alpha=0.3, edgecolor='none', facecolor='silver')
    plt.xlabel('Time [ms]')
    plt.ylabel('Lick Frequency [Hz]')
    plt.title("{}".format(titleStr2))
    #plt.show()
    plt.savefig(os.path.join(os.getcwd(),"{}_freq_{}.svg".format(titleStr,status)))
    plt.close('all')
    return None

def makeLickFrequencyPlots2C(tmsDict,status):
    """Call LickFrequencyLineByStim function for each genotype.
    Used for SRGAP2C data."""
    plots = LickFrequencyLine2C(tmsDict,status,bins=21)
    return None

def CorrectLineByStim2C(tmsDict,status,liml=-2000,limr=2000,bins=41):
    """Plots proportion of licks correct with respect to stimulus as function of time.
    Used for SRGAP2C data."""
    proportionsSRGAP2CRoughDict   = defaultdict(list)
    proportionsSRGAP2CRougherDict = defaultdict(list)
    proportionsWTRoughDict        = defaultdict(list)
    proportionsWTRougherDict      = defaultdict(list)
    plotCentersDict               = defaultdict(list)
    titleStr                      = 'SRGAP2C'
    titleStr2                     = "Correct Licks (by stim) - {} ({})".format(titleStr,status)
    # for each mouse
    # cycle the smaller class
    # wasteful computationally, but will simply overwrite previous values with no consequences
    if len(tmsDict['WT']) <= len(tmsDict['SRGAP2C']):
        for WTDf, SRGAP2CDf in zip(cycle(tmsDict['WT']),tmsDict['SRGAP2C']):
            # gather mouse names
            mouseWT      = ''.join(WTDf.mouse.unique())
            mouseSRGAP2C = ''.join(SRGAP2CDf.mouse.unique())
            # identify correct licks
            WTCorrects      = np.where((WTDf.left.notna() & WTDf.stim.isin(['ROUGH'])) | (WTDf.right.notna() & WTDf.stim.isin(['SMOOTH'])),1,0) 
            WTDf            = WTDf.copy().assign(correct=WTCorrects)
            SRGAP2CCorrects = np.where((SRGAP2CDf.left.notna() & SRGAP2CDf.stim.isin(['ROUGH'])) | (SRGAP2CDf.right.notna() & SRGAP2CDf.stim.isin(['SMOOTH'])),1,0) 
            SRGAP2CDf       = SRGAP2CDf.copy().assign(correct=SRGAP2CCorrects)
            # grup by stimulus
            SRGAP2CRoughDf   = SRGAP2CDf.groupby('stim').get_group('SMOOTH')
            SRGAP2CRougherDf = SRGAP2CDf.groupby('stim').get_group('ROUGH')
            WTRoughDf        = WTDf.groupby('stim').get_group('SMOOTH')
            WTRougherDf      = WTDf.groupby('stim').get_group('ROUGH')
            # merge left and right lick times
            tempWTRougherDf         = pd.DataFrame({'left':WTRougherDf.left, 'right':WTRougherDf.right})  
            tempWTRoughDf           = pd.DataFrame({'left':WTRoughDf.left, 'right':WTRoughDf.right})  
            tempSRGAP2CRougherDf    = pd.DataFrame({'left':SRGAP2CRougherDf.left, 'right':SRGAP2CRougherDf.right})  
            tempSRGAP2CRoughDf      = pd.DataFrame({'left':SRGAP2CRoughDf.left, 'right':SRGAP2CRoughDf.right})
            licksWTRougherList      = tempWTRougherDf.stack().groupby(level=0).sum()
            licksWTRoughList        = tempWTRoughDf.stack().groupby(level=0).sum()
            licksSRGAP2CRougherList = tempSRGAP2CRougherDf.stack().groupby(level=0).sum()
            licksSRGAP2CRoughList   = tempSRGAP2CRoughDf.stack().groupby(level=0).sum()
            WTRougherDf             = WTRougherDf.assign(licks=licksWTRougherList)
            WTRoughDf               = WTRoughDf.assign(licks=licksWTRoughList)
            SRGAP2CRougherDf        = SRGAP2CRougherDf.assign(licks=licksSRGAP2CRougherList)
            SRGAP2CRoughDf          = SRGAP2CRoughDf.assign(licks=licksSRGAP2CRoughList)  
            # calculate proportions binned by time for laser on and off for each stimulus
            proportionsWTRougher, plotCentersWTRougher, errorWTRougher                = bin_by_time_and_binary(WTRougherDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
            proportionsWTRough, plotCentersWTRough, errorWTRough                      = bin_by_time_and_binary(WTRoughDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
            proportionsSRGAP2CRougher, plotCentersSRGAP2CRougher, errorSRGAP2CRougher = bin_by_time_and_binary(SRGAP2CRougherDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
            proportionsSRGAP2CRough, plotCentersSRGAP2CRough, errorSRGAP2CRough       = bin_by_time_and_binary(SRGAP2CRoughDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
            # average centers of bins to correct for numpy rounding errors
            plotCentersListWT      = [np.mean([p1,p2]) for p1,p2 in zip(plotCentersWTRougher,plotCentersWTRough)]
            plotCentersListSRGAP2C = [np.mean([p1,p2]) for p1,p2 in zip(plotCentersSRGAP2CRougher,plotCentersSRGAP2CRough)]
            # add to dictionaries
            proportionsWTRougherDict[mouseWT]           = proportionsWTRougher.values
            proportionsWTRoughDict[mouseWT]             = proportionsWTRough.values
            plotCentersDict[mouseWT]                    = plotCentersListWT
            proportionsSRGAP2CRougherDict[mouseSRGAP2C] = proportionsSRGAP2CRougher.values
            proportionsSRGAP2CRoughDict[mouseSRGAP2C]   = proportionsSRGAP2CRough.values
            plotCentersDict[mouseSRGAP2C]               = plotCentersListSRGAP2C
    else:
        for WTDf, SRGAP2CDf in zip(tmsDict['WT'],cycle(tmsDict['SRGAP2C'])):
            # gather mouse names
            mouseWT      = ''.join(WTDf.mouse.unique())
            mouseSRGAP2C = ''.join(SRGAP2CDf.mouse.unique())
            # identify correct licks
            WTCorrects      = np.where((WTDf.left.notna() & WTDf.stim.isin(['ROUGH'])) | (WTDf.right.notna() & WTDf.stim.isin(['SMOOTH'])),1,0) 
            WTDf            = WTDf.copy().assign(correct=WTCorrects)
            SRGAP2CCorrects = np.where((SRGAP2CDf.left.notna() & SRGAP2CDf.stim.isin(['ROUGH'])) | (SRGAP2CDf.right.notna() & SRGAP2CDf.stim.isin(['SMOOTH'])),1,0) 
            SRGAP2CDf       = SRGAP2CDf.copy().assign(correct=SRGAP2CCorrects)
            # grup by stimulus
            SRGAP2CRoughDf   = SRGAP2CDf.groupby('stim').get_group('SMOOTH')
            SRGAP2CRougherDf = SRGAP2CDf.groupby('stim').get_group('ROUGH')
            WTRoughDf        = WTDf.groupby('stim').get_group('SMOOTH')
            WTRougherDf      = WTDf.groupby('stim').get_group('ROUGH')
            # merge left and right lick times
            tempWTRougherDf         = pd.DataFrame({'left':WTRougherDf.left, 'right':WTRougherDf.right})  
            tempWTRoughDf           = pd.DataFrame({'left':WTRoughDf.left, 'right':WTRoughDf.right})  
            tempSRGAP2CRougherDf    = pd.DataFrame({'left':SRGAP2CRougherDf.left, 'right':SRGAP2CRougherDf.right})  
            tempSRGAP2CRoughDf      = pd.DataFrame({'left':SRGAP2CRoughDf.left, 'right':SRGAP2CRoughDf.right})
            licksWTRougherList      = tempWTRougherDf.stack().groupby(level=0).sum()
            licksWTRoughList        = tempWTRoughDf.stack().groupby(level=0).sum()
            licksSRGAP2CRougherList = tempSRGAP2CRougherDf.stack().groupby(level=0).sum()
            licksSRGAP2CRoughList   = tempSRGAP2CRoughDf.stack().groupby(level=0).sum()
            WTRougherDf             = WTRougherDf.assign(licks=licksWTRougherList)
            WTRoughDf               = WTRoughDf.assign(licks=licksWTRoughList)
            SRGAP2CRougherDf        = SRGAP2CRougherDf.assign(licks=licksSRGAP2CRougherList)
            SRGAP2CRoughDf          = SRGAP2CRoughDf.assign(licks=licksSRGAP2CRoughList)  
            # calculate proportions binned by time for laser on and off for each stimulus
            proportionsWTRougher, plotCentersWTRougher, errorWTRougher                = bin_by_time_and_binary(WTRougherDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
            proportionsWTRough, plotCentersWTRough, errorWTRough                      = bin_by_time_and_binary(WTRoughDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
            proportionsSRGAP2CRougher, plotCentersSRGAP2CRougher, errorSRGAP2CRougher = bin_by_time_and_binary(SRGAP2CRougherDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
            proportionsSRGAP2CRough, plotCentersSRGAP2CRough, errorSRGAP2CRough       = bin_by_time_and_binary(SRGAP2CRoughDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
            # average centers of bins to correct for numpy rounding errors
            plotCentersListWT      = [np.mean([p1,p2]) for p1,p2 in zip(plotCentersWTRougher,plotCentersWTRough)]
            plotCentersListSRGAP2C = [np.mean([p1,p2]) for p1,p2 in zip(plotCentersSRGAP2CRougher,plotCentersSRGAP2CRough)]
            # add to dictionaries
            proportionsWTRougherDict[mouseWT]           = proportionsWTRougher.values
            proportionsWTRoughDict[mouseWT]             = proportionsWTRough.values
            plotCentersDict[mouseWT]                    = plotCentersListWT
            proportionsSRGAP2CRougherDict[mouseSRGAP2C] = proportionsSRGAP2CRougher.values
            proportionsSRGAP2CRoughDict[mouseSRGAP2C]   = proportionsSRGAP2CRough.values
            plotCentersDict[mouseSRGAP2C]               = plotCentersListSRGAP2C
    # generate dataframes
    proportionsWTRougherDf      = pd.DataFrame(proportionsWTRougherDict)
    proportionsWTRoughDf        = pd.DataFrame(proportionsWTRoughDict)
    proportionsSRGAP2CRougherDf = pd.DataFrame(proportionsSRGAP2CRougherDict)
    proportionsSRGAP2CRoughDf   = pd.DataFrame(proportionsSRGAP2CRoughDict)
    plotCentersDf               = pd.DataFrame(plotCentersDict)
    # calculate means and errors
    proportionsWTRougherMeanList      = proportionsWTRougherDf.mean(axis=1).values
    proportionsWTRougherSEMList       = proportionsWTRougherDf.sem(axis=1).values
    proportionsWTRoughMeanList        = proportionsWTRoughDf.mean(axis=1).values
    proportionsWTRoughSEMList         = proportionsWTRoughDf.sem(axis=1).values
    proportionsSRGAP2CRougherMeanList = proportionsSRGAP2CRougherDf.mean(axis=1).values
    proportionsSRGAP2CRougherSEMList  = proportionsSRGAP2CRougherDf.sem(axis=1).values
    proportionsSRGAP2CRoughMeanList   = proportionsSRGAP2CRoughDf.mean(axis=1).values
    proportionsSRGAP2CRoughSEMList    = proportionsSRGAP2CRoughDf.sem(axis=1).values
    # save to csv
    proportionsSRGAP2CRoughDf.to_csv("correct_{}_2C_rough.csv".format(status),index=False)
    proportionsSRGAP2CRougherDf.to_csv("correct_{}_2C_rougher.csv".format(status),index=False)
    proportionsWTRoughDf.to_csv("correct_{}_WT_rough.csv".format(status),index=False)
    proportionsWTRougherDf.to_csv("correct_{}_WT_rougher.csv".format(status),index=False)
    # average centers of bins again to correct for numpy rounding errors
    plotCentersMeanList = plotCentersDf.mean(axis=1).values
    # plot
    fig, ax = plt.subplots()
    ax.plot(plotCentersMeanList,proportionsWTRougherMeanList,linestyle='--',color='black',label='WT (rougher)')
    ax.plot(plotCentersMeanList,proportionsWTRoughMeanList,linestyle='-',color='black',label='WT (rough)')
    ax.plot(plotCentersMeanList,proportionsSRGAP2CRougherMeanList,linestyle='--',color='#035a70',label='SRGAP2C (rougher)')
    ax.plot(plotCentersMeanList,proportionsSRGAP2CRoughMeanList,linestyle='-',color='#035a70',label='SRGAP2C (rough)')
    legend_without_duplicate_labels(ax)
    ax.autoscale()
    ax.set_ylim(bottom=0,top=1.05)
    ax.get_yaxis().set_ticks(np.linspace(0.25,1.0,4))
    ax.set_xlim(left=liml,right=limr)
    simpleaxis(ax)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.fill_between(plotCentersMeanList, proportionsWTRougherMeanList-proportionsWTRougherSEMList, proportionsWTRougherMeanList+proportionsWTRougherSEMList, alpha=0.3, edgecolor='none', facecolor='silver')
    plt.fill_between(plotCentersMeanList, proportionsWTRoughMeanList-proportionsWTRoughSEMList, proportionsWTRoughMeanList+proportionsWTRoughSEMList, alpha=0.3, edgecolor='none', facecolor='silver')
    plt.fill_between(plotCentersMeanList, proportionsSRGAP2CRougherMeanList-proportionsSRGAP2CRougherSEMList, proportionsSRGAP2CRougherMeanList+proportionsSRGAP2CRougherSEMList, alpha=0.15, edgecolor='none', facecolor='#035a70')
    plt.fill_between(plotCentersMeanList, proportionsSRGAP2CRoughMeanList-proportionsSRGAP2CRoughSEMList, proportionsSRGAP2CRoughMeanList+proportionsSRGAP2CRoughSEMList, alpha=0.15, edgecolor='none', facecolor='#035a70')
    plt.xlabel('Time [ms]')
    plt.ylabel('Lick Probability')
    plt.title("{}".format(titleStr2))
    #plt.show()
    plt.savefig(os.path.join(os.getcwd(),"{}_concat_correct_STIM_{}.svg".format(titleStr,status)))
    plt.close('all')
    return None

def makeCorrectLineByStim2C(tmsDict,status):
    """Call CorrectLineByStim2C function.
    Used for SRGAP2C data."""
    plots = CorrectLineByStim2C(tmsDict,status,bins=21)
    return None

def CorrectLine2C(tmsDict,status,liml=-2000,limr=2000,bins=41):
    """Plots proportion of licks correct as function of time.
    Used for SRGAP2C data."""
    proportionsSRGAP2CDict = defaultdict(list)
    proportionsWTDict      = defaultdict(list)
    plotCentersDict        = defaultdict(list)
    titleStr               = 'SRGAP2C'
    titleStr2              = "Correct Licks - {} ({})".format(titleStr,status)
    # for each mouse
    # cycle the smaller class
    # wasteful computationally, but will simply overwrite previous values with no consequences
    if len(tmsDict['WT']) <= len(tmsDict['SRGAP2C']):
        for WTDf, SRGAP2CDf in zip(cycle(tmsDict['WT']),tmsDict['SRGAP2C']):
            # gather mouse names
            mouseWT      = ''.join(WTDf.mouse.unique())
            mouseSRGAP2C = ''.join(SRGAP2CDf.mouse.unique())
            # identify correct licks
            WTCorrects      = np.where((WTDf.left.notna() & WTDf.stim.isin(['ROUGH'])) | (WTDf.right.notna() & WTDf.stim.isin(['SMOOTH'])),1,0) 
            WTDf            = WTDf.copy().assign(correct=WTCorrects)
            SRGAP2CCorrects = np.where((SRGAP2CDf.left.notna() & SRGAP2CDf.stim.isin(['ROUGH'])) | (SRGAP2CDf.right.notna() & SRGAP2CDf.stim.isin(['SMOOTH'])),1,0) 
            SRGAP2CDf       = SRGAP2CDf.copy().assign(correct=SRGAP2CCorrects)
            # merge left and right lick times
            tempWTDf         = pd.DataFrame({'left':WTDf.left, 'right':WTDf.right})  
            tempSRGAP2CDf    = pd.DataFrame({'left':SRGAP2CDf.left, 'right':SRGAP2CDf.right})  
            licksWTList      = tempWTDf.stack().groupby(level=0).sum()
            licksSRGAP2CList = tempSRGAP2CDf.stack().groupby(level=0).sum()
            WTDf             = WTDf.assign(licks=licksWTList)
            SRGAP2CDf        = SRGAP2CDf.assign(licks=licksSRGAP2CList)
            # calculate proportions binned by time for laser on and off for each genotype
            proportionsWT, plotCentersWT, errorWT                = bin_by_time_and_binary(WTDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
            proportionsSRGAP2C, plotCentersSRGAP2C, errorSRGAP2C = bin_by_time_and_binary(SRGAP2CDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
            # add to dictionaries
            proportionsWTDict[mouseWT]           = proportionsWT.values
            plotCentersDict[mouseWT]             = plotCentersWT
            proportionsSRGAP2CDict[mouseSRGAP2C] = proportionsSRGAP2C.values
            plotCentersDict[mouseSRGAP2C]        = plotCentersSRGAP2C
    else:
        for WTDf, SRGAP2CDf in zip(tmsDict['WT'],cycle(tmsDict['SRGAP2C'])):
            # gather mouse names
            mouseWT      = ''.join(WTDf.mouse.unique())
            mouseSRGAP2C = ''.join(SRGAP2CDf.mouse.unique())
            # identify correct licks
            WTCorrects      = np.where((WTDf.left.notna() & WTDf.stim.isin(['ROUGH'])) | (WTDf.right.notna() & WTDf.stim.isin(['SMOOTH'])),1,0) 
            WTDf            = WTDf.copy().assign(correct=WTCorrects)
            SRGAP2CCorrects = np.where((SRGAP2CDf.left.notna() & SRGAP2CDf.stim.isin(['ROUGH'])) | (SRGAP2CDf.right.notna() & SRGAP2CDf.stim.isin(['SMOOTH'])),1,0) 
            SRGAP2CDf       = SRGAP2CDf.copy().assign(correct=SRGAP2CCorrects)
            # merge left and right lick times
            tempWTDf         = pd.DataFrame({'left':WTDf.left, 'right':WTDf.right})  
            tempSRGAP2CDf    = pd.DataFrame({'left':SRGAP2CDf.left, 'right':SRGAP2CDf.right})  
            licksWTList      = tempWTDf.stack().groupby(level=0).sum()
            licksSRGAP2CList = tempSRGAP2CDf.stack().groupby(level=0).sum()
            WTDf             = WTDf.assign(licks=licksWTList)
            SRGAP2CDf        = SRGAP2CDf.assign(licks=licksSRGAP2CList)
            # calculate proportions binned by time for laser on and off for each genotype
            proportionsWT, plotCentersWT, errorWT                = bin_by_time_and_binary(WTDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
            proportionsSRGAP2C, plotCentersSRGAP2C, errorSRGAP2C = bin_by_time_and_binary(SRGAP2CDf,time_col='licks',bin_col='correct',liml=liml,limr=limr,bins=bins)
            # add to dictionaries
            proportionsWTDict[mouseWT]           = proportionsWT.values
            plotCentersDict[mouseWT]             = plotCentersWT
            proportionsSRGAP2CDict[mouseSRGAP2C] = proportionsSRGAP2C.values
            plotCentersDict[mouseSRGAP2C]        = plotCentersSRGAP2C
    # generate dataframes
    proportionsWTDf      = pd.DataFrame(proportionsWTDict)
    proportionsSRGAP2CDf = pd.DataFrame(proportionsSRGAP2CDict)
    plotCentersDf        = pd.DataFrame(plotCentersDict)
    # calculate means and errors
    proportionsWTMeanList      = proportionsWTDf.mean(axis=1).values
    proportionsWTSEMList       = proportionsWTDf.sem(axis=1).values
    proportionsSRGAP2CMeanList = proportionsSRGAP2CDf.mean(axis=1).values
    proportionsSRGAP2CSEMList  = proportionsSRGAP2CDf.sem(axis=1).values
    # save to csv
    proportionsSRGAP2CDf.to_csv("correct_{}_2C.csv".format(status),index=False)
    proportionsWTDf.to_csv("correct_{}_WT.csv".format(status),index=False)
    # average centers of bins again to correct for numpy rounding errors
    plotCentersMeanList = plotCentersDf.mean(axis=1).values
    # plot
    fig, ax = plt.subplots()
    ax.plot(plotCentersMeanList,proportionsWTMeanList,linestyle='-',color='black',label='WT')
    ax.plot(plotCentersMeanList,proportionsSRGAP2CMeanList,linestyle='-',color='#035a70',label='SRGAP2C')
    legend_without_duplicate_labels(ax)
    ax.autoscale()
    ax.set_ylim(bottom=0,top=1.05)
    ax.get_yaxis().set_ticks(np.linspace(0.25,1.0,4))
    ax.set_xlim(left=liml,right=limr)
    simpleaxis(ax)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.fill_between(plotCentersMeanList, proportionsWTMeanList-proportionsWTSEMList, proportionsWTMeanList+proportionsWTSEMList, alpha=0.3, edgecolor='none', facecolor='silver')
    plt.fill_between(plotCentersMeanList, proportionsSRGAP2CMeanList-proportionsSRGAP2CSEMList, proportionsSRGAP2CMeanList+proportionsSRGAP2CSEMList, alpha=0.15, edgecolor='none', facecolor='#035a70')
    plt.xlabel('Time [ms]')
    plt.ylabel('Lick Probability')
    plt.title("{}".format(titleStr2))
    #plt.show()
    plt.savefig(os.path.join(os.getcwd(),"{}_concat_correct_{}.svg".format(titleStr,status)))
    plt.close('all')
    return None

def makeCorrectLine2C(tmsDict,status):
    """Call CorrectLine2C function.
    Used for SRGAP2C data."""
    plots = CorrectLine2C(tmsDict,status,bins=21)
    return None

def LickCount2C(tmsWTDict,tmsSRGAP2CDict,status,liml=-2000,limr=2000):
    """Plots average and SEM licks per session by genotype.
    Used for SRGAP2C data."""
    WTSessions      = [int(k) for k in tmsWTDict.keys()]
    SRGAP2CSessions = [int(k) for k in tmsSRGAP2CDict.keys()]
    sessions        = [sess+1 for sess in range(max(max([WTSessions,SRGAP2CSessions], key=max)))]
    # initiate dictionaries
    countWTDict      = defaultdict(list)
    countSRGAP2CDict = defaultdict(list)
    sessionsDict     = defaultdict(list)
    titleStr         = 'SRGAP2C'
    titleStr2        = "Lick Count - {}".format(titleStr)
    # for each day
    for sessionWT,tmListWT, in tmsWTDict.items():
        # for each mouse session
        for tmWT in tmListWT:
            # gather mouse name
            mouseWT = ''.join(tmWT.mouse.unique())
            # merge left and right lick times
            tempWTDf    = pd.DataFrame({'left':tmWT.left, 'right':tmWT.right})  
            licksWTList = tempWTDf.stack().groupby(level=0).sum()
            tmWT        = tmWT.assign(licks=licksWTList)
            # calculate lick counts bounded by [liml, limr] for whole session for each mouse
            binsList = np.linspace(liml,limr,2)
            WTLicks  = tmWT.groupby(pd.cut(tmWT.licks,binsList, include_lowest=True)).licks.agg(['count'])['count'].values[0]
            # calculate average lick counts per trial in bounded time for whole session for each mouse
            WTLicksCount = WTLicks / (len(tmWT.trial.unique()))
            # store in dict
            countWTDict[mouseWT].append(WTLicksCount)
    # for each day
    for sessionSRGAP2C,tmListSRGAP2C in tmsSRGAP2CDict.items():
        # for each mouse session
        for tmSRGAP2C in tmListSRGAP2C:
            # gather mouse name
            mouseSRGAP2C = ''.join(tmSRGAP2C.mouse.unique())
            # merge left and right lick times 
            tempSRGAP2CDf    = pd.DataFrame({'left':tmSRGAP2C.left, 'right':tmSRGAP2C.right})  
            licksSRGAP2CList = tempSRGAP2CDf.stack().groupby(level=0).sum()
            tmSRGAP2C        = tmSRGAP2C.assign(licks=licksSRGAP2CList)
            # calculate lick counts bounded by [liml, limr] for whole session for each mouse
            binsList     = np.linspace(liml,limr,2)
            SRGAP2CLicks = tmSRGAP2C.groupby(pd.cut(tmSRGAP2C.licks,binsList, include_lowest=True)).licks.agg(['count'])['count'].values[0]
            # calculate average lick counts per trial in bounded time for whole session for each mouse
            SRGAP2CLicksCount = SRGAP2CLicks / (len(tmSRGAP2C.trial.unique()))
            # store in dict
            countSRGAP2CDict[mouseSRGAP2C].append(SRGAP2CLicksCount) 
    countWTDf      = pd.DataFrame.from_dict(countWTDict, orient='index').transpose()  
    countSRGAP2CDf = pd.DataFrame.from_dict(countSRGAP2CDict, orient='index').transpose()  
    # calculate means and errors
    countWTMeanList      = countWTDf.mean(axis=1).values
    countWTSEMList       = countWTDf.sem(axis=1).values
    countSRGAP2CMeanList = countSRGAP2CDf.mean(axis=1).values
    countSRGAP2CSEMList  = countSRGAP2CDf.sem(axis=1).values
    # plot
    fig, ax = plt.subplots()
    ax.plot(sessions,countWTMeanList,linestyle='-',color='black',label='WT')  
    ax.plot(sessions,countSRGAP2CMeanList,linestyle='-',color='#035a70',label='SRGAP2C')  
    legend_without_duplicate_labels(ax,pos='upper right')
    ax.autoscale()
    ax.set_ylim(bottom=0,top=18)
    ax.get_yaxis().set_ticks(np.linspace(2,18,9))
    ax.set_xlim(left=min(sessions),right=max(sessions))
    simpleaxis(ax)
    plt.fill_between(sessions, countWTMeanList-countWTSEMList, countWTMeanList+countWTSEMList, alpha=0.3, edgecolor='none', facecolor='silver')
    plt.fill_between(sessions, countSRGAP2CMeanList-countSRGAP2CSEMList, countSRGAP2CMeanList+countSRGAP2CSEMList, alpha=0.15, edgecolor='none', facecolor='#035a70')
    plt.xlabel('Session')
    plt.ylabel('Licks Per Trial')
    plt.title("{}".format(titleStr2))
    #plt.show()
    plt.savefig(os.path.join(os.getcwd(),"{}_lick_count_{}.svg".format(titleStr,status)))
    plt.close('all')
    return None    

def makeLickCount2C(tmsWTDict,tmsSRGAP2CDict,status):
    """Call LickCount2C function to plot average and SEM number of licks by genotype 
    across sessions. Used for SRGAP2C data."""
    plots = LickCount2C(tmsWTDict,tmsSRGAP2CDict,status)
    return None

def SwitchLatency2CLT(tmsWTDict,tmsSRGAP2CDict,status):
    """Plots boxplot of switch latency between reward sides for WT and SRGAP2C mice.
    Individual mouse values calculated are medians to eliminate biases from outliers.
    Three days of training plotted adjacently for each genotype.
    Used for SRGAP2C LT data."""
    # initiate dictionaries
    latencyWTDict      = defaultdict(list)
    latencySRGAP2CDict = defaultdict(list)
    titleStr           = "LT_latency"
    titleStr2          = "Lick Train Switch Latency"
    # for each day (WT)
    for sessionWT, tmListWT in tmsWTDict.items():
        # for each mouse session
        for tm in tmListWT:
            # gather mouse name
            mouse = ''.join(tm.mouse.unique())
            # identify correct licks
            corrects = np.where((tm.left.notna() & tm.rwsd.isin(['ROUGH'])) | (tm.right.notna() & tm.rwsd.isin(['SMOOTH'])),1,0) 
            tm       = tm.copy().assign(correct=corrects)
            # merge left and right lick times
            tempDf    = pd.DataFrame({'left':tm.left, 'right':tm.right})  
            licksList = tempDf.stack().groupby(level=0).sum()
            tm        = tm.assign(licks=licksList)
            # record first correct lick time by trial
            firstCorrectList = tm.sort_values(by=['correct','raw_time'],ascending=[False,True]).groupby('trial').first().licks
            # calculate median latency
            medianLatency = firstCorrectList.median()
            # store mouse data
            latencyWTDict[mouse].append(medianLatency)
    # for each day (SRGAP2C)
    for sessionSRGAP2C, tmListSRGAP2C in tmsSRGAP2CDict.items():
        # for each mouse session
        for tm in tmListSRGAP2C:
            # gather mouse name
            mouse = ''.join(tm.mouse.unique())
            # identify correct licks
            corrects = np.where((tm.left.notna() & tm.rwsd.isin(['ROUGH'])) | (tm.right.notna() & tm.rwsd.isin(['SMOOTH'])),1,0) 
            tm       = tm.copy().assign(correct=corrects)
            # merge left and right lick times
            tempDf    = pd.DataFrame({'left':tm.left, 'right':tm.right})  
            licksList = tempDf.stack().groupby(level=0).sum()
            tm        = tm.assign(licks=licksList)
            # record first correct lick time by trial
            firstCorrectList = tm.sort_values(by=['correct','raw_time'],ascending=[False,True]).groupby('trial').first().licks
            # calculate median latency
            medianLatency = firstCorrectList.median()
            # store mouse data
            latencySRGAP2CDict[mouse].append(medianLatency)
    # convert dictionaries to dataframes
    latencyWTDf      = pd.DataFrame.from_dict(latencyWTDict,orient='index').transpose()
    latencySRGAP2CDf = pd.DataFrame.from_dict(latencySRGAP2CDict,orient='index').transpose()
    # save to csv
    latencySRGAP2CDf.to_csv("{}_latency_2C.csv".format(status),index=False)
    latencyWTDf.to_csv("{}_latency_WT.csv".format(status),index=False)
    # determine maximum number of sessions
    WTSessions      = [int(k) for k in tmsWTDict.keys()]
    SRGAP2CSessions = [int(k) for k in tmsSRGAP2CDict.keys()]
    sessions        = [sess+1 for sess in range(max(max([WTSessions,SRGAP2CSessions], key=max)))]
    # set sessions as x-axis ticks
    ticks = sessions
    # store DFs as lists of lists
    latencyWTLicks      = latencyWTDf.stack().groupby(level=0).apply(list)  
    latencySRGAP2CLicks = latencySRGAP2CDf.stack().groupby(level=0).apply(list)  
    # plot
    fig, ax   = plt.subplots()
    bpWT      = ax.boxplot(latencyWTLicks, positions=np.array(range(len(WTSessions)))*2.0-0.4, widths=0.6, patch_artist=True)
    bpSRGAP2C = ax.boxplot(latencySRGAP2CLicks, positions=np.array(range(len(SRGAP2CSessions)))*2.0+0.4, widths=0.6, patch_artist=True)
    set_box_color(bpWT, 'silver', 0.8)
    set_box_color(bpSRGAP2C, '#035a70', 0.8)
    # fake lines to generate legend
    ax.plot([], c='black', label='WT')
    ax.plot([], c='#035a70', label='SRGAP2C')
    legend_without_duplicate_labels(ax,pos='upper right')
    ax.autoscale()
    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    plt.xlabel('LT Session')
    plt.ylabel('Switch Latency [ms]')
    plt.title("{}".format(titleStr2))
    #plt.show()
    plt.savefig(os.path.join(os.getcwd(),"{}_2C.svg".format(titleStr)))
    plt.close('all')
    return None 

def makeSwitchLatency2CLT(tmsWTDict,tmsSRGAP2CDict,status):
    """Call SwitchLatency2CLT function to plot switch latency for licktrain sessions.)
    Used for SRGAP2C LT data."""
    plots = SwitchLatency2CLT(tmsWTDict,tmsSRGAP2CDict,status)
    return None

def SwitchCount2CLT(tmsWTDict,tmsSRGAP2CDict,status):
    """Plots boxplot of number of licks before switching reward sides for WT and SRGAP2C mice.
    Individual mouse values calculated are medians to eliminate biases from outliers.
    Three days of training plotted adjacently for each genotype.
    Used for SRGAP2C LT data."""
    # initiate dictionaries
    switchWTDict      = defaultdict(list)
    switchSRGAP2CDict = defaultdict(list)
    titleStr          = "LT_switch"
    titleStr2         = "Lick Train Licks Before Switching Sides"
    # for each day (WT)
    for sessionWT, tmListWT in tmsWTDict.items():
        # for each mouse session
        for tm in tmListWT:
            # gather mouse name
            mouse = ''.join(tm.mouse.unique())
            # identify correct licks
            corrects = np.where((tm.left.notna() & tm.rwsd.isin(['ROUGH'])) | (tm.right.notna() & tm.rwsd.isin(['SMOOTH'])),1,0) 
            tm       = tm.copy().assign(correct=corrects)
            # merge left and right lick times
            tempDf    = pd.DataFrame({'left':tm.left, 'right':tm.right})  
            licksList = tempDf.stack().groupby(level=0).sum()
            tm        = tm.assign(licks=licksList)
            # record number of licks before switching side for first correct lick by trial
            # note that the first correct lick is NOT included in this count
            # i.e., if the fourth lick is correct, the script will return the value three
            licksBeforeSwitchList = tm.groupby('trial').correct.apply(lambda x: np.argmax(x.values))
            # calculate median latency
            medianLicks = licksBeforeSwitchList.median()
            # store mouse data
            switchWTDict[mouse].append(medianLicks)
    # for each day (SRGAP2C)
    for sessionSRGAP2C, tmListSRGAP2C in tmsSRGAP2CDict.items():
        # for each mouse session
        for tm in tmListSRGAP2C:
            # gather mouse name
            mouse = ''.join(tm.mouse.unique())
            # identify correct licks
            corrects = np.where((tm.left.notna() & tm.rwsd.isin(['ROUGH'])) | (tm.right.notna() & tm.rwsd.isin(['SMOOTH'])),1,0) 
            tm       = tm.copy().assign(correct=corrects)
            # merge left and right lick times
            tempDf    = pd.DataFrame({'left':tm.left, 'right':tm.right})  
            licksList = tempDf.stack().groupby(level=0).sum()
            tm        = tm.assign(licks=licksList)
            # record number of licks before switching side for first correct lick by trial
            # note that the first correct lick is NOT included in this count
            # i.e., if the fourth lick is correct, the script will return the value three
            licksBeforeSwitchList = tm.groupby('trial').correct.apply(lambda x: np.argmax(x.values))
            # calculate median latency
            medianLicks = licksBeforeSwitchList.median()
            # store mouse data
            switchSRGAP2CDict[mouse].append(medianLicks)
    # convert dictionaries to dataframes
    switchWTDf      = pd.DataFrame.from_dict(switchWTDict,orient='index').transpose()
    switchSRGAP2CDf = pd.DataFrame.from_dict(switchSRGAP2CDict,orient='index').transpose()
    # save to csv
    switchSRGAP2CDf.to_csv("{}_switch_2C.csv".format(status),index=False)
    switchWTDf.to_csv("{}_switch_WT.csv".format(status),index=False)
    # determine maximum number of sessions
    WTSessions      = [int(k) for k in tmsWTDict.keys()]
    SRGAP2CSessions = [int(k) for k in tmsSRGAP2CDict.keys()]
    sessions        = [sess+1 for sess in range(max(max([WTSessions,SRGAP2CSessions], key=max)))]
    # set sessions as x-axis ticks
    ticks = sessions
    # store DFs as lists of lists
    switchWTLicks      = switchWTDf.stack().groupby(level=0).apply(list)  
    switchSRGAP2CLicks = switchSRGAP2CDf.stack().groupby(level=0).apply(list)  
    # plot
    fig, ax   = plt.subplots()
    bpWT      = ax.boxplot(switchWTLicks, positions=np.array(range(len(WTSessions)))*2.0-0.4, widths=0.6, patch_artist=True)
    bpSRGAP2C = ax.boxplot(switchSRGAP2CLicks, positions=np.array(range(len(SRGAP2CSessions)))*2.0+0.4, widths=0.6, patch_artist=True)
    set_box_color(bpWT, 'silver', 0.8)
    set_box_color(bpSRGAP2C, '#035a70', 0.8)
    # fake lines to generate legend
    ax.plot([], c='black', label='WT')
    ax.plot([], c='#035a70', label='SRGAP2C')
    legend_without_duplicate_labels(ax,pos='upper right')
    ax.autoscale()
    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    plt.xlabel('LT Session')
    plt.ylabel('Licks Before Switch')
    plt.title("{}".format(titleStr2))
    #plt.show()
    plt.savefig(os.path.join(os.getcwd(),"{}_2C.svg".format(titleStr)))
    plt.close('all')
    return None 

def makeSwitchCount2CLT(tmsWTDict,tmsSRGAP2CDict,status):
    """Call SwitchCount2CLT function to plot umber of licks before switching reward sides for licktrain sessions.
    Used for SRGAP2C LT data."""
    plots = SwitchCount2CLT(tmsWTDict,tmsSRGAP2CDict,status)
    return None

def Raster2C(tm,genotype,liml=-2000,limr=5000):
    """Plot trial matrix.
    Used for SRGAP2C data."""
    # re-sort trial matrix by stimulus and outcome only
    tm           = tm.copy().sort_values(by=['stim','outcome','raw_time'])
    mouse        = ''.join(tm.mouse.unique())
    date         = ''.join(tm.date.unique())
    # transform trial matrix into dictionaries with
    # (trial,[times]) k,v pairs
    leftDict     = tm.groupby('trial')['left'].apply(list).to_dict()
    rightDict    = tm.groupby('trial')['right'].apply(list).to_dict()
    # form dataframe from dict
    leftDf       = pd.DataFrame.from_dict(leftDict, orient='index')
    rightDf      = pd.DataFrame.from_dict(rightDict, orient='index')  
    # turn dataframe into list of numpy arrays and remove nans
    leftNdarray  = leftDf.to_numpy()
    leftNdarray  = [row[~np.isnan(row)] for row in leftNdarray] 
    rightNdarray = rightDf.to_numpy()
    rightNdarray = [row[~np.isnan(row)] for row in rightNdarray] 
    # sort list of arrays to match original dataframe 
    # this will reverse default Python sorting and restore sorting by stim, outcome, etc.
    loffs                            = tm.trial.unique()
    inds_unsort                      = np.argsort(loffs)
    leftNdarray_unsort               = np.zeros_like(leftNdarray)
    leftNdarray_unsort[inds_unsort]  = leftNdarray
    rightNdarray_unsort              = np.zeros_like(rightNdarray)
    rightNdarray_unsort[inds_unsort] = rightNdarray
    # gather stim change and outcome change indices and unsort them
    stim_chg        = tm.loc[tm.ne(tm.shift()).apply(lambda x: x.index[x].tolist())['stim']]
    stim_chg_unsort = [np.where(loffs == sc.trial)[0][0] for i, sc in stim_chg.iterrows()]
    outc_chg        = tm.loc[tm.ne(tm.shift()).apply(lambda x: x.index[x].tolist())['outcome']]
    outc_chg_unsort = [np.where(loffs == oc.trial)[0][0] for i, oc in outc_chg.iterrows()]
    # plot
    fig, ax = plt.subplots()
    ax.eventplot(leftNdarray_unsort,colors='blue',label='Left')  
    ax.eventplot(rightNdarray_unsort,colors='red',label='Right') 
    legend_without_duplicate_labels(ax, title="n={} trials".format(len(tm.trial.unique())))
    ax.autoscale()
    ax.set_ylim(bottom=-0.5,top=max(len(leftNdarray_unsort),len(rightNdarray_unsort))+0.5)
    ax.set_xlim(left=liml,right=limr)
    ax.get_yaxis().set_ticks([])
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # plot outcome changes as horizontal dashed lines
    for i, oc in enumerate(outc_chg_unsort):
        if oc not in stim_chg_unsort:
            plt.hlines(oc-0.5,xmin=xlim[0],xmax=xlim[1],linestyles='dashed')
    # plot opto groups by color
    # fb11 = plt.fill_between(xlim, y1=[ylim[0],ylim[0]], y2=[stim_chg_unsort[1],stim_chg_unsort[1]], 
    #                 facecolor='yellow', edgecolor='none', lw=1., zorder=0, alpha=1.)
    # fb12 = plt.fill_between(xlim, y1=[ylim[0],ylim[0]], y2=[stim_chg_unsort[1],stim_chg_unsort[1]], 
    #                 facecolor='none', edgecolor='white', hatch="////\\\\\\\\", lw=0., zorder=0)
    # fb21 = plt.fill_between(xlim, y1=[stim_chg_unsort[1],stim_chg_unsort[1]], y2=[stim_chg_unsort[2],stim_chg_unsort[2]], 
    #                 facecolor='yellow', edgecolor='none', lw=1., zorder=0, alpha=0.3)
    # fb31 = plt.fill_between(xlim, y1=[stim_chg_unsort[2],stim_chg_unsort[2]], y2=[stim_chg_unsort[3],stim_chg_unsort[3]], 
    #                 facecolor='chartreuse', edgecolor='none', lw=1., zorder=0, alpha=0.45)
    # fb32 = plt.fill_between(xlim, y1=[stim_chg_unsort[2],stim_chg_unsort[2]], y2=[stim_chg_unsort[3],stim_chg_unsort[3]], 
    #                 facecolor='none', edgecolor='white', hatch="////\\\\\\\\", lw=0., zorder=0)
    # fb41 = plt.fill_between(xlim, y1=[stim_chg_unsort[3],stim_chg_unsort[3]], y2=[ylim[1],ylim[1]], 
    #                 facecolor='chartreuse', edgecolor='none', lw=1., zorder=0, alpha=0.3)
    roughBack   = plt.fill_between(xlim, y1=[stim_chg_unsort[0],stim_chg_unsort[0]], y2=[stim_chg_unsort[1],stim_chg_unsort[1]],
                    facecolor='chartreuse', edgecolor='none', lw=1., zorder=0, alpha=0.3)
    smoothBack  = plt.fill_between(xlim, y1=[stim_chg_unsort[1],stim_chg_unsort[1]], y2=[max(tm.trial.unique()),max(tm.trial.unique())],
                    facecolor='yellow', edgecolor='none', lw=1., zorder=0, alpha=0.3)
    plt.xlabel('Time [ms]')
    plt.ylabel('Trial')
    plt.title("{}: {} ({})".format(genotype,mouse,date))
    #plt.show()
    plt.savefig(os.path.join(os.getcwd(),"{}_raster_{}.svg".format(mouse,date)))
    plt.close('all')
    return None

def makeRaster2C(tmsWTDict,tmsSRGAP2CDict):
    """Call Raster for all sessions for all mice of all genotypes.
    Used for SRGAP2C data."""
    tmsWTListFlat      = flatten_dict_values(tmsWTDict)
    tmsSRGAP2CListFlat = flatten_dict_values(tmsSRGAP2CDict)
    plots = [Raster2C(tm,'WT') for tm in tmsWTListFlat]
    plots = [Raster2C(tm,'SRGAP2C') for tm in tmsSRGAP2CListFlat]
    return None

def RasterNoOpto(tm,plot_type,liml=-2000,limr=5000):
    """Plot trial matrix as raster. No regard for opto status."""
    # re-sort trial matrix by stimulus and outcome only
    tm           = tm.copy().sort_values(by=['stim','outcome','raw_time'])
    mouse        = ''.join(tm.mouse.unique())
    date         = ''.join(tm.date.unique())
    # transform trial matrix into dictionaries with
    # (trial,[times]) k,v pairs
    leftDict     = tm.groupby('trial')['left'].apply(list).to_dict()
    rightDict    = tm.groupby('trial')['right'].apply(list).to_dict()
    # form dataframe from dict
    leftDf       = pd.DataFrame.from_dict(leftDict, orient='index')
    rightDf      = pd.DataFrame.from_dict(rightDict, orient='index')  
    # turn dataframe into list of numpy arrays and remove nans
    leftNdarray  = leftDf.to_numpy()
    leftNdarray  = [row[~np.isnan(row)] for row in leftNdarray] 
    rightNdarray = rightDf.to_numpy()
    rightNdarray = [row[~np.isnan(row)] for row in rightNdarray] 
    # sort list of arrays to match original dataframe 
    # this will reverse default Python sorting and restore sorting by stim, outcome, etc.
    loffs                            = tm.trial.unique()
    inds_unsort                      = np.argsort(loffs)
    leftNdarray_unsort               = np.zeros_like(leftNdarray)
    leftNdarray_unsort[inds_unsort]  = leftNdarray
    rightNdarray_unsort              = np.zeros_like(rightNdarray)
    rightNdarray_unsort[inds_unsort] = rightNdarray
    # gather stim change and outcome change indices and unsort them
    stim_chg        = tm.loc[tm.ne(tm.shift()).apply(lambda x: x.index[x].tolist())['stim']]
    stim_chg_unsort = [np.where(loffs == sc.trial)[0][0] for i, sc in stim_chg.iterrows()]
    outc_chg        = tm.loc[tm.ne(tm.shift()).apply(lambda x: x.index[x].tolist())['outcome']]
    outc_chg_unsort = [np.where(loffs == oc.trial)[0][0] for i, oc in outc_chg.iterrows()]
    # plot
    fig, ax = plt.subplots()
    ax.eventplot(leftNdarray_unsort,colors='blue',label='Left')  
    ax.eventplot(rightNdarray_unsort,colors='red',label='Right') 
    legend_without_duplicate_labels(ax, title="n={} trials".format(len(tm.trial.unique())))
    ax.autoscale()
    ax.set_ylim(bottom=-0.5,top=max(len(leftNdarray_unsort),len(rightNdarray_unsort))+0.5)
    ax.set_xlim(left=liml,right=limr)
    ax.get_yaxis().set_ticks([])
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # plot outcome changes as horizontal dashed lines
    for i, oc in enumerate(outc_chg_unsort):
        if oc not in stim_chg_unsort:
            plt.hlines(oc-0.5,xmin=liml,xmax=limr,linestyles='dashed')
    # plot opto groups by color
    # fb11 = plt.fill_between(xlim, y1=[ylim[0],ylim[0]], y2=[stim_chg_unsort[1],stim_chg_unsort[1]], 
    #                 facecolor='yellow', edgecolor='none', lw=1., zorder=0, alpha=1.)
    # fb12 = plt.fill_between(xlim, y1=[ylim[0],ylim[0]], y2=[stim_chg_unsort[1],stim_chg_unsort[1]], 
    #                 facecolor='none', edgecolor='white', hatch="////\\\\\\\\", lw=0., zorder=0)
    # fb21 = plt.fill_between(xlim, y1=[stim_chg_unsort[1],stim_chg_unsort[1]], y2=[stim_chg_unsort[2],stim_chg_unsort[2]], 
    #                 facecolor='yellow', edgecolor='none', lw=1., zorder=0, alpha=0.3)
    # fb31 = plt.fill_between(xlim, y1=[stim_chg_unsort[2],stim_chg_unsort[2]], y2=[stim_chg_unsort[3],stim_chg_unsort[3]], 
    #                 facecolor='chartreuse', edgecolor='none', lw=1., zorder=0, alpha=0.45)
    # fb32 = plt.fill_between(xlim, y1=[stim_chg_unsort[2],stim_chg_unsort[2]], y2=[stim_chg_unsort[3],stim_chg_unsort[3]], 
    #                 facecolor='none', edgecolor='white', hatch="////\\\\\\\\", lw=0., zorder=0)
    # fb41 = plt.fill_between(xlim, y1=[stim_chg_unsort[3],stim_chg_unsort[3]], y2=[ylim[1],ylim[1]], 
    #                 facecolor='chartreuse', edgecolor='none', lw=1., zorder=0, alpha=0.3)
    roughBack   = plt.fill_between(xlim, y1=[stim_chg_unsort[0],stim_chg_unsort[0]], y2=[stim_chg_unsort[1],stim_chg_unsort[1]],
                    facecolor='chartreuse', edgecolor='none', lw=1., zorder=0, alpha=0.3)
    smoothBack  = plt.fill_between(xlim, y1=[stim_chg_unsort[1],stim_chg_unsort[1]], y2=[max(tm.trial.unique()),max(tm.trial.unique())],
                    facecolor='yellow', edgecolor='none', lw=1., zorder=0, alpha=0.3)
    plt.xlabel('Time [ms]')
    plt.ylabel('Trial')
    plt.title("{}: {} ({})".format(plot_type,mouse,date))
    #plt.show()
    plt.savefig(os.path.join(os.getcwd(),"{}_raster_{}.svg".format(mouse,date)))
    plt.close('all')
    return None

def makeRasterNoOpto(tmsDict, plot_type='NO OPTO'):
    """Call RasterNoOpto for all sessions for all mice.
    Disregards opto status."""
    tmsListFlat = flatten_dict_values(tmsDict)
    plots = [RasterNoOpto(tm,plot_type) for tm in tmsListFlat]
    return None

def makeAll(csv_file,beh):
    """Extracts filenames from csv of format columns=['genotype','mouse','filepath']
    and generates all relevant plots"""
    # for all plots except naive versus expert
    if beh != 'naive_expert':
        print('Generating {} trial matrices...'.format(beh))
        cols  = ['genotype','mouse','file']
        fisDf = pd.read_csv(csv_file,names=cols)
        fisDf = fisDf.dropna(axis=0, how='any')
        fisDf = fisDf.drop(0).reset_index() 
        # create dictionary of files with (mouse,[files]) (k,v) pairs
        fisDict = fisDf.groupby('mouse')['file'].apply(list).to_dict()  
        # create dictionary of genotypes with (mouse,genotype) (k,v) pairs
        genesDict = fisDf.groupby('mouse')['genotype'].apply(list).to_dict() 
        genesDict = {k: v[0] for k,v in genesDict.items()}
        # create dictionary of tms with (genotype, [tms]) (k,v) pairs
        # for each mouse, make TrialMatrix for each file and a concatenated TrialMatrix
        # append concatenated TrialMatrix to dotms at appropriate key
        tmsDict = defaultdict(list)
        _       = [tmsDict[genesDict[mouse]].append(makeTrialMatrix(fis, opto=3)) for mouse,fis in fisDict.items()]
        # concatenate to single df for each transgene
        # add trials to previous maximum trial number to ensure no trial number overlap
        # make copies of all tms to modify without pointing to same object
        tmsDictConcat = {}
        for genotype,tmList in tmsDict.items():
            max_trial  = 0
            tmCopyList = []
            for tm in tmList:
                tmCopy              = tm.copy()
                tmCopy.trial        = tmCopy.trial.apply(lambda x: x + max_trial)
                max_trial           = tmCopy.trial.max() + 1
                tmCopyList.append(tmCopy)
            direction               = [False, True, True, True, True]
            tmConcat                = sortTM(pd.concat(tmCopyList).reset_index(drop=True),direction,'opto','stim','outcome','mouse','trial')
            tmsDictConcat[genotype] = tmConcat.reset_index(drop=True)
        # generate concatenated mouse and whole genotype congruent-lick plots
        print('Generating {} congruent line plots...'.format(beh))
        #makeCongruentLine(tmsDict,beh,per_mouse=True)
        # generate concatenated mouse and whole genotype correct-lick plots
        print('Generating {} correct line plots...'.format(beh))
        #makeCorrectLine(tmsDict,beh,per_mouse=True)
        # generate concatenated whole genotype congruent-lick plots with adjusted statistics
        print('Generating {} congruent line plots by stimulus...'.format(beh))
        makeCongruentLine2(tmsDict,beh)    
        # generate concatenated whole genotype correct-lick plots with adjusted statistics
        print('Generating {} correct line plots by stimulus...'.format(beh))
        makeCorrectLine2(tmsDict,beh)
        # generate single-value behavioral characterization plot for all mice together
        print('Generating {} first lick histogram and arrow plots...'.format(beh))
        makeFirstLickPlots(flatten_dict_values(tmsDict),beh,genesDict)
        # generate lick latency plot for all mice together
        print('Generating {} lick latency plots...'.format(beh))
        makeFirstCongruentLickLatencyPlots(flatten_dict_values(tmsDict),beh,genesDict)        
        # generate whole genotype post-laser trial correct-lick plots
        print('Generating {} n plus one plots...'.format(beh))
        makeLicksAfterLaserPlots(tmsDict,beh)
        # generate whole genotype lick frequency plots by stimulus (line) and by side (histo)
        print('Generating {} frequency plots by stimulus...'.format(beh))
        makeLickFrequencyPlots(tmsDict,beh)
    else:
        print('Generating {} trial matrices...'.format(beh))
        cols  = ['mouse','naive','expert']
        fisDf = pd.read_csv(csv_file,names=cols)
        fisDf = fisDf.dropna(axis=0, how='any')
        fisDf = fisDf.drop(0).reset_index() 
        # create list of TMs for each learning status (naive, expert) and add to dict
        # for each mouse, make TrialMatrix for each file
        # append TrialMatrix to list of appropriate type (naive, expert)
        naiveTMs  = []
        _         = [naiveTMs.append(makeTrialMatrix([fi],opto=3)) for fi in fisDf.naive]
        expertTMs = []
        _         = [expertTMs.append(makeTrialMatrix([fi],opto=3)) for fi in fisDf.expert]
        # create dictionary of tms with (status, [tms]) (k,v) pairs
        tmsDict = {'naive': naiveTMs, 'expert': expertTMs}
        # generate genotype-nonspecific lick frequency plots by stimulus
        print('Generating {} frequency plots by stimulus...'.format(beh))
        makeNaiveExpertFrequencyPlots(tmsDict,beh)
        # generate genotype-nonspecific lick frequency plots by stimulus
        print('Generating {} correct plots by stimulus...'.format(beh))
        makeNaiveExpertCorrectPlots(tmsDict,beh)
    plotsMade = True
    return plotsMade
    
def makeAllLesion(csv_file,beh):
    """Extracts filenames from csv of format columns=['genotype','mouse','filepath']
    and generates all relevant plots. Used for lesion data."""
    print('Generating {} trial matrices...'.format(beh))
    cols  = ['date','mouse','file']
    fisDf = pd.read_csv(csv_file,names=cols)
    fisDf = fisDf.dropna(axis=0, how='any')
    fisDf = fisDf.drop(0).reset_index() 
    # create dictionary of files with (mouse,[files]) (k,v) pairs
    fisDict = fisDf.groupby('date')['file'].apply(list).to_dict()
    # create dictionary of tms with (day, [tms]) (k,v) pairs
    # for each mouse, make TrialMatrix for each day
    tmsDict = defaultdict(list)
    _       = {tmsDict[day].extend([makeTrialMatrix([fi]) for fi in fis]) for day,fis in fisDict.items()}
    # generate concatenated day-by-day congruent-lick plots by stimulus
    print('Generating {} congruent line plots by stimulus...'.format(beh))
    makeCongruentLine2Lesion(tmsDict,beh)    
    # generate concatenated day-by-day correct-lick plots by stimulus
    print('Generating {} correct line plots by stimulus...'.format(beh))
    makeCorrectLine2Lesion(tmsDict,beh)
    # create dictionary of tms with (day, [tms]) (k,v) pairs
    # for each mouse, make TrialMatrix for each day
    tmsDictLikeOpto = defaultdict(list)
    [tmsDictLikeOpto['pre'].append(makeTrialMatrix(fis)) for fis in (list(map(list,zip(fisDict[-2],fisDict[-1]))))]
    tmsDictLikeOpto['post'].extend([makeTrialMatrix([fi]) for fi in fisDict[1]])
    # generate concatenated 'opto-like' correct line plots by stimulus
    print('Generating {} opto-like correct line plots by stimulus...'.format(beh))
    makeCorrectLineLikeOpto(tmsDictLikeOpto,beh)
    # generate concatenated 'opto-like' frequency line plots by stimulus
    print('Generating {} opto-like frequency line plots by stimulus...'.format(beh))
    makeFreqLineLikeOpto(tmsDictLikeOpto,beh)
    plotsMade = True
    return plotsMade

def makeAll2C(csv_file,status):
    """Extracts filenames from csvs for each status type
    and generates all relevant plots. Used for SRGAP2C data."""
    if status is not 'LT' and status is not 'all':
        print('Generating {} trial matrices...'.format(status))
        cols  = ['genotype','mouse','file']
        fisDf = pd.read_csv(csv_file,names=cols)
        fisDf = fisDf.dropna(axis=0, how='any')
        fisDf = fisDf.drop(0)
        # create dictionary of files with (mouse,[files]) (k,v) pairs
        fisDict = fisDf.groupby('genotype')['file'].apply(list).to_dict()
        # create dictionary of tms with (day, [tms]) (k,v) pairs
        # for each mouse, make TrialMatrix for each day
        # if FA, do not ignore non-random trials
        tmsDict = defaultdict(list)
        _       =      {tmsDict[genotype].extend([makeTrialMatrix([fi]) for fi in fis]) for genotype,fis in fisDict.items()} if 'FA' not in status \
                  else {tmsDict[genotype].extend([makeTrialMatrix([fi],FA=True) for fi in fis]) for genotype,fis in fisDict.items()}
        # generate concatenated whole genotype correct-lick plots by stimulus
        print('Generating {} correct line plots by stimulus...'.format(status))
        makeCorrectLineByStim2C(tmsDict,status)
        # generate concatenated whole genotype correct-lick plots
        print('Generating {} correct line plots...'.format(status))
        makeCorrectLine2C(tmsDict,status)
        # generate whole genotype lick frequency plots by genotype by stimulus
        print('Generating {} frequency plots by genotype by stimulus...'.format(status))
        makeLickFrequencyPlotsByStim2C(tmsDict,status)
        # generate whole genotype lick frequency plots by genotype
        print('Generating {} frequency plots by genotype...'.format(status))
        makeLickFrequencyPlots2C(tmsDict,status)
        plotsMade = True
    elif status is 'LT':
        print('Generating {} trial matrices...'.format(status))
        cols  = ['date','genotype','mouse','file']
        fisDf = pd.read_csv(csv_file,names=cols)
        fisDf = fisDf.dropna(axis=0, how='any')
        fisDf = fisDf.drop(0).reset_index() 
        # create dictionary of files with (mouse,[files]) (k,v) pairs for each genotype
        WTDict      = fisDf.groupby('genotype').get_group('WT').groupby('date')['file'].apply(list).to_dict()
        SRGAP2CDict = fisDf.groupby('genotype').get_group('SRGAP2C').groupby('date')['file'].apply(list).to_dict()
        # create dictionary of tms with (day, [tms]) (k,v) pairs for each genotype
        # for each mouse, make TrialMatrix for each day
        tmsWTDict      = defaultdict(list)
        _              = {tmsWTDict[day].extend([makeTrialMatrix([fi],LT=True) for fi in fis]) for day,fis in WTDict.items()}
        tmsSRGAP2CDict = defaultdict(list)
        _              = {tmsSRGAP2CDict[day].extend([makeTrialMatrix([fi],LT=True) for fi in fis]) for day,fis in SRGAP2CDict.items()}
        # generate whole genotype side switch latency plot by day by genotype
        print('Generating side switch latency plots by genotype...')
        makeSwitchLatency2CLT(tmsWTDict,tmsSRGAP2CDict,status)
        # generate whole genotype side switch lick count plot by day by genotype
        print('Generating side switch lick count plots by genotype...')
        makeSwitchCount2CLT(tmsWTDict,tmsSRGAP2CDict,status)
        plotsMade = True
    elif status is 'all':
        print('Generating {} trial matrices...'.format(status))
        cols           = ['session','WT_mouse','WT_file','SRGAP2C_mouse','SRGAP2C_file']
        fisDf          = pd.read_csv(csv_file,names=cols).drop(0)
        WTDf           = fisDf[['session','WT_mouse','WT_file']].dropna(axis=0,how='any')
        SRGAP2CDf      = fisDf[['session','SRGAP2C_mouse','SRGAP2C_file']].dropna(axis=0,how='any')
        WTDict         = WTDf.groupby('session')['WT_file'].apply(list).to_dict()
        tmsWTDict      = defaultdict(list)
        _              = {tmsWTDict[session].extend([makeTrialMatrix([fi]) for fi in fis]) for session,fis in WTDict.items()}
        SRGAP2CDict    = SRGAP2CDf.groupby('session')['SRGAP2C_file'].apply(list).to_dict()
        tmsSRGAP2CDict = defaultdict(list)
        _              = {tmsSRGAP2CDict[session].extend([makeTrialMatrix([fi]) for fi in fis]) for session,fis in SRGAP2CDict.items()}
        # generate whole genotype lick count plot by genotype
        print('Generating lick count plots by genotype...')
        makeLickCount2C(tmsWTDict,tmsSRGAP2CDict,status)
        # generate lick raster for all mice
        print('Generating lick rasters by mouse...')
        makeRaster2C(tmsWTDict,tmsSRGAP2CDict)
        plotsMade = True
    return plotsMade

def makeAllNoOpto(csv_file, plot_type='NO OPTO'):
    """Extracts filenames from csv of format columns=['genotype','mouse','filepath']
    and generates raster plots for each ardulines file plots. Used for data without
    regard for opto status."""
    print('Generating {} trial matrices...'.format(plot_type))
    cols  = ['genotype','mouse','file']
    fisDf = pd.read_csv(csv_file,names=cols)
    fisDf = fisDf.dropna(axis=0, how='any')
    fisDf = fisDf.drop(0).reset_index() 
    # create dictionary of files with (mouse,[files]) (k,v) pairs
    fisDict = fisDf.groupby('mouse')['file'].apply(list).to_dict()  
    # create dictionary of genotypes with (mouse,genotype) (k,v) pairs
    genesDict = fisDf.groupby('mouse')['genotype'].apply(list).to_dict() 
    genesDict = {k: v[0] for k,v in genesDict.items()}
    # create dictionary of tms with (mouse, [tms]) (k,v) pairs
    # for each mouse, make TrialMatrix for each day
    tmsDict = defaultdict(list)
    _       = {tmsDict[mouse].extend([makeTrialMatrix([fi]) for fi in fis]) for mouse,fis in fisDict.items()} 
    # generate raster plots for each file
    print('Generating {} raster plots by session...'.format(plot_type))
    makeRasterNoOpto(tmsDict)
    plotsMade = True
    return plotsMade

def run(csv_files,behaviors=['disc','det','naive_expert']):
    """Highest level function. Makes all plots for each behavior.
    List of csv files must be equal length to list of behaviors to analyze."""
    plotsMade = [makeAll(csv,beh) for csv,beh in zip(csv_files,behaviors)]

def runLesion(csv_files,behaviors=['disc','det']):
    """Highest level function. Makes all plots for each behavior.
    List of csv files must be equal length to list of behaviors to analyze.
    Used for lesion data."""
    plotsMade = [makeAllLesion(csv,beh) for csv,beh in zip(csv_files,behaviors)]

def run2C(csv_files,status=['naive_FA','expert_FA', 'naive','expert','LT','all']):
    """Highest level function. Makes all plots for each learning status.
    List of csv files must be equal length to list of behaviors to analyze.
    Used for SRGAP2C data."""
    plotsMade = [makeAll2C(csv,stat) for csv,stat in zip(csv_files,status)]

def runNoOpto(csv_file):
    """Highest level function. Makes raster plots for individual 
    sessions for individual mice. Accepts single csv filke as input.
    Used for non-opto-related data."""
    plotsMade = makeAllNoOpto(csv_file)