"""
In this script we test whether case assignments are random.
"""

import numpy   as np 
import pandas  as pd
import seaborn as sb

from tqdm       import tqdm
from matplotlib import pyplot as plt
from itertools  import product
from joblib     import Parallel, delayed, cpu_count

from routines   import get_project_folder, load_raw_features, false_discovery_correction, get_pval, create_subplots


def examine_case_assignment_bias( variable='case_type', confidence_th=0.4, retvals=False ):
    """
    Analyze bias in case assignment by running binomial tests either with respect to 'case_type' or 'entity_label'. Only
    consider cases with a classification confidence above confidence_th and if retvals is True, return the raw p-values, 
    else visualize the result and return nothing. 
    """

    assert variable in ['case_type','entity_label'], 'Variable must be either case_type or entity_label'

    # Load info about judge, decision date, circuit and assignment variable of interest.
    ####################################################################################################################
    cases =  load_raw_features( confidence_th )                             # load the features
    vars  = ['judge_id','decision_date','circuit' ] + [variable]            # relevant columns    
    cases =  cases.loc[:,vars]                                              # extract relevant columns
    cases =  cases[ cases[variable] != 'unsure' ]                           # not interesting for this analysis

    # For fixed variable, circuit and decade: Calculate for each judge the p-value that the assigned number of cases of
    # this case deviates from the baseline. 
    ####################################################################################################################
    tg    =  pd.Grouper(key='decision_date', freq='10YE')       # group by decade
    pvals =  []                                                 # store p-values

    for (circuit,epoch), sdf in cases.groupby(['circuit',tg]): # iterate each circuit and epoch

        for subset, ssdf in sdf.groupby(variable):             # iterate each subset of cases

            if len(ssdf) < 100: continue                       # not enough cases to be representative

            bl = len(ssdf) / len(sdf)  # baseline: fraction of this type for this (circuit,epoch)

            for judge, df in ssdf.groupby('judge_id'): # iterate each judge for this (circuit,epoch) and this subset

                nc   = len(df) # number of cases of this type for this judge in this (circuit,epoch)
                totc = len( sdf[ sdf['judge_id']==judge ] ) # tot number of cases for this judge in this (circuit,epoch)
                pval = get_pval( n_succ=nc, n_tot=totc, baseline=bl )
                pvals += [ {'circuit':circuit, 'epoch':epoch, 'subset':subset, 'judge':judge, 
                            'baseline':bl, 'nr_cases':nc, 'total_nr_cases':totc, 'pval':pval, 
                            } ]

    pvals = pd.DataFrame(pvals)

    if retvals: return pvals

    # If H0 holds True, then the p-values are uniformly distributed. This is visualized by means of QQ-plots based on
    # data calculated below. Here, we simply correct for multiple testing and count the fraction of values below the 
    # 5% threshold. 
    ####################################################################################################################
    info = false_discovery_correction( pvals['pval'] )

    # We expect that empirically, a fraction of p% judges is below a p-value of p%. Here, we calculate this fraction
    # and we always subsample the judges to get error bars. In a sense, this is akin to a QQ-plot.
    ####################################################################################################################
    ps   =  np.linspace( 0.05, 0.95, 19 )                                               # p-values to check
    em   =  []                                                                          # store empirical fractions

    for (circuit, epoch, subset), sdf in pvals.groupby(['circuit','epoch','subset']):   # iterate each constellation

        judges =   sdf[ 'judge' ].unique()                                              # all judges
        nc     =   len(judges)                                                          # nr of judges
        loj    = [ np.random.choice( judges, size=nc//2)  for _ in range(50)  ]         # subselect some
        ndfs   = [ sdf[ sdf['judge'].isin(j) ] for j in loj ]                           # sub-frame for judges

        for p in ps:                                                                    # iterate each p-value
            
            fixed  =   {'subset':subset, 'epoch':epoch, 'circuit':circuit,'p':p}        # fixed values
            fracs   = [ len(ndf[ ndf['pval'] <= p ]) / len(ndf) for ndf in ndfs ]       # empirical fraction below p
            em     += [ {**fixed, 'frac':f} for f in fracs ]                            # append to list

    em = pd.DataFrame( em )                                                             # make DataFrame

    # Write results to the output.
    ####################################################################################################################
    dir = get_project_folder(f'data/case_assignment_bias/by_{variable}/', create=True )
    _   = pvals.to_parquet(  f'{dir}p-values.parquet' )
    _   = em.to_parquet(     f'{dir}empirical_thresholds.parquet' )
    with open(f'{dir}p-value_corrections.txt', 'w') as f: print(info, file=f)

    # Create the plotting figure and color scheme. 
    ####################################################################################################################
    _        = sb.set_style('whitegrid')
    nplots   = len(em['subset'].unique())
    fig, axs = create_subplots( nplots=nplots, n_cols=nplots//2, width=8, height=7, sharex='all', sharey='all' )
    fs       = 13

    # Iterate each circuit, and visualize empirical ratios per subset of cases.
    ####################################################################################################################
    for i, (subset, sdf) in enumerate( em.groupby('subset') ):

        ax = axs.flatten()[i]
        _  = ax.plot( [0,0.95], [0,0.95], linestyle=':', linewidth=4, color='black' )        

        for j, (circuit, ssdf) in enumerate( sdf.groupby('circuit') ):

            l   = ssdf.groupby('p')['frac'].describe()
            c   = ssdf['p'].corr(ssdf['frac'])
            crc = 'D.C.' if circuit==0 else circuit # nicer formatting
            oc  = sb.color_palette('deep',12)
            x   = l.index+np.random.normal(scale=0.007, size=len(l)) # add jitter to x-values for visibility
            _   = ax.errorbar(  x               = x, 
                                y               = l['mean'],
                                yerr            = l['std'],
                                fmt             =  'o',
                                linestyle       =  'none',
                                color           =  oc[j], 
                                markerfacecolor =  oc[j], 
                                ecolor          =  oc[j],
                                markersize      =  5,
                                linewidth       =  2,
                                elinewidth      =  3,
                                capsize         =  0,
                                alpha           =  0.5,
                                label           =  f'{crc} ({c:.2f})',
                                )
        ax.set_xlim(0.04, 0.96)
        ax.set_title( f"{variable.replace('_',' ')}: {subset.replace('_',' ')}", fontsize=fs )
        ax.legend( ncol=3, fontsize=fs-1, title='circuit',                       framealpha=0.5 )
        if i%3==0: ax.set_ylabel( 'empirical fraction',                          fontsize=fs )
        if i>2:    ax.set_xlabel( 'p-value',                                     fontsize=fs )
            
    _    =  fig.savefig(f'{dir}{variable}_bias_examination.pdf', bbox_inches='tight')
    _    =  plt.clf()


if __name__=='__main__':

    examine_case_assignment_bias( variable='case_type' )
    examine_case_assignment_bias( variable='entity_label' )