"""
Visualizations for the case outcome predictions. 
"""

import numpy    as np 
import pandas   as pd
import seaborn  as sb

from matplotlib            import pyplot as plt
from sklearn.decomposition import NMF
from scipy.stats           import binom

from routines                import get_project_folder, get_case_cols, get_pval, visualize_accuracy_per_signal_strength
from routines                import figure_annotations, load_raw_features, get_case_types
from case_classifications    import load_all_case_classifications, classify_one_feature_constellation    


def visualize_win_rates(  ):
    """
    Create a plot that visualizes the plaintiff win rates and p-values. 
    """

    # Load the data.
    ####################################################################################################################
    df  =   load_raw_features( confidence_th=0.4 )

    # Calculate judge win rate and associated p-values.
    ####################################################################################################################
    stats                = df.groupby('judge_id')['judgement'].describe(  )                   # stats per judge    
    stats                = stats[ stats['count'] > 20 ]                                       # min amount of cases
    bl                   = df['judgement'].mean()                                             # average win rate 
    calc_pval            = lambda x: get_pval( int(x['mean']*x['count']), int(x['count']),bl) # fix baseline arg
    stats['pval']        = stats.apply( calc_pval, axis=1 )                                   # get the p-value
    stats['significant'] = [ True if p <= 0.10 else False for p in stats['pval'] ]            # whether sign or not
    sig_mass             = stats['significant'].mean()                                        # frac of sig values

    # Create figures and other plotting arguments.
    ####################################################################################################################
    fig, ax  = plt.subplots( figsize=(8,6) )
    fs      =  14
    ax_in   =  ax.inset_axes( [0.60, 0.60, 0.35, 0.30] )  # [x0, y0, width, height]
    fixed   = {'hue':'significant', 'data':stats, 'palette':'coolwarm', 'stat':'density'}    

    # Plot distribution of win rates.
    ####################################################################################################################
    fixed  = {'hue':'significant', 'data':stats, 'palette':'coolwarm', 'stat':'density'}

    sb.histplot(x='mean', ax=ax, element='step', fill=True, alpha=0.5, linewidth=0, **fixed, legend=True  )
    ax.axvline(x=bl, color='blabck', linestyle='--', label=f'baseline: {100*bl:.1f}%')
    ax.set_ylabel('relative frequency',           fontsize=fs )
    ax.set_xlim(0,1)    

    # Plot distribution of the null model (i.e. sample win rates from baseline).
    ####################################################################################################################
    nulls = [ binom.rvs( n=int(n), p=bl  ) / n for n in stats['count'] ] # sample for each judge
    _     = sb.kdeplot( nulls, linestyle=':', color='grey', linewidth=2, label='null model', ax=ax )

    ax.legend(loc='lower right', frameon=False )
    ax.set_title('plaintiff win rate per judge', fontsize=fs)

    # Plot distribution of p-values.
    ####################################################################################################################
    sb.histplot( x='pval', fill=True, bins=40, ax=ax_in , linewidth=0.5, **fixed )
    ax_in.text( x=0.1, y=5, s=f'significant mass:{100*sig_mass:.0f}%', fontsize=fs-5)
    ax_in.set_xlabel('p-value',             fontsize=fs-3)
    ax_in.set_ylabel('relative\nfrequency', fontsize=fs-4)
    ax_in.set_xlim(0,1)
    ax_in.get_legend().remove()
    ax_in.grid(False)

    # Write to the output.
    ####################################################################################################################
    dir = get_project_folder(f'data/features/', create=True)
    _   = fig.savefig(f'{dir}distribution_of_win_rates.pdf', bbox_inches='tight')


def visualize_feature_distributions( ):
    """
    Create overview plots of all features.
    """

    # Load the data.
    ####################################################################################################################
    df = load_raw_features( confidence_th=0.4 ).reset_index()

    # create the plotting window
    ####################################################################################################################
    height    =  16
    width     =  22
    fig, axes =  plt.subplots( nrows=2, ncols=3, figsize=(width, height) )
    _         =  fig.subplots_adjust( hspace=0.35, wspace=0.25 )
    fs        =  13
    axs       =  []

    # Plot prior win rate. 
    ####################################################################################################################
    ax   =   axes[0,0]
    axs += [ ax ] 
    _    = sb.kdeplot( df['win_rate'].values, ax=ax, color='ForestGreen', fill=True, linewidth=0 )
    _    = ax.set_xlim( df['win_rate'].min(),  df['win_rate'].max() )
    _    = ax.set_title('prior plaintiff win-rate', fontsize=fs )
    _    = ax.set_xlabel(' ',                       fontsize=fs )
    _    = ax.set_ylabel('relative frequency',      fontsize=fs )        

    # Plot binary judge features.
    ####################################################################################################################
    binary           = ['party_republican', 'promoted', 'gender_male']                     # binary features
    labels         = ['Republican', 'promoted', 'male', 'ABA rating?']                   # nice lables
    sdf            = df.loc[:,binary]                                                    # restrict to binary features
    sdf            = sdf.stack().reset_index()                                           # unstack
    sdf            = sdf.rename(columns={'level_1':'feature',0:'value'})                 # rename
    replace        = dict([(value,label) for (value,label) in zip(binary,labels)])       # for nice labels
    sdf['feature'] = sdf['feature'].replace(replace)                                     # make nice labels
    counts         = sdf.groupby('feature')['value'].value_counts(normalize=True)        # count each binary feature
    counts         = counts.rename('relative count').reset_index()                       # make into frame

    ax      =   axes[0,1]
    axs    += [ ax ]  
    _       = ax.set_title('binary judge features', fontsize=fs, color='black')
    _       = sb.barplot( data=counts, x='value', y='relative count', hue='feature', palette='Set2', 
                          alpha=0.8,   width=0.45, ax=ax )
    _       = ax.set_xticklabels(['False','True'])
    _       = ax.set_xlabel('')
    _       = ax.legend(ncol=2, frameon=False, loc='upper left', fontsize=fs-4 )    
    _       = ax.set_ylim(top=1.2)

    # Plot experience and workload.
    ####################################################################################################################
    ax       =   axes[0,2]
    axs     += [ ax ] 
    pfeats   = [ 'experience', 'workload',  ]
    colors   =  sb.color_palette('colorblind', 2)

    for feat,col in zip( pfeats, colors ):
        sb.kdeplot( data=df.loc[:,feat], color=col, fill=True, linewidth=0, alpha=0.5, label=feat, ax=ax )

    ax.set_xlim(left=0, right=75)
    ax.legend(loc='best', frameon=False)
    ax.set_xlabel(' ',                      fontsize=fs )
    ax.set_ylabel('relative count',         fontsize=fs )
    ax.set_title('judge experience and workload', fontsize=fs )

    # Plot file date.
    ####################################################################################################################
    ax   =   axes[1,0]
    axs += [ ax ]     
    _    = sb.histplot( df['decision_date'], element='step', fill=True, stat='frequency', linewidth=0, ax=ax )
    _    = ax.set_xlim( df['decision_date'].min(),  df['decision_date'].max() )
    _    = ax.set_title('case decision date',  fontsize=fs )
    _    = ax.set_xlabel(' ',                  fontsize=fs )
    _    = ax.set_ylabel('relative frequency', fontsize=fs )

    # Plot the cases pie chart.
    ####################################################################################################################
    text_col  =  'black'
    ax        =   axes[1,1]
    axs       += [ax]
    _         = ax.set_title('case type', fontsize=fs, color=text_col)

    cases     = df['case_type'].value_counts(normalize=True)                    # count the cases
    case_lbls = [ label.replace('_','\n') for label,frac in cases.items() ]     # case labels

    pt, txt, pcts  = ax.pie(cases,
                            labels        =   case_lbls,
                            startangle    =   70,
                            autopct       =  '%1.0f%%',
                            pctdistance   =   0.8,
                            labeldistance =   1.1,
                            wedgeprops    = {'linewidth':2, 'edgecolor':'white'},
                            textprops     = {'color':text_col, 'fontsize':fs}
                          )

    _ = plt.setp( pcts, color='white', fontweight='bold', fontsize=fs-3 )

    palette = get_case_cols()
    for i, patch in enumerate(pt):
        label = patch.get_label()
        label = label.replace(' ','_').replace('\n','_')
        _     = patch.set_color( palette[label] )       
        _     = patch.set_edgecolor('white')
        _     = patch.set_linewidth(2)             

    # Plot the circuit pie chart.
    ####################################################################################################################
    ax           =  axes[1,2]
    axs         += [ax]
    _            = ax.set_title('circuits', fontsize=fs, color=text_col)

    circuits     = df['circuit'].dropna().astype(int)                           # make integer
    circuits     = circuits.value_counts(normalize=True).sort_index()           # count the circuits
    circuit_lbls = [ f'{label}' for label,frac in circuits.items() ]            # circuit names
    circuit_lbls = [ 'D.C.' if l=='0' else l for l in circuit_lbls ]            # special case
    ns           = len(circuit_lbls)                                            # number of circuits/pie slices
    circuit_cols = sb.color_palette('crest', ns)                                # gradual color for circuits
 
    pt, txt, pcts  = ax.pie(circuits,                                           # plot the pie-chart
                            labels        =   circuit_lbls ,
                            colors        =   circuit_cols,
                            startangle    =   45,
                            autopct       =  '%1.0f%%',
                            pctdistance   =   0.8,
                            labeldistance =   1.1,
                            wedgeprops    = {'linewidth':2, 'edgecolor':'white'},
                            textprops     = {'color':text_col, 'fontsize':fs}
                          )

    _            = plt.setp( pcts, color='white', fontweight='bold', fontsize=fs-3 )   
        
    # Annotate the plots and write to the output.
    ####################################################################################################################
    _   = figure_annotations( axs, xy=(0.0,1.08), fontsize=14, ha='right', va='bottom' )
    dir = get_project_folder(f'data/features/', create=True)
    _   = fig.savefig(f'{dir}feature_overview.pdf', bbox_inches='tight', dpi=600 )
    _   = plt.clf()
    _   = plt.close()


def visualize_biographic_feature_classification_results( ):
    """
    We visualize the prediction accuracy based on biographic judge features.
    """

    predictions = load_all_case_classifications(method        =  'GB',
                                                feat          =  'bio',
                                                split         =  'generic',
                                                confidence_th =   0.4 )  

    fig, ax =  plt.subplots( figsize=(8,6) )
    _       =  visualize_accuracy_per_signal_strength( df=predictions, ax=ax, legend_outside=False)
    dir     =  get_project_folder(f'data/case_outcome_predictions/bio_based/', create=True)
    _       =  ax.set_title('')
    _       =  fig.savefig( f'{dir}biographic_feature_classification_results.pdf', bbox_inches='tight' )


def visualize_citation_feature_classification_results( ):
    """
    We visualize the prediction accuracy based on citation features along with some visualizations of the NMF concept.
    """

    # Load the features.
    ####################################################################################################################
    df    = load_raw_features( confidence_th=0.4 )  
    df    = df.reset_index().set_index(['judge_id','CAP_ID','date']).dropna()

    # Restrict to top 40 cases.
    ####################################################################################################################
    cite_df         =   df.drop('decision_date',axis=1, errors='ignore').reset_index()
    cite_cols       = [ c for c in df.columns if 'cite_count_case_' in c ]
    cite_cols       =   cite_cols[:40] # show only most cited cases
    cite_df         =   df.loc[:,cite_cols]
    cite_df         =   cite_df.reset_index(['CAP_ID','date'], drop=True)
    cite_df         =   cite_df[ ~cite_df.index.duplicated(keep='last') ] 

    # Randomly select some judges, but proportional to how often they cite.
    ####################################################################################################################
    judges          =   cite_df.index
    probas          =   cite_df.sum(axis=1)
    some_judges     =   np.random.choice( judges, size=20, p=probas/probas.sum()  )
    cite_df         =   cite_df.loc[ some_judges, : ]
    cite_df.index   = [ f'judge ID {i}' for i in cite_df.index ]  
    cite_df.columns = [ c.split('cite_count_case_')[1].zfill(7) for c in cite_df.columns ] 
    cite_df.columns = [ f'case nr. {c}' for c in cite_df.columns ]          

    # Calculate a non-negative matrix factorization. Just like described in [1], here we first subsample data per case 
    # to make sure the embedding is representative across case types. 
    # [1] 2021 - Yang et al. - Identifying latent activity behaviors and lifestyles.
    ####################################################################################################################
    X          = df.groupby('case_type').sample(10**3, replace=True)           # like in [1]
    X          = X.loc[ :, cite_cols ]                                         # restrict to relevant cases
    nc         = 3                                                             # embedding dimension
    model      = NMF( n_components=nc, init='random',                          # intiialize class
                      random_state=0, max_iter=2000 )                          
    _          = model.fit(X)                                                  # X ~ W*H
    W          = model.transform( X )                                          # transform
    H          = model.components_
    H          = pd.DataFrame(H, index=range(1,nc+1) )        

    # Load the NMF based classification results.
    ####################################################################################################################
    clfs    =  load_all_case_classifications( feat='cite_count', split='generic', confidence_th=0.4 )

    # Create the plotting window.
    ####################################################################################################################
    height         = 8
    width          = 22
    fig            = plt.figure(figsize=(width, height))
    gs             = plt.GridSpec( nrows=20, ncols=2,  )
    _              = fig.subplots_adjust( hspace=0.7, wspace=0.13 )
    fs             = 13
    axs            = []    

    # Plot heatmap of cited cases.
    ####################################################################################################################
    ax              = fig.add_subplot( gs[:13,0] )
    axs            += [ax]
    vmax            =   0.03
    ticks           = [ 0,    0.01,  0.02,  0.03  ]
    labels          = ['0%',  '1%',  '2%',  '3%'  ]
    _               = sb.heatmap(cite_df,  cmap='flare',  ax=ax,  vmax=vmax, 
                                    cbar_kws={'extend':'max', 'location': 'top' }
                                    )
    _               = ax.figure.axes[-1].set_xlabel('relative citation count', size=fs)    
    c_bar           = ax.collections[0].colorbar
    _               = c_bar.set_ticks(ticks)
    _               = c_bar.set_ticklabels(labels)       

    # Plot heatmap of NMF embedding.
    ####################################################################################################################
    ax       = fig.add_subplot( gs[17:,0] )
    axs     += [ax]
    fs       = 14
    vmax     =  0.05
    ticks    = [ 0,    0.01,  0.02,  0.03,  0.04, 0.05 ]
    labels   = ['0%',  '1%',  '2%',  '3%',  '4%', '5%' ]
    _        = sb.heatmap(H, cmap='Blues', ax=ax, vmax=vmax, cbar_kws={'extend':'max','location':'bottom','shrink':0.5})
    _        = ax.figure.axes[-1].set_xlabel('weight', size=fs)    
    c_bar    = ax.collections[0].colorbar
    _        = c_bar.set_ticks(ticks)
    _        = c_bar.set_ticklabels(labels)   
    _        = ax.set_ylabel('embedding\ndimension', fontsize=fs)
    _        = ax.set_xlabel('')    
    _        = ax.set_xticks([])

    # Plot the accuracy for NMF features.
    ####################################################################################################################
    ax      =  fig.add_subplot( gs[:,1] )
    _       =  visualize_accuracy_per_signal_strength( df=clfs, ax=ax, legend_outside=False )
    _       =  ax.set_title('')
    axs    += [ax]          
    
    # Annotate figure numbers and write figure to the output.
    ####################################################################################################################
    #_   = figure_annotations( axs, xy=(-0.05,1.0), fontsize=14, ha='right', va='bottom' )
    dir = get_project_folder(f'data/case_outcome_predictions/NMF_based/', create=True)
    _   = fig.savefig( f'{dir}citation_feature_classification_results.pdf', bbox_inches='tight' )


def visualize_SHAP_values( ):
    """
    Trivial wrapper that calls the 'classify_one_feature_constellation' function so that we can get the Shapley values
    across the relevant feature specifications. 
    """

    # Fix some generic arguments.
    ####################################################################################################################
    args = {'analysis':True, 'control':True, 'test_size': 0.25, 'split':'generic', 'seed':0, 'embed_dim':30}

    # First, create the Shapley values for civil rights both for predictions with biographic and citation features. 
    ####################################################################################################################
    for feat in ['bio','cite_count']:
        _ = classify_one_feature_constellation( case_type= 'civil_rights', feat=feat, **args )

    # Next, create the Shapley values for all case types for biographic features, then aggregate them into a single
    # figure for convenience. 
    ####################################################################################################################
    fig, axs = plt.subplots( figsize=(18, 12), nrows=2, ncols=3 ) 
    fs       = 13
    cases    = get_case_types()

    for i, ct in enumerate(cases):

        if ct != 'civil_rights': classify_one_feature_constellation( case_type=ct, feat='bio', **args )

        sdir   = f'data/case_outcome_predictions/classification_per_constellation/'
        sdir  += f'case_type={ct}_feat=bio_control=True_test_size=0.25_split=generic_seed=0_embed_dim=30/'
        dir    = get_project_folder(sdir, create=False) 
        ax     = axs.flatten()[i]
        img    = plt.imread(f'{dir}SHAP_overview.png')
        _      = ax.imshow(img)
        _      = ax.axis('off')
        _      = ax.set_title(f"case type: {ct.replace('_',' ')}", fontsize=fs )   

    dir = get_project_folder(f'data/case_outcome_predictions/bio_based/', create=True)
    _   = fig.savefig( f'{dir}bio_features_SHAP_overview.pdf', bbox_inches='tight' )

    # We also create another plot for Shapley values with citation features and no NMF embedding (slow!)
    ####################################################################################################################
    _    =  print('Starting slow execution of Shapley calculation without NMF embedding.')
    args = {'analysis':True, 'control':False, 'test_size': 0.25, 'split':'generic', 'seed':0, 'embed_dim':None}
    _    =  classify_one_feature_constellation( case_type='civil_rights', feat='cite_count', **args )    


if __name__ == '__main__':

    # visualize_win_rates(  )
    # visualize_feature_distributions(  )
    # visualize_biographic_feature_classification_results(  )
    # visualize_citation_feature_classification_results(  )
    visualize_SHAP_values(  )