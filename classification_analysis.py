"""
In this script we run various analyses on the case outcome predictions.
"""

import re

import numpy   as np 
import pandas  as pd
import seaborn as sb

import statsmodels.formula.api as smf

from itertools         import product 
from tqdm              import tqdm
from numpy             import arange, hstack
from sklearn.metrics   import accuracy_score
from matplotlib        import pyplot as plt

from routines             import get_project_folder, balanced_downsample, train_classifier, false_discovery_correction
from routines             import get_pval, get_case_cols, binomial_confidence_interval, get_case_types
from case_classifications import load_features, load_all_case_classifications


def check_NMF_emb_dim_dependency( ):
    """
    Analyze dependency on embedding dimension. 
    """

    # Train a classifier for fixed embedding dimension.
    ####################################################################################################################
    def classify( case_type, seed, emb_dim ):
        """
        Wrapper routine to classify for given case type, repetition and embeding dimension for parallelization below.
        """

        X_IS, y_IS, X_OS, y_OS = load_features(     case_type      =   case_type, 
                                                    feat           =  'cite_count',
                                                    control        =   False, 
                                                    rescale        =   True, 
                                                    test_size      =   0.25,
                                                    seed           =   seed,
                                                    embed_dim      =   emb_dim, 
                                                    confidence_th  =   0.4,
                                                    )
        
        preds, _  = train_classifier( X_IS, y_IS, X_OS, y_OS, method='GB', k=2 )
        preds     = preds.assign(**{'case_type':case_type, 'seed':seed, 'emb_dim':emb_dim})

        return preds

    # Train classifiers as a function of embedding dimension.
    ####################################################################################################################
    case_type  =    get_case_types()                                            # all case types
    seeds      =    arange(10)                                                  # seeds (for standard error)
    emb_dim    =    hstack([ arange(1,10), arange(10,20,2), arange(20,50,4) ])  # embedding dimensions
    combs      =    list(product(case_type, seeds, emb_dim))                    # all combinations 
    preds      =  [ classify(c,s,e) for (c,s,e) in tqdm(combs) ]                # parallelize internally in k-fold  
    preds      =    pd.concat( preds, axis=0 )                                  # merge all results together
    sdir       =   f'data/case_outcome_predictions/NMF_based/'                  # subfolder to store
    dir        =    get_project_folder( sdir, create=True )                     # where to store
    _          =    preds.to_parquet(f'{dir}NMF_dim_dependency.parquet')        # write to the output

    # Calculate the accuracy, and subsequently rescale for each case type and rep such that NMF dim =1 is set to 0.
    ####################################################################################################################
    get_acc =  lambda x: accuracy_score( x['true'], x['predicted'] )          # calculate accuracy
    accs    =  preds.groupby(['case_type','emb_dim','rep']).apply( get_acc )  # accuracy per type
    accs    =  accs.rename('accuracy').reset_index()                          # reset

    # Plot the dependency.
    ####################################################################################################################
    fig, axs =  plt.subplots( nrows=3, ncols=2, figsize=(15,8), sharex=True )    
    _        =  fig.subplots_adjust( wspace=0.15, hspace=0.2 )
    fs       =  14

    for i, (ct,sdf) in enumerate( accs.groupby('case_type') ):
        
        ax = axs.flatten()[i]
        _  = sb.lineplot( x='emb_dim', y='accuracy', errorbar=('sd',1),  data=sdf, ax=ax, 
                         linestyle='--', marker='s', color= get_case_cols()[ct]  )
        _  =  ax.set_title(ct, fontsize=fs )
        _  =  ax.set_xlim(1, 48)
        _  =  ax.set_xlabel(f'NMF embedding dimension', fontsize=fs)    
        _  =  ax.set_ylabel( f'accuracy', fontsize=fs )

    fig.savefig(f'{dir}NMF_dim_dependency.pdf', bbox_inches='tight')
    plt.clf()
    plt.close()    


def explain_biographic_judge_variables( ):
    """
    Use citation features to explain bibliographic judge features.
    """

    # Load the regression targets (biographic judge features) and independent variables (citation NMF embeddings).
    ####################################################################################################################
    fixed = {   'case_type'      :   None,          # use all case types             
                'control'        :   False,         # we don't need time as control variable
                'rescale'        :   True ,         # standardize
                'seed'           :   0,             # not relevant
                'split'          :  'generic',      # irrelevant because all daya is in-sample
                'test_size'      :   0.001,         # we basically want all the data
                'embed_dim'      :   30,            # just pick it large enough
                'confidence_th'  :   0.4,           # cf. sensitivty checks 
                }

    targets, _, _, _ = load_features( feat='bio',        **fixed )                      # info about judges
    feats,   _, _, _ = load_features( feat='cite_count', **fixed )                      # info about their citations
    rem              = [ c for c in targets.columns if 'circuit_' in c ]                # control variables
    feats            = feats.drop(   rem, axis=1 )                                      # remove control variables
    targets          = targets.drop( rem, axis=1 )                                      # remove control variables

    # Define dependent and independent variables.
    ####################################################################################################################
    indep_vars =   list(feats.columns)                                                  # all independent variabls
    cont       = ['experience','win_rate','workload']                                   # arguments for regression
    disc       = ['gender_male','party_republican','promoted']                          # arguments for classification
    dep_vars   =   cont + disc                                                          # all dependent arguments
    dir        =   get_project_folder(f'data/explain_judge_variables/', create=True )   # where to store

    # Implement regression routine.
    ####################################################################################################################
    def regress( dv='experience', alpha=0.01 ):
        """
        Run a logistic or linear regression with regularization for a given dependent variable dv. 
        """

        X       = feats.copy()                                                  # what data to use: true or random
        X       = pd.concat([X, targets.loc[:,dv]], axis=1)                     # data for regression
        formula = f"{dv} ~ {' + '.join(indep_vars)}"                            # smf parser for regression    
        logit   = True if dv in disc else False                                 # whether logistic regression or not
        func    = smf.mnlogit if logit else smf.ols                             # logistic or linear regression
        model   = func( formula=formula, data=X )                               # intialize instance
        args    = {'alpha':alpha, 'refit':True, 'maxiter':500, 'disp':False }  # fitting arguments
        model   = model.fit_regularized( **args ) if logit else model.fit()     # fit    

        return model     

    # Iterate each dependent variable, run the regression for each and write to the output.
    ####################################################################################################################
    M = {}                                                                      # store model in dictionary
    for dv in dep_vars:                                                         # iterate each dependent variable

        model   = regress( dv )                                                 # store model in a dictionary
        M[dv]   = model                                                         # store model in a dictionary
        reg     = model.summary()                                               # regression summary
        fn      = f'regression_of_{dv}'                                         # unique filename
        _       = print( reg, file=open( f"{dir}{fn}.txt" , "a"))               # write log light to the output

    # Summarize all regression results in a table and also write it to the output in LaTeX format.
    ####################################################################################################################
    def models_to_latex_table( models_dict ):
        """
        Transform a dictionary of statmodels into a table comparing the coefficients of multiple models.
        """

        # Initialize the LaTeX table
        table = "\\clearpage\n"
        table += "\\begin{table}[ht]\n"
        table += "\\centering\n"
        table += "\\begin{tabular}{l%s}\n" % ('c' * len(models_dict))
        table += "\\toprule\n"

        # Get the sorted model names and add them as the first row of the table
        model_names = sorted(models_dict.keys())
        table += "& %s \\\\\n" % " & ".join([name.replace("_", " ") for name in model_names])
        table += "\\midrule\n"

        # Generate R^2 or Pseudo R^2 values for each model
        r2_values = [
            f"{model.rsquared*100:.1f}\\%" if hasattr(model, 'rsquared') else f"{model.prsquared*100:.1f}\\%"
            for model in models_dict.values()
        ]
        table += "\\textbf{(Pseudo-) R\\textsuperscript{2}} & %s \\\\\n" % " & ".join(r2_values)

        # Add a double line separator
        table += "\\midrule\\midrule\n"

        # Get all covariates from the models
        all_covariates = set().union(*(model.params.index for model in models_dict.values()))

        # Sort the covariates in a specific order
        sorted_covariates = sorted(
            all_covariates, key=lambda x: [int(num) if num.isdigit() else num for num in re.findall(r'\d+|\D+', x)]
        )

        # Iterate over covariates and models to generate coefficient values
        for covariate in sorted_covariates:
            coefficients_str = ""
            for model_name in model_names:
                model = models_dict[model_name]
                coefficient = model.params.loc[covariate].squeeze()
                p_value = model.pvalues.loc[covariate].squeeze()

                # Determine asterisks based on the significance level
                ast = "" if p_value >= 0.05 else "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"

                coefficients_str += f"{coefficient:.2f}{ast} ({p_value:.2f})"
                coefficients_str += " & "

            table += f"{covariate.replace('_', ' ')} & {coefficients_str[:-3]} \\\\\n"

        # Finalize the table and add necessary LaTeX formatting
        table += "\\bottomrule\n"
        table += "\\end{tabular}\n"
        table += "\\caption{Model Comparison}\n"
        table += "\\label{tab:model_comparison}\n"
        table += "\\end{table}\n"
        table += "\\clearpage\n"

        # Return the generated table
        return table
    
    table = models_to_latex_table( M )
    print( table, file=open( f"{dir}judge_feature_explainability.tex" , "a")) 


def examine_predicability_per_judge( ):
    """
    Examine the predicability of cases per judge. Only consider judges with at least mc cases after subsampling, and
    validate outperformance as significant per the q% quantile. Only consider threshold with a proba that deviates by 
    more than th from 0.5.
    """
    
    # Load the data.
    ####################################################################################################################
    predictions = load_all_case_classifications( method        =  'GB',
                                                 feat          =  'cite_count',
                                                 split         =  'judge',
                                                 confidence_th =   0.4 )  
    

    # Check outcome accuracy as a function of confidence threshold.
    ####################################################################################################################
    mc    =  50                                                             # minimal number of cases per judge
    th    =  0.025                                                          # chosen threshold (cf. below) 
    ths   =  np.arange(0, 0.075, 0.005 )                                    # predictive confidence thresholds
    vals  = []                                                              # stores results as a function of threshold
    for t in ths:

        above      =  (predictions['proba'] - 0.5).abs() >= t               # True if enough confidence
        sdf        =   predictions[ above ]                                 # predictions above threshold
        sdf        =   balanced_downsample( sdf, target='true' )            # make symmetric
        judges     =   sdf.groupby('judge_id')['proba'].count()             # number of cases per judge
        judges     =   judges[ judges >= mc ].index                         # only consider judges with enough cases
        nr_judges  =   len(judges)                                          # total number of judges
        acc        =   (sdf['predicted']==sdf['true']).mean()               # accuracy of the classifier
        vals      +=   [ (t,nr_judges,acc) ]                                # store results 

    vals    = pd.DataFrame( vals, columns=['th','#judges','accuracy'] )     # make into DataFrame
    vals    = vals.set_index('th')                                          # set threshold as index
    fig, ax = plt.subplots( figsize=(12,7) )
    fs      = 14
    axt     = ax.twinx()
    _       = vals['accuracy'].plot( ax=ax, color='ForestGreen',  marker='o', linestyle='--', label='accuracy' )
    _       = vals['#judges'].plot(  ax=axt, color='SteelBlue',   marker='s', linestyle=':',  label='number of cases' )
    _       = axt.axvline( x=th, color='red', linestyle=':', linewidth=2 )
    _       = ax.set_xlabel('prediction confidence level', fontsize=fs)
    _       = ax.set_ylabel('accuracy',          fontsize=fs, color='ForestGreen')
    _       = axt.set_ylabel('number of judges', fontsize=fs, color='SteelBlue')
    _       = ax.set_xlim( min(ths), max(ths) )
    _       = axt.grid(False)  

    dir     =  get_project_folder(f'data/case_outcome_predictions/per_judge/', create=True) # where to store results
    _       =  vals.to_csv( f'{dir}case_predictability_confidence.csv' )
    _       =  fig.savefig( f'{dir}case_predictability_confidence.pdf', bbox_inches='tight' )    
    _       =  plt.clf()
    _       =  plt.close()
    
    # Calculate for each judge the prediction accuracy. 
    ####################################################################################################################
    def judge_predictability( seed=0 ):
        """
        Sub-routine that examines for each judge the predictability on out of sample cases. In principle, this function
        could just be run once. But the balanced_downsample introduces some stochasticity, which is why this function
        can be called multiple times to get confidence intervals.
        """

        # Count the number of judges that are outside the confidene interval, as well as the number of p-values that are
        # rejecting the null-hypothesis of random guessing.
        ################################################################################################################
        vals  = []

        for judge, sdf in predictions.groupby('judge_id'):

            sdf  =  sdf[ (0.5-sdf['proba']).abs() >=th ]
            sdf  =  balanced_downsample( sdf, target='true', seed=seed )

            if len(sdf) < mc: continue
            
            n       =   len(sdf)                                                    # total nr of cases for this judge
            nr_win  =   sdf['true'].sum()                                           # total number of plaintiff wins
            maj     =   max( nr_win, n-nr_win)                                      # majority winners
            succ    =   (sdf['true']==sdf['predicted']).sum()                       # number of successful predictions
            acc     =   succ / n                                                    # accuracy of our predictions
            lb, ub  =   binomial_confidence_interval(n, p=0.5, q=90)                # 90% conf intv for random guessing
            cl      =  'over' if succ>=ub else 'under' if succ<=lb else 'undecided' # significance of prediction
            pv      =   get_pval( succ, n, baseline=0.5, alternative='greater' )    # p-val for H0: random guessing
            vals   += [ (judge, n, maj, succ, acc, lb, ub, cl, pv) ]                # append values

        cols            = [ 'judge', 'nr_cases', 'majority', 'success', 'accuracy']
        cols           += [ 'lower bound', 'upper bound', 'class', 'pvalue' ]
        df              =   pd.DataFrame(vals, columns=cols )
        df['p_adjust']  =   false_discovery_correction( df['pvalue'], formatted=False )['bh']
        df['CI_lb']     =   0.5 - df['lower bound'] / df['nr_cases'] # normalized lower bound for CI
        df['CI_ub']     =   df['upper bound'] / df['nr_cases'] - 0.5 # normalized upper bound for CI
        df              =   df.sort_values( by='accuracy' )
        df['index']     =   range(len(df))
        df['class']     =   df['class'].map( {'under':'underperform', 'over':'outperform', 'undecided':'undecided'} )

        # Count the number of judges that are outside the confidene interval, as well as the number of p-values that are 
        # rejecting the null-hypothesis of random guessing. 
        ################################################################################################################
        count       =   df['class'].value_counts( normalize=False )
        count.index =  [ f'judge_significance_{i}' for i in count.index ] 
        pinfo       =   false_discovery_correction( df['pvalue'], formatted=False )
        pinfo       =   pd.Series(pinfo)
        pinfo.index = [ f'p_val_{i}' for i in pinfo.index ]  
        info        =   pd.concat([count,pinfo])

        return df, info
    
    # Run the above routine multiple times to get error bars.
    ####################################################################################################################
    nr    =   30                                                      # number of repetitions
    rets  = [ judge_predictability(seed=i) for i in range(1,nr+1) ]   # run the routine multiple times
    dfs   = [  i[0] for i in rets ]                                   # extract the DataFrames
    infos = [  i[1] for i in rets ]                                   # concatenate the results

    # Average the results and write to the output. 
    ####################################################################################################################
    desc = pd.concat(infos, axis=1).T
    desc = desc.describe().T
    _    = desc.to_csv( f'{dir}predictability_per_judge_statistics.csv', index=True )

    # Select a single representative example to visualize the concept. 
    ####################################################################################################################
    df   = dfs[2]
    info = infos[2]
    _    = info.to_csv( f'{dir}predictability_per_judge.csv', header=None )

    # Define the figure axis and plotting colors. 
    ####################################################################################################################
    fig, (ax_dist, ax) = plt.subplots( 2, 1, 
                                        figsize     =  (7, 8), 
                                        sharex      =  True, 
                                        gridspec_kw = {'height_ratios': [1, 5],'hspace': 0}, 
                                        )

    cols = {    'outperform':   'forestgreen', 
                'undecided':    'grey',
                'underperform': 'orangered',
                }
    
    # Plot the distribution of prediction accuracies.
    ####################################################################################################################
    sb.histplot( x          =  'accuracy', 
                data       =   df,
                hue        =  'class',
                hue_order  = ['outperform','underperform','undecided'],      
                linewidth  =   0,
                palette    =   cols,
                fill       =   True, 
                alpha      =   0.5,                        
                ax         =   ax_dist,
                legend     =   True,
                )     

    _   = ax_dist.yaxis.grid(False)
    _   = ax_dist.set_ylabel('frequency')
    leg = ax_dist.get_legend()
    _   = leg.set_title(None)
    _   = leg.set_frame_on(False)

    # Skip middle data to avoid the figure from beeing too cluttered.
    ####################################################################################################################
    skip         = 10
    df           = df.copy(deep=True)
    nl, nh       = 20, 40 # how many to show at the bottom and on the top
    dfl          = df.iloc[:nl]
    dfh          = df.iloc[-nh:]
    dfh['index'] = skip + np.arange(nl, nl+nh)
    df           = pd.concat([ dfl, dfh], axis=0 )    

    # Plot the win-rate by judge and the null model confidence intervals. 
    ####################################################################################################################
    ax.errorbar(x          =    [0.5] * len(df), 
                y          =    df['index'],
                xerr       =    df.loc[ :, ['CI_lb','CI_lb']].T, 
                fmt        =   'none',
                color      =   'grey', 
                elinewidth =    0.5,
                capsize    =    0, 
                alpha      =    0.5, 
                )

    ax.scatter( x          =  df['accuracy'],
                y          =  df['index'],
                #s          =  [  20 if row['class']=='undecided' else  40 for _, row in df.iterrows() ], 
                #marker     =  [ 's' if row['class']=='undecided' else 'o' for _, row in df.iterrows() ], 
                marker     =  'o',
                s          =   30,
                edgecolor  =  'none', 
                facecolor  =  [ cols[row['class']] for _, row in df.iterrows() ],
                alpha      =   1.0, 
                )

    _ = ax.set_xlabel( 'prediction accuracy per judge' )
    _ = ax.set_ylabel( 'judges sorted by prediction accuracy' )
    _ = ax.set_yticks([])
    _ = ax.set_xlim( left=(0.5-df['CI_lb']).min()-0.02, right=df['accuracy'].max()+0.01 )
    _ = ax.set_ylim( bottom=-1, top=df['index'].max()+1 )

    # Add dots to indicate the break.
    ####################################################################################################################
    for i in range(skip):
        ax.text(    x          =  0.495, 
                    y          =  nl+i,
                    s          =  '....',
                    color      =   'grey',
                    fontsize   =   12, 
                    )    

    # Write all to the output.
    ####################################################################################################################
    fig.savefig( f'{dir}predictability_per_judge.pdf', bbox_inches='tight' )
    plt.clf()
    plt.close()        


if __name__=='__main__':

    # check_NMF_emb_dim_dependency( )
    # explain_biographic_judge_variables( )
    examine_predicability_per_judge( )