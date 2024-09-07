"""
In this file we predict case outcomes using a variety of machine learning methods and feature constellations.
"""


import os
import shap

import numpy   as np 
import pandas  as pd
import seaborn as sb

import statsmodels.formula.api as smf

from math       import log10, floor
from tqdm       import tqdm
from itertools  import product
from matplotlib import pyplot as plt
from joblib     import Parallel, delayed, cpu_count

from sklearn.metrics         import accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.decomposition   import NMF

from routines import get_project_folder, load_raw_features, balanced_downsample, train_classifier


def load_features(   case_type      =  'contract',
                     feat           =  'bio',
                     control        =   True, 
                     rescale        =   True, 
                     test_size      =   0.25, 
                     split          =  'generic',
                     seed           =   0,
                     embed_dim      =   30,
                     confidence_th  =   0.4, 
                     ):
    """
    Load the raw features, selected the relevant subset and rescale the data for a classification task.

    iput:
    ----
    case_type:      Case type to consider. Keep all cases if set to None.
    feat:           Features to consider.  Either biographic judge features ('bio') or citation counts ('cite_count').
    control:        If True, add circuit and time as control variable.
    rescale:        If True, standardize the features.
    test_size:      Fraction of test data.
    split:          Either random train-test split ('generic'), by judge ('judge') or across time ('temporal').
    seed:           Random number seed for the train-test split. 
    embed_dim:      NMF embedding dimension for citation count features (don't embed if None).
    confidence_th:  Only consider consider cases with an outcome classification confidence |p-0.5| > confidence_th.
    """

    # Load features and restrict to case type of interest.
    ####################################################################################################################
    df  =  load_raw_features( confidence_th )                                # load the data
    df  =  df[ df['case_type']==case_type ] if case_type is not None else df # restrict to this case type
    df  =  df.drop('case_type', axis=1)     if case_type is not None else df # remove case type info

    # Rescale time to seconds and then standardize for convenience in regression.
    ####################################################################################################################
    df                  = pd.get_dummies( df, columns=['circuit'] )   # 1-hot encoding
    df['decision_date'] = df['decision_date'].astype(int) / 10**15    # make smaller integer
    df                  = df.sort_values(by='decision_date')          # sort by time (important if temporal split)

    # Restrict to features of interest.
    ####################################################################################################################
    control_vars  =  [ c for c in df.columns if 'circuit_' in c ] + [ 'decision_date' ] # control variables 
    fixed_eff     =  [ 'party_republican', 'promoted', 'gender_male' ]                  # fixed judge effects
    bio           =   fixed_eff + [ 'experience','workload','win_rate' ]                # bibliographic judge features
    count_vars    =  [ c for c in df.columns if 'cite_count_case_' in c ]               # citaiton counts

    if   feat=='bio':         keep = bio    
    elif feat=='cite_count':  keep = count_vars
    else:                     raise ValueError(f'invalid feat argument {feat}')

    keep =  keep + control_vars if control else keep                   # add control variables
    keep = ['judgement'] + keep                                        # need to keep target
    df   =  df.loc[:,keep]                                             # restrict to features of interest
    df   =  df.dropna()                                                # drop missing values

    # Train test split. If we split by judge, we must have enough data-points for the split. If we split by time, we 
    # want to make sure the data is sorted by time. By contrast, if we don't split by time, we purposefully make sure 
    # that the data is randomly shuffled
    ####################################################################################################################
    tts    =  lambda sdf, **kwargs: train_test_split( sdf, test_size=test_size, random_state=seed, **kwargs )    
    counts =  df.groupby('judge_id')['judgement'].count().rename('judge_case_count') 
    df     =  pd.merge( df, counts, left_on='judge_id', right_index=True ) # add judge case count into
    df     =  df if not split=='judge' else df[ df['judge_case_count'] >= int(1/test_size)+1 ] # must have enough
    df     =  df.sample(frac=1, replace=False) if not split=='temporal' else df.sort_values( by='date' )
    df     =  df.drop('judge_case_count', axis=1) # remove helper column
    
    if   split=='generic':   IS, OS = tts( df, shuffle=True  )
    elif split=='temporal':  IS, OS = tts( df, shuffle=False )
    elif split=='judge':

        IS, OS = [], []
        for _, sdf in df.groupby('judge_id'):

            Is, Os   = tts( sdf )
            IS      += [Is]
            OS      += [Os]

        IS = pd.concat(IS, axis=0)
        OS = pd.concat(OS, axis=0)

    else: raise ValueError(f'invalid split {split}')

    # Standardize some features.
    ####################################################################################################################
    if rescale: 

        rescale   = [ 'decision_date', 'experience', 'workload', 'win_rate']     # features to standardize
        rescale  +=   count_vars if embed_dim is None else []                    # rescale citation counts
        rescale   =   np.intersect1d( IS.columns, rescale )                      # remaining features 
        not_resc  =   np.setdiff1d(   IS.columns, rescale )                      # features to not rescale

        if len(rescale)  > 0:

            IS_resc, OS_resc =   IS.loc[:,rescale], OS.loc[:,rescale]                   # features to standardize
            SS               =   StandardScaler()                                       # initialize scaler 
            IS_resc          =   pd.DataFrame(  SS.fit_transform(IS_resc),              # in-sample transform
                                                index   =        IS_resc.index, 
                                                columns =        IS_resc.columns )
            OS_resc          =   pd.DataFrame(  SS.transform(    OS_resc),              # out-of-sample transform
                                                index   =        OS_resc.index, 
                                                columns =        OS_resc.columns )

            IS_nr, OS_nr     =   IS.loc[:,not_resc], OS.loc[:,not_resc]                 # features that are not rescaled
            IS               =   pd.concat([IS_resc, IS_nr], axis=1)                    # in-sample features
            OS               =   pd.concat([OS_resc, OS_nr], axis=1)                    # out-of-sample features

    # Apply NMF dimension reduction of citation count features.
    ####################################################################################################################
    if embed_dim is not None and feat=='cite_count': 

        IS_cite, OS_cite  =  IS.loc[:,count_vars], OS.loc[:,count_vars]          # extract citation features
        nmf               =  NMF( n_components=embed_dim, max_iter=1000 )        # initialize class 
        cols              =  [f'NMF_dim_{i}' for i in range(embed_dim)]          # column names
        IS_cite           =  pd.DataFrame( nmf.fit_transform( IS_cite ),         # transform IS
                                            index   = IS_cite.index, 
                                            columns = cols ) 
        OS_cite           =  pd.DataFrame( nmf.transform(    OS_cite ),          # transform OS
                                            index   = OS_cite.index, 
                                            columns = cols )  
        SS                = StandardScaler()                                     # standardize the NMF embedding
        IS_cite           = pd.DataFrame(   SS.fit_transform(IS_cite),           # in-sample transform
                                            index   =        IS_cite.index, 
                                            columns =        IS_cite.columns)
        OS_cite           = pd.DataFrame(   SS.transform(    OS_cite),           # out-of-sample transform
                                            index   =        OS_cite.index, 
                                            columns =        OS_cite.columns)        
            
        not_cite          =  np.setdiff1d(  IS.columns, count_vars )             # features that are not citation feats
        IS_nc, OS_nc      =  IS.loc[:,not_cite], OS.loc[:,not_cite]              # features to 

        IS                = pd.concat( [IS_cite, IS_nc], axis=1)                 # in-sample features
        OS                = pd.concat( [OS_cite, OS_nc], axis=1)                 # out-of-sample features   

    # Balance the data, for easier interpretation of the out of sample results. 
    ####################################################################################################################
    IS = balanced_downsample( IS )                                         # always balance the training data
    OS = balanced_downsample( OS )                                         # balance the test data

    X_IS, y_IS = IS.drop('judgement', axis=1), IS['judgement']             # split up
    X_OS, y_OS = OS.drop('judgement', axis=1), OS['judgement']             # split up

    X_IS, X_OS = X_IS.astype('float'), X_OS.astype('float')                # make sure its float

    return X_IS, y_IS, X_OS, y_OS


def classify_one_feature_constellation( analysis=False, methods=None, **kwargs ):
    """
    Run one specific feature constellation specified via kwargs. 

    input:
    ----
    analysis:   If True, plot various analysis of the data to a unique sub-folder. 
    methods:    List of methods to run. If None, run all methods ('LR', 'MP', 'RF' and 'GB', see train_classifier).
    kwargs:     Arguments for the 'load_features' function which specifies the feature constellation.
    """

    # load the data 
    ####################################################################################################################
    X_IS, y_IS, X_OS, y_OS = load_features( **kwargs )    # load the data
    all_preds              = []                           # stores all classifications 

    output   =        f'Training data shape: {X_IS.shape}'
    output  += '\n' + f'Test data shape:     {X_OS.shape}'
    output  += '\n' + f'Baseline:            {100*y_IS.mean():.2f}%'    

    # Run a classic, in-sample logistic regression.
    ####################################################################################################################
    try:

        X          = pd.concat([X_IS, X_OS], axis=0 )                       # merge together
        y          = pd.concat([y_IS, y_OS], axis=0 )                       # merge together
        reg_data   = pd.concat([X,y], axis=1).astype(float)                 # merge together 
        reg_data   = reg_data.drop('circuit_9', axis=1, errors='ignore')    # remove one due to co-linearity
        dep_var    = 'judgement'                                            # dependent variable
        indep_var  = [ c for c in reg_data.columns if c != dep_var ]        # independent variables
        formula    = f"{dep_var} ~ {' + '.join(indep_var)}"                 # smf parser for regression
        model      =  smf.mnlogit(formula=formula, data=reg_data)           # intialize instance
        model      =  model.fit_regularized( disp=0, alpha=0.001  )         # fit
        reg        =  model.summary()                                       # regression summary

    except:
        print(f'Failed to run logistic regression with parameters {kwargs}.')
        reg = None

    # Define a sub-routine to subsample the score to obtain error bars. 
    ####################################################################################################################
    def subsample_score( t, p, score=accuracy_score ):
        """
        Rather than returning one score, return a sub-sampled average score and associated standard deviation.

        :param t:           Pandas Series of true values.
        :param p:           Pandas Series of predicted values.
        :param score:       Score function of two arguments: true value t and predicted value p.
        """

        t, p    =   pd.Series(t).rename('true'), pd.Series(p).rename('predicted')
        df      =   pd.concat([t,p], axis=1)
        samples = [ df.sample(frac=0.75, replace=True).dropna() for _ in range(100) ]
        vals    = [ score(sdf['true'], sdf['predicted']) for sdf in samples ]
        mean    =   np.nanmean(vals)
        std     =   np.nanstd(vals)
        std     =   str(round(std, -int(floor(log10(abs(std))))+ 1 )) # round to significant digits
        mean    =   str( np.round(mean, len(std)-2) )

        return str(mean) + u" \u00B1 "+ str(std)
        
    # Run classifications. 
    ####################################################################################################################
    all     = [ 'LR', 'MP', 'RF', 'GB' ]
    methods = all if methods is None else methods
    MLs     = {} # store all classifiers

    for method in methods: # iterate the methods of interest

        preds, ML            =  train_classifier( X_IS, y_IS, X_OS, y_OS, method=method )
        preds                =  preds.assign(**kwargs)      
        preds['#traindata']  =  len(X_IS)
        preds['#testdata']   =  len(X_OS)            
        preds['method']      =  method
        preds['score']       =  subsample_score( preds['predicted'], preds['true'] )
        MLs[method]          =  ML

        output     += '\n' + f"{method} - Out of Sample Accuracy : {preds['score'].iloc[0]}"
        all_preds  += [ preds ]

    all_preds = pd.concat(all_preds, axis=0) # merge back together

    if not analysis: return all_preds, reg

    # Create unique output folder and write to the output.
    ####################################################################################################################
    sdir =  '_'.join([ f'{key}={val}' for (key,val) in kwargs.items() ]) 
    dir  =  get_project_folder(f'data/case_outcome_predictions/classification_per_constellation/{sdir}/', create=True )

    print( output,  file=open( f'{dir}overview.txt',             "a"))
    print( reg,     file=open( f"{dir}logistic_regression.txt" , "a"))
        
    # Check accuracy as a function of increasing signal strength.
    ####################################################################################################################
    for method, preds in all_preds.groupby('method'):

        pred            = preds.copy(deep=True)                                             # don't overwrite 
        bins            = pd.qcut(pred['proba'],10, duplicates='drop')                      # create bins 
        pred['bin']     = [f'[{np.round(b.left,2)}, {np.round(b.right,2)}]' for b in bins]  # nice names
        pred['pred']    = np.round(pred['proba'])                                           # get prediction
        get_acc         = lambda x: accuracy_score(  x['true'], x['pred'] )                 # get accuracy
        get_prec        = lambda x: precision_score( x['true'], x['pred'] )                 # get precision
        get_rec         = lambda x: recall_score(    x['true'], x['pred'] )                 # get recall
        a               = pred.groupby('bin').apply( get_acc ).rename('accuracy')           # accuracy per bin
        p               = pred.groupby('bin').apply( get_prec).rename('precision')          # precision per bin
        r               = pred.groupby('bin').apply( get_rec ).rename('recall')             # recall per bin
        bin_accs        = pd.concat([a,p,r], axis=1)                                        # merge
        bin_accs        = bin_accs.stack().reset_index()                                    # stack
        bin_accs        = bin_accs.rename(columns={'level_1':'metric',0:'score'})           # rename         

        fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(9,12))
        fs       = 13
        ax       = axs[1] # plot bars
        _        = sb.barplot(  x          =  'bin',
                                y          =  'score',
                                hue        =  'metric', 
                                data       =   bin_accs,
                                palette    =   'icefire',
                                capsize    =   0.2,
                                ax         =   ax, 
                                )
        ax.set_xlabel('signal strength', fontsize=fs)
        ax.set_ylabel('accuracy', fontsize=fs)  

        # Plot ROC cuve. 
        ################################################################################################################
        ax        =  axs[0]
        FP, TP, _ =  roc_curve(  preds['true'].values, preds['predicted'].values )
        roc       =  100 * roc_auc_score(   preds['true'].values, preds['predicted'].values )
        acc       =  100 * accuracy_score(  preds['true'].values, preds['predicted'].values )
        f1        =  100 * f1_score(        preds['true'].values, preds['predicted'].values )
        score     =  f'ROC score={roc:.0f}%\naccuracy={acc:.0f}%\nF1={f1:.0f}%'
        _         =  ax.plot([0, 1], [0, 1], 'k--')
        _         =  ax.plot( FP, TP )
        _         =  ax.text( x=0.1, y=0.8, s=score, fontsize=10 )    
        _         =  ax.set_xlabel('false positive rate')
        _         =  ax.set_ylabel('true positive rate')

        # Plot signal as inset of ROC curve plot. 
        ####################################################################################################################
        ax_in = ax.inset_axes( [0.60, 0.15, 0.35, 0.25] )  # [x0, y0, width, height] 
        sb.kdeplot( preds['proba'].values, ax=ax_in, color='green', fill=True, linewidth=0 )
        ax_in.grid(False)
        ax_in.patch.set_alpha(0.03)     
        ax_in.set_xlabel('predicted probability', fontsize=fs-2) 
        ax_in.set_ylabel('frequency',    fontsize=fs-2) 

        fig.savefig(f'{dir}prediction_overview_{method}.pdf', bbox_inches='tight')      
        plt.clf()
        plt.close()

    # Rename the features so they look nice in the SHAP plots, then calculate SHAP values
    ####################################################################################################################
    X_OS          =   X_OS.rename(columns={'win_rate':'win-rate','gender_male':'male','party_republican':'Republican'})
    X_OS.columns  = [ c.replace('_',' ') for c in X_OS.columns ]
    X_OS          =   X_OS.rename( columns={'circuit 0': 'D.C. circuit'} )
    ML            =   MLs['GB']                                  # get the gradient boost model
    explainer     =   shap.TreeExplainer(ML, approximate=False ) # initialize shap instance
    shap_values   =   explainer(X_OS)                            # extract shap values

    # Create SHAP beeswarm plot.
    ####################################################################################################################    
    plt.clf()
    shap.plots.beeswarm( shap_values, max_display=21, show=False )
    fig, ax = plt.gcf(), plt.gca()
    fs      = 21
    ax.tick_params(labelsize=fs)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=fs)    
    cb_ax = fig.axes[1] 
    cb_ax.tick_params(labelsize=fs)
    cb_ax.set_ylabel("feature value", fontsize=fs)
    plt.savefig( f'{dir}SHAP_overview.png', bbox_inches='tight', dpi=1000 )
    plt.clf()
    plt.close()                 

    return all_preds, reg


def run_all_case_classifications( ):
    """
    Iterate al type of case-type and parameter constellations and write the results to the output. 
    """

    # Load the raw features just as a means to extract all available case-types. 
    ####################################################################################################################
    df     = load_raw_features( confidence_th=0 )                # load all data
    cases  = list(df['case_type'].unique())                      # all case types
    del df                                                       # save memory before parallelizing

    # Define experiments of interest and form all combinations.
    ####################################################################################################################
    vary  = {                                                          # all arguments
                'case_type'    :   cases,                              # all case types
                'feat'         : [ 'bio', 'cite_count' ],              # what feature constellations to run
                'control'      : [  True ],                            # control variables: yes or no?
                'split'        : [ 'generic', 'judge' ],               # how to do the train-test split
                'seed'         :  list(np.arange(5)),                  # sample a few times to get standard error
                'confidence_th': [ 0.4 ]                               # confidence thresholds
                }
    
    keys     =  list( vary.keys()    )                                  # list of keys
    lov      =  list( vary.values()  )                                  # list of values
    combs    =  list( product(*lov) )                                   # all value combinations
    lo_args  = [dict(zip(keys, vals)) for vals in combs]                # list of different x dicts    

    # Iterate each experiment, train ML models and write results to the output.
    ####################################################################################################################
    func     =  classify_one_feature_constellation                          # function to parallelize
    pf       =  Parallel( n_jobs=cpu_count()//2 )                           # dont use too many jobs: memory issue
    rets     =  pf( delayed(func)(**kwargs) for kwargs in tqdm(lo_args) )   # execute in parallel
    ml, regs =  list(zip(*rets))                                            # split into two lists

    # Write ML predictions to the output.
    ####################################################################################################################
    dir       = get_project_folder(f'data/case_outcome_predictions/', create=True)
    all_preds = pd.concat(ml, axis='index')
    _         = all_preds.to_parquet(f'{dir}all_classifications.parquet')

    # Write all regression tables to the output.
    ####################################################################################################################
    sdir = f'data/case_outcome_predictions/logistic_regressions/'

    for (kwargs, reg) in zip( lo_args, regs ):                          # iterate each regression table

        fn  =  '_'.join([ f'{k}={v}' for (k,v) in kwargs.items() ])     # all function names

        dir = get_project_folder( f'{sdir}txt/' )                       # where to store
        _   =  print( reg, file=open( f"{dir}{fn}.txt" , "a"))          # write to the output    

        try:
            dir =  get_project_folder( f'{sdir}tex/' )                  # where to store
            tbl =  reg.tables[1].as_latex_tabular()                     # extract latex tables       
            _   =  print( tbl, file=open( f"{dir}{fn}.tex" , "a"))      # write to the output as tex file
        except:
            pass 


def load_all_case_classifications( **kwargs ):
    """
    Just a convenience wrapper that access the entire data-frame and returns relevant subset as specified via kwargs.
    """ 

    fdir  =  get_project_folder(f'data/case_outcome_predictions/', create=False) 
    df    =  pd.read_parquet(f'{fdir}all_classifications.parquet')

    for key, val in kwargs.items(): df = df[ df[key]==val ]

    return df


if __name__=='__main__':

    run_all_case_classifications( )