"""
Collection of various routines used throught the project. 
"""

import os

import numpy   as np
import pandas  as pd
import seaborn as sb

from string                  import ascii_lowercase
from xgboost                 import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.neural_network  import MLPClassifier
from sklearn.metrics         import accuracy_score

from matplotlib              import pyplot as plt
from scipy.stats             import binom, binomtest, false_discovery_control

sb.set_style('whitegrid')

def get_project_folder( subfolder=None, create=True ):
	"""
	Safely returns and/or create a (sub-) path to this git repo. Hard-code in here your path to this git repo and put 
    the features.parquet file into a subfolder called /data/features/. 

	:param subfolder:   if not None, a subfolder of the git repo
	:param create:      if True, create the folder if it does not yet exist
	:return folder:     path to folder
	"""

	# hard-code the folder to this git repo
	####################################################################################################################
	f = 'add/path/to/this/repo/here'

	if  os.path.exists(f): folder = f
	else: raise ValueError('unknown root folder ')

	# add subfolder and check if it exists, if not, create it
	####################################################################################################################
	if subfolder is not None: folder += subfolder
	if create: os.makedirs( folder, exist_ok=True  )
	if not os.path.exists(folder): raise ValueError(f'folder {folder} does not exist')

	return folder


def load_raw_features( confidence_th=0.4 ):
    """
    Load all features from the features.parquet file and return only those where the tranformer's outcome classification
    deviates by more than confidence_th from 0.5.
    """
    dir             =   get_project_folder(f'data/features/', create=False)         # where the features are stored    
    df              =   pd.read_parquet(   f'{dir}features.parquet' )               # load the data
    above           =  (df['judgement_confidence'] - 0.5).abs() >= confidence_th    # True if enough confidence
    df              =   df[ above ]                                                 # predictions above threshold
    df['judgement'] =  df['judgement'].astype(int)                                  # from bool to int

    return df

def get_case_types( ):
    """
    Return a list of all case types. 
    """
    
    df    = load_raw_features( confidence_th=0.0 )
    cases = list( df['case_type'].unique() )    

    return cases


def balanced_downsample( X, target='judgement', seed=None ):
    """
    Downsample the majority target class of column 'target' in X to the size of the minority class.
    """

    counts = X[target].value_counts()                                  # count all target values
    n      = counts.min()                                              # target value with less data
    Xs     = []                                                        # stores sub-frames of each target value

    if len(X)==0 or n==0: return pd.DataFrame(columns=X.columns)       # if no data, return empty DataFrame

    for i,(_, df) in enumerate(X.groupby(target)):                     # group by target
        seed =  None if seed is None else 2*seed+i                     # define seed
        sdf  =  df.sample( n=n, replace=False, random_state=seed  )    # each class should have n values
        Xs  += [sdf]                                                   # collect

    X = pd.concat( Xs, axis='index' )                                  # concatenate
    X = X.sample( frac=1, replace=False )                              # shuffle

    return X


def train_classifier(X_IS, y_IS, X_OS, y_OS, method='GB', k=3 ):
    """
    Train a binary classifier (to predict case outcomes) with k-fold cross-validation. 

    X_IS:       In-sample features.
    y_IS:       In-sample target variable.
    X_OS:       Out-of-sample features.
    y_OS:       Out-of-sample target variable.
    method:     What classifier to train. Either gradient-boost ('GB'), random forest ('RF'), multi-layer perceptron
                ('MP') or logistic regression ('LR'). For each method, we use 5-fold cross-validation to fit the 
                hyperparameters.
    k:          Number of folds in the cross-validation.
    """ 

    # Define the classifiers and their hyperparameters.
    ####################################################################################################################
    nf = X_IS.shape[1] # number of features
    classifiers = {
                    'GB': (XGBClassifier(), {
                                                    'n_estimators':     [25, 50, 100],
                                                    'max_depth':        [2, 4, 5],
                                                    'learning_rate':    [0.01, 0.1, 0.2],
                                                }),
                    'RF': (RandomForestClassifier(), {
                                                    'n_estimators':      [ 500 ],
                                                    'max_depth':         [ 2, 3, 4 ],
                                                    'min_samples_leaf':  [ 0.01, 0.05, 0.1 ],
                                                }),
                    'MP': (MLPClassifier(max_iter=1000), {
                                                    'hidden_layer_sizes': [ (nf, nf//2), (2*nf, nf//2) ],
                                                    'alpha':              [ 0.0001, 0.001, 0.01 ]
                                                }),
                    'LR': (LogisticRegression(max_iter=1000), {
                                                                'C': [ 0.01, 0.1, 1, 10 ],
                                                                'penalty': ['l2'],
                                                            })
                }
    
    classifier, param_grid = classifiers[method] # select the classifier and its hyperparameters

    # Run k-fold cross-validation. 
    ####################################################################################################################
    grid_search = GridSearchCV(estimator    =  classifier,
                               param_grid   =  param_grid,
                               scoring      =  'accuracy',
                               cv           =  k,
                               n_jobs       = -1,
                               refit        =  True,
                               verbose      =  1 )
    
    grid_search.fit(X_IS, y_IS)         # fit the model on the IS data
    ML = grid_search.best_estimator_    # get the best model

    # Predict out of sample values.
    ####################################################################################################################
    oi       =  list(ML.classes_).index(1.0)                           
    y_pred   =  pd.Series( ML.predict(       X_OS ),       index=X_OS.index ).rename('predicted')
    y_proba  =  pd.Series( ML.predict_proba( X_OS )[:,oi], index=X_OS.index ).rename('proba')
    y_true   =  y_OS.rename('true')
    preds    =  pd.concat([y_proba, y_pred, y_true], axis=1)

    return preds, ML


def get_pval( n_succ, n_tot, baseline, alternative='two-sided' ):
    """
    Return the p-value of a binomial test testing whether a total number of successes (n_succ) out of a total number
    of observations (n_tot) is statistically different from the baseline success rate. 
    """   

    ret  = binomtest(   k           =  n_succ,       # number of successes
                        n           =  n_tot,        # total number of observations
                        p           =  baseline,     # base-line success rate 
                        alternative =  alternative ) # alternative to H_0, i.e.
                                                     # H_0 = 'success rate has baseline probability'
    return ret.pvalue


def false_discovery_correction( pvals, formatted=True ):
    """
    Adjust p-values for multiple testing and count the fraction of values below the 5% threshold. Return either raw or 
    nicely formatted values. 
    """

    th       =  0.05                                          # threshold for significance
    nv       =  len(pvals)                                    # number of p-values
    by_pvals =  false_discovery_control( pvals, method='by' ) # adjust p-values for multiple testing
    bh_pvals =  false_discovery_control( pvals, method='bh' ) # adjust p-values for multiple testing
    raw      =  ( pvals    < th ).sum()    / len(pvals)       # frac below th w/o correction
    bf       =  ( pvals    < th/nv ).sum() / len(pvals)       # frac below th with Bonferroni correction
    by       =  ( by_pvals < th ).sum()    / len(pvals)       # frac below th with Benjamini-Yekutieli correction
    bh       =  ( bh_pvals < th ).sum()    / len(pvals)       # frac below th with Benjamini-Hochberg correction

    if not formatted: return {'raw':raw, 'bf':bf, 'bh':bh, 'by':by, }

    txt      = f'nr of p-values: {nv:,}' + '\n'
    txt     += f'uncorrected fraction of values below {th}: {100*raw:.1f}%' + '\n'
    txt     += f'Bonferroni corrected fraction below {th}: {100*bf:.1f}%'  + '\n'
    txt     += f'Benjamini-Yekutieli corrected fraction below {th}: {100*by:.1f}%' + '\n'
    txt     += f'Benjamini-Hochberg corrected fraction below {th}: {100*bh:.1f}%' 

    return txt  


def binomial_confidence_interval( n, p=0.5, q=90 ):
    """
    Calculate the q% confidence interval for the number of successes in n trials of a binomial experiment with 
    probability of success p.
    """
    
    confidence_level = q / 100.0 # confidence level as a decimal
    lower_bound      = binom.ppf(      (1 - confidence_level) / 2, n, p )
    upper_bound      = binom.ppf(  1 - (1 - confidence_level) / 2, n, p )

    if lower_bound > 0: lower_bound = lower_bound     # adjust for rounding issue
    if upper_bound < n: upper_bound = upper_bound + 1 # adjust for rounding issue

    return lower_bound, upper_bound     


def get_case_cols():
    """
    Map case types to colors for consistent plotting across figures. 
    """
    
    cases    =  get_case_types()
    palette  =  sb.color_palette('colorblind', len(cases))
    palette  =  dict(zip(cases,palette))
    
    return palette  


def create_subplots( nplots, n_cols=2, width=8, height=4, **kwargs ):
    """
    Automatically create a smart arrangement of subplots. 

    :param nplots:      number of subplots
    :param n_cols:      number of columns
    :param width:       figure width
    :param height:      figure height 
    :param kwargs:      additional arguments for the subplots function
    """

    n_rows    = nplots // n_cols if nplots%n_cols==0 else nplots//n_cols + 1
    rem       = n_cols - nplots%n_cols # number of empty plots in last row
    figsize   = (width*n_cols, height*n_rows)
    fig, axs  = plt.subplots(n_rows, n_cols, figsize=figsize, **kwargs )
    _         = fig.subplots_adjust(hspace=0.25, wspace=0.1)

    if rem < n_cols: 
        for i in range(1,rem+1): # turn off windows in last row
            axs.flatten()[-i].set_visible(False) 

    return fig, axs 


def figure_annotations( axs, xy=(0.0,1.08), fontsize=17, ha='right', va='bottom' ):
    """
    Add (a), (b), (c), ... annotations to the subplots of list of axes axs. 
    """
    # annotate the axes
    ####################################################################################################################
    for i, ax in enumerate(axs):  ax.annotate(  text        = f'({ascii_lowercase[i]})',
                                                xy          =  xy, 
                                                xycoords    = 'axes fraction',
                                                fontsize    =  fontsize,
                                                ha          =  ha,
                                                va          =  va )      


def visualize_accuracy_per_signal_strength( df, ax, nb=5, hue='case_type', legend_outside=True ): 
    """
    We show prediction accuracies across different configurations as barplot. 

    :param df:      DataFrame with 'proba', 'predicted', 'true','seed' and 'hue' column (hue is used for seaborn)
    :param ax:      axis instance to plot into
    :param nb:      number of bins 
    :param hue:     column name along which to differentiate
    """

    # Check the input.
    ####################################################################################################################
    for col in ['proba','predicted','true','seed',hue]: assert col in df.columns, f'missing column {col}'
    
    # Calculate overall accuracy per case type as well as number of data per bin, this info is shown in the legend.
    ####################################################################################################################
    get_acc  =   lambda x: accuracy_score( x['true'], x['predicted'] )
    tot_acc  =   df.groupby(hue)[['true','predicted']].apply( get_acc )
    ncombs   =   nb * len(df['seed'].unique())  # number of bin and seed combinations
    count    =   df.groupby(hue)['true'].count() // ncombs
    count    =   count.to_dict()
    tot_acc  =   np.round( 100 * tot_acc, 1 )
    tot_acc  =   tot_acc.to_dict()
    labels   =   [ f"{key.replace('_',' ')}: {val}% ({count[key]} cases per bin)" for key,val in tot_acc.items() ]
    labels   =   [ f"{key.replace('_',' ')}: {val}%" for key,val in tot_acc.items() ]
    tot_acc  =   dict([ (key,l) for key,l in zip( tot_acc.keys(), labels ) ])
    
    # Group by case type and seed and by predict proba bins, and then calculate accuracy.
    ###########################################################################################a#########################
    accs = []
    
    for ((h,it), sdf) in df.groupby([hue,'seed']):
                
        proba               =  sdf['proba']
        proba              +=  np.random.normal(scale=1e-6, size=len(sdf)) # noise to avoid duplicates in bin
        sdf['bin']          =  pd.qcut( proba , nb, labels=range(1,nb+1) )
        vals                =  sdf.groupby('bin')[['true','predicted']].apply(get_acc)
        vals                =  vals.reset_index().rename({0:'accuracy'}, axis=1)
        vals[hue]           =  h        
        vals['seed']        =  it
        accs               += [vals]
        
    accs              =  pd.concat(accs,axis=0).dropna()    
    accs[hue]         =  accs[hue].replace(tot_acc) # add info about total score     
    nt                =   len( accs[hue].unique() )

    if hue=='case_type': # make color consistent will all other plots
        palette           =  get_case_cols() if hue=='case_type' else sb.color_palette('deep',nt)
        palette           =  pd.Series(palette).rename('color').to_frame()
        palette           =  palette.reset_index()
        palette['name']   =  palette['index'].map(tot_acc)
        palette           =  palette.set_index('name')['color'].to_dict()
    else: # general colors
        hues     =  accs[hue].unique()
        cols     =  sb.color_palette('cubehelix',len(hues))
        palette  =  dict([(h,c) for h,c in zip(hues,cols)])        

    # Create a bar plot that shows the accuracy as a function of signal strength.
    ####################################################################################################################
    fs   =  14
    _    =  sb.barplot( x          =  'bin',
                        y          =  'accuracy',
                        errorbar   =  'sd',
                        hue        =   hue,
                        data       =   accs,
                        palette    =   palette, 
                        alpha      =  0.75, 
                        ax         =   ax )

    ax.set_xlabel('classification score quintile', fontsize=fs)
    ax.set_ylabel('accuracy', fontsize=fs)  
    ax.set_ylim( bottom=0.4 ) 
    ax.set_title('accurarcy per probabilistic classification quintile', fontsize=fs)
    ax.legend(  frameon=False, title='overall accuracy:', 
                loc ='upper center', 
                ncol = 1                  if legend_outside else 1, 
                fontsize=fs+2             if legend_outside else fs-4,
                bbox_to_anchor=(1.5, 0.9) if legend_outside else None )    