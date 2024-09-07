"""
Based on the judge's case summary, we have run a longformer-based binary classifier to determine whether a case is
won by the plaintiff or not. The output of that classifier is a sigmoid activation function that we interpret as 
the probability that the plaintiff won that case. To obtain a final classification, we have to determine a threshold 
above/below which we trust the longformer's output. For instance, if the longformer predicts a probability of 0.51 that 
the plaintiff won the case, we trust it less than if the longformer predicts a probability of 0.99. Therefore, we 
define the 'confidence threshold' as the deviation from 0.5 at which we trust the longformer's output. In this file, 
we check the sensitivity of our results on the confidence threshold. Generally, we find that a threshold of 0.4 is a 
good choice, that is we trust the longformer's output if it predicts a probability below 0.1 or above 0.9. However, 
as shown in this file, the results are robust to changes in the confidence threshold ranging from 0.05 to 0.55.
"""

import numpy   as np 
import pandas  as pd
import seaborn as sb

from itertools         import product 
from tqdm              import tqdm
from numpy             import arange
from sklearn.metrics   import accuracy_score
from matplotlib        import pyplot as plt
from matplotlib.ticker import FuncFormatter

from routines             import get_project_folder, load_raw_features, train_classifier
from routines             import visualize_accuracy_per_signal_strength, figure_annotations
from case_classifications import load_features


def check_transformer_classifier_sensitivity( ):
    """
    We test the longformer's output as a function of the confidence threshold by relying on validation data that the 
    transformer has not seen during training, but for which we know the ground truth from IDB. 
    """

	# Load all the features for all cases.
	####################################################################################################################
    df    =  load_raw_features( confidence_th=0.0 )                     # load the raw features
    ths   =  np.arange(0, 0.48, 0.02 )                                  # sensitivity thresholds
    vals  = []                                                          # stores results as a function of threshold

    # Check outcome accuracy as a function of confidence threshold.
    ####################################################################################################################
    for th in tqdm(ths):

        above    =  (df['judgement_confidence'] - 0.5).abs() >= th               # True if enough confidence
        sdf      =   df[ above ]                                                 # predictions above threshold
        nr_cases =   len(sdf)                                                     # total number of datapoints    
        sdf      =   sdf[ sdf['subset']=='validation' ]                          # restrict to validation set
        acc      =   (sdf['judgement_ground_truth']==sdf['judgement']).mean()    # accuracy of the classifier
        vals    +=   [ (th,nr_cases,acc) ]                                       # store results 

    vals = pd.DataFrame( vals, columns=['th','#cases','accuracy'] )              # make into DataFrame
    vals = vals.set_index('th')                                                  # set threshold as index

    # Visualize the result. 
    ####################################################################################################################
    fig, ax = plt.subplots( figsize=(12,7) )
    fs      = 14
    axt     = ax.twinx()
    _       = vals['accuracy'].plot( ax=ax, color='ForestGreen',  marker='o', linestyle='--', label='accuracy' )
    _       = vals['#cases'].plot(  ax=axt, color='SteelBlue',    marker='s', linestyle=':',  label='number of cases' )
    _       = axt.axvline( x=0.4, color='red', linestyle=':', linewidth=2 )
    _       = ax.set_xlabel('classification confidence threshold', fontsize=fs)
    _       = ax.set_ylabel('accuracy',         fontsize=fs, color='ForestGreen')
    _       = axt.set_ylabel('number of cases', fontsize=fs, color='SteelBlue')
    _       = ax.set_xlim( min(ths), max(ths) )
    _       = axt.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}')) # add comma separator to 1000s
    _       = axt.grid(False)

    # Write to the output
    ####################################################################################################################
    dir  = get_project_folder( f'data/confidence_threshold_sensitivity_analysis/', create=True )
    _    = vals.to_csv(     f'{dir}transformer_classifier_sensitivity.csv' )
    _    = fig.savefig(     f'{dir}transformer_classifier_sensitivity.pdf', bbox_inches='tight' )
    _    = plt.clf()
    _    = plt.close()


def check_case_prediction_dependency( ):
    """
    Here, we check the dependency of our main result, namely the predictability of case outcome as a function of judge
    features, on the confidence threshold and the machine learning method. 
    """

    # Train a classifier for fixed confidence threshold.
    ####################################################################################################################
    def classify( th, seed ):
        """
        Wrapper routine to classify for given case type, seed and embeding dimension for parallelization below.
        """
        X_IS, y_IS, X_OS, y_OS = load_features(     case_type      =   None, 
                                                    feat           =  'bio',
                                                    control        =   True, 
                                                    rescale        =   True,
                                                    split          =  'generic',
                                                    test_size      =   0.25,
                                                    seed           =   seed,
                                                    confidence_th  =   th, 
                                                    )
        
        preds   = []
        for ml in MLs:
        
            sp, _  =   train_classifier( X_IS, y_IS, X_OS, y_OS, method=ml, k=2 )
            sp     =   sp.assign( **{'th':th, 'ml':ml, 'seed':seed } )
            preds +=  [sp]

        preds = pd.concat( preds, axis=0 )

        return preds
    
    # Train classifiers as a function of confidence threshold.
    ####################################################################################################################
    MLs      =  ['LR', 'GB', 'RF', 'MP']                                       # methods
    ths      =    np.arange( 0.02, 0.44, step=0.02)                            # threshold steps    
    seeds    =    arange(10)                                                   # repetitions (for standard error)
    combs    =    list(product( ths, seeds ))                                  # all combinations 
    preds    =  [ classify(t,s) for (t,s) in tqdm(combs) ]                     # parallelize internally in k-fold
    preds    =    pd.concat( preds, axis=0 )                                   # merge all results together

    sdir    =  f'data/confidence_threshold_sensitivity_analysis/'              # subfolder to store
    dir     =  get_project_folder( sdir, create=True)                          # complete path
    _       =  preds.to_parquet(f'{dir}case_prediction_sensitivity.parquet')   # write to the output

    # Calculate the accuracy per combination.
    ####################################################################################################################
    get_acc =  lambda x: accuracy_score( x['true'], x['predicted'] )                       # calculate accuracy
    accs    =  preds.groupby(['th','ml','seed'])[['true','predicted']].apply( get_acc )    # accuracy per type
    accs    =  accs.rename('accuracy').reset_index()                                       # reset

    # Plot the dependency of prediction accuracy on threshold cut-off: Once for accuracy, and once as a function of 
    # signal strength (i.e. predict_proba) for the gradient boosting classifier.
    ####################################################################################################################
    fig, axs =  plt.subplots( nrows=2, ncols=1, figsize=(16,16) )
    _        =  fig.subplots_adjust( hspace=0.3 )
    fs       =  14

    ax       =  axs[0]
    _        =  sb.lineplot(x='th', y='accuracy', errorbar='ci', hue='ml', data=accs, ax=ax, linestyle='--', marker='s')
    _        =  ax.set_xlabel( f'classification confidence threshold', fontsize=fs )
    _        =  ax.set_ylabel( f'accuracy', fontsize=fs )
    _        =  ax.legend(loc='center left', fontsize=fs)
    _        =  ax.set_xlim( accs['th'].min(), accs['th'].max() )
    _        =  ax.set_title( f'accuracy as a function of confidence threshold', fontsize=fs+1 )

    ax       =  axs[1]
    sp       =  preds[ preds['ml']=='GB' ]         # restrict to gradient boosting classifier (~best)
    sp['th'] =  np.round(sp['th'],2).astype(str)   # round to avoid ugly cut-offs in string-conversion
    _        =  visualize_accuracy_per_signal_strength( sp, ax=ax, nb=5, hue='th', legend_outside=False )
    _        =  ax.set_title(f'{ax.get_title()} for gradient boost method', fontsize=fs+1 )

    _        =  figure_annotations( axs, xy=(-0.01,1.00), fontsize=14, ha='right', va='bottom' )
    _        =  fig.savefig(f'{dir}case_prediction_sensitivity.pdf', bbox_inches='tight' )
    _        =  plt.clf()
    _        =  plt.close()

if __name__=='__main__':

    check_transformer_classifier_sensitivity( )
    check_case_prediction_dependency( )