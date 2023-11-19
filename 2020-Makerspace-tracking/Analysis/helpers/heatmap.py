import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pingouin import distance_corr

def compute_correlation(df, outcomes, predictors, test='pearson', height=None,
                        plot_r=True, plot_p=False, sort_by=None,
                        title="", save_fig=None):
    '''
    This function returns a heatmap of correlation coefficients where significant results are highlighted
    df: a dataframe
    outcomes: a list of column names of outcome measures
    predictors: a list of column names of predictors
    plot_r: whether to plots the heatmap of correlation coefficients, default True
    plot_p: whether to plots the heatmap of p-values, default False
    save_fig: a path to save the figure as a file (e.g., ./heatmap.pdf)
    '''
    
    # size of the table
    if height == None: 
        height = int(len(predictors) * 2)

    # compute the correlations
    filtered = df[list(set(outcomes+predictors))].astype('float')
    correlations = filtered.corr()
    
    #compute the p-value for each correlation
    pvalues = pd.DataFrame(index=filtered.columns, columns=filtered.columns)
    for col1 in filtered.columns:
        for col2 in filtered.columns:
            if col1 == col2: 
                pvalues.at[col1,col2] = 1
            else:
                tmp_df = filtered[[col1,col2]].dropna()
                if len(tmp_df[col1]) > 1 and len(tmp_df[col2]) > 1: 
                    pval = pearsonr(tmp_df[col1],tmp_df[col2])[1]
                    if test == 'dist':
                        dcor,pval = distance_corr(tmp_df[col1],tmp_df[col2])
                        correlations.at[col1,col2] = dcor
                    pvalues.at[col1,col2] = pval

    for col in pvalues.columns:
        pvalues[col] = pd.to_numeric(pvalues[col], errors='coerce')
        
    # plotting correlations    
    if plot_r:
        # create a new figure
        plt.figure(figsize = [15, height])
        plt.title(title)
        correlations = correlations.loc[outcomes][predictors]
        if sort_by != None: correlations = correlations.sort_values(sort_by)

        # add annotations for sig results
        annot = correlations.copy().map(str)
        for index, row in annot.iterrows():
            for col in annot.columns:
                val = str(round(float(annot.at[index,col]), 2))
                #val = annot.at[index,col][0:3] #str(round(float(annot.at[index,col]), 2))
                if pvalues.at[index,col] < 0.01: val += ' **'
                elif pvalues.at[index,col] < 0.05: val += ' *'
                elif pvalues.at[index,col] < 0.1: val +=  ' †'
                annot.at[index,col] = val
        
        ax = sns.heatmap(correlations, annot=annot, fmt='', cmap="coolwarm")
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        
        # adding a outline for sig. results
        for i,row in enumerate(outcomes):
            for j,col in enumerate(predictors):
                if float(pvalues.at[row,col]) < 0.05:
                    ax.add_patch(plt.Rectangle((j,i), 1, 1, edgecolor='green', fill=False, lw=1))
                elif float(pvalues.at[row,col]) < 0.1:
                    ax.add_patch(plt.Rectangle((j,i), 1, 1, edgecolor='blue', fill=False, lw=1, ls='--'))

    # plotting p-values
    if plot_p: 
        # display the heatmap
        plt.figure(figsize = [15, height])
        plt.title(title)
        pvalues = pvalues.loc[outcomes][predictors]
        ax = sns.heatmap(pvalues, annot=True, cmap="coolwarm")
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        # adding a outline for sig. results
        for i, row in enumerate(outcomes):
            for j, col in enumerate(predictors):
                if pvalues.loc[row, col] < 0.05:
                    ax.add_patch(plt.Rectangle((j,i), 1, 1, edgecolor='green', fill=False, lw=1))
                elif pvalues.loc[row, col] < 0.1:
                    ax.add_patch(plt.Rectangle((j,i), 1, 1, edgecolor='blue', fill=False, lw=1, ls='--'))
        
    # save the figure and show the result
    if save_fig != None: plt.savefig(save_fig, bbox_inches='tight')
    plt.show()