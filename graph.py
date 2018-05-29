# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patheffects as PathEffects
import statsmodels.api as sm
import statsmodels.stats.weightstats as ws
from statsmodels.stats.outliers_influence import summary_table
import seaborn as sns
import scipy
import settings
import re
from math import log10

#%%
def graph(y, x='', select='', groupby='', kind=None):
    '''
    Y, X, X_reg, xlabel, ylabel, labels, stats = graph(y, x, select, groupby, kind)
    
    2018-02-16 - luc.vandeputte@arcelormittal.com
    '''
        
    if type(y) is not str:
        print("Please specify y as a string")
        return None, None, None, '', '', None, "Please specify y as a string"
    elif type(x) is not str:
        print("Please specify x as a string")
        return None, None, None, '', '', None, "Please specify x as a string"
    elif type(select) is not str:
        print("Please specify select as a string")
        return None, None, None, '', '', None, "Please specify select as a string"
    elif type(groupby) is not str:
        print("Please specify groupby as a string")
        return None, None, None, '', '', None, "Please specify groupby as a string"
    else:
        # Define labels
        ylabel = y
        xlabel = x
        # Remove spaces after commas in y string
        labels = ylabel.replace(', ', ',').split(',')
        
        # List of possible variable names to be recognized by regular expression engine
        dfs = '|'.join(settings.dataframes.keys())
        # Where are looking for names of variables before dot char
        regex = re.compile(r"(" + dfs + ")\.")     
        # Replace local variable name with the one accessible globally from settings.dataframes
        y = regex.sub(r"settings.dataframes['\1'].", y)
        x = regex.sub(r"settings.dataframes['\1'].", x)
        select = regex.sub(r"settings.dataframes['\1'].", select)
        # Correct 'select' string if we do multiple selections without brackets
        if (select.find('(') == -1) & ((select.find('&') != -1) | (select.find('|') != -1)):
            for group in re.split('&|\|', select):
                select = select.replace(group.strip(), '({})'.format(group.strip()))
        groupby_src = groupby
        groupby = regex.sub(r"settings.dataframes['\1'].", groupby)
        
        # Define Y as a list of y variables
        Y = eval('[' + y + ']')

        X = []
        X_reg  = []
        # Define X
        if x == '':
            # if no x specified, use index with length of first y
            for i in range(len(Y)):
                '''
                If you create new x variable make sure it has same index as variables in dataframe. 
                Otherwise filtering might not work because because of different indices it is refering to. 
                You can also reset index in dataframe: d = d.reset_index(drop=True)
                '''
                #X.append(pd.Series(range(len(Y[i]))))
                i_series = pd.Series(Y[i].index)
                i_series.index = Y[i].index
                X.append(i_series)
            xlabel = 'index'
        elif x.find(',') > 0:
            # if more than 1 x specified -> multiple regression : predicted value on x axis
            #xlabel = 'predicted ' + ylabel
            X_reg = eval('[' + x + ']')
            for i in range(len(Y)):
                X.append(X_reg[0])
        else:
            # use specified x axis
            Xo = eval(x)
            for i in range(len(Y)):
                X.append(Xo)
        
        # Apply selection
        if select != '':
            sel = eval(select)
            for i in range(len(Y)):
                X[i] = X[i][sel]
                Y[i] = Y[i][sel]
            if X_reg:
                for i in range(len(X_reg)):
                    X_reg[i] = X_reg[i][sel]
        # Apply groupby
        if groupby != '':
            # Get grouping series
            gr = eval(groupby)
            # Check how many groups are specified and throw exception if there are too many of them 
            groups_count = len(gr.unique())
            if groups_count > 300:
                raise ValueError("Too many groups have been specified ({}). Try different grouping variable.".format(groups_count))
            # Apply selection
            if select != '':
                gr = gr[sel]
            # Get unique values (sorted)
            gj = sorted(gr.unique())
            # Store original values (Y, X and labels)
            Y_orig = Y
            X_orig = X
            labels_orig = labels
            # Prepare empty lists
            Y = []
            X = []
            labels = []
            # Make sub groups
            k = 0
            for i in range(len(Y_orig)):
                for j in gj:
                    s = (gr==j)
                    Y.append(Y_orig[i][s])
                    X.append(X_orig[0][s])
                    #labels.append(labels_orig[i] + ' (' + groupby_src + ' = ' + str(j) + ')')
                    labels.append( str(j) ) #fix by GRKE
                    k = k + 1

        # Statistical info
        stats = pd.DataFrame()
        for i in range(len(Y)):
            if not Y[i].name:
                Y[i].name = labels[i]
            stats = stats.append(Y[i].describe())
        # Use the labels as index
        stats = stats.set_index(pd.Series(labels))
        # Adapt order of columns if dataframe has columns
        if 'max' in stats.columns:
            stats = stats[['count','mean','std','min','25%','50%','75%','max']]

        # Show graphs    
#        if kind == None:
#            for i in range(len(Y)):
#                #if sum(X[i]==Y[i]) == len(Y[i]):
#                if sum(abs(X[i]-Y[i])) == 0:
#                    plt.plot(X[i], Y[i], '-')
#                else:
#                    plt.plot(X[i], Y[i], 'd')
#        elif kind == 'hist':
#            for i in range(len(Y)):
#                plt.hist(Y[i])
#        
#        plt.grid(True)
#        plt.legend(labels)
#        plt.xlabel(xlabel)
#        plt.show()

        return Y, X, X_reg, xlabel, ylabel, labels, stats
#%%
def graph_hist(y, bins='auto', select='', groupby='',
             cumulative=False, percentage=False, histtype='bar', 
             xlim=None, ylim=None, size=(10, 6), ax=None):
    """ 
    stats, tests = graph_hist(y, bins='auto', select='', groupby='',
                                 cumulative=False, percentage=False, histtype='bar',
                                 xlim=None, ylim=None, size=(10, 6), ax=None)

    Examples: am.graph.graph_hist('d.C_tap_r')
              am.graph.graph_hist('d.C_tap_r, d.C_tap_c')
              am.graph.graph_hist('d.C_tap_r, d.C_tap_c', select='d.C_tap_r > 0')
              am.graph.graph_hist('d.C_tap_r', groupby='d.cv')

    2018-02-18 - luc.vandeputte@arcelormittal.com
    """

    Y, X, X_reg, xlabel, ylabel, labels, stats = graph(y, '', select, groupby)

    if type(y) is not str:
        print("Please specify y as a string")
        return None, ax
    elif type(bins) is not str:
        print("Please specify bins as a string")
        return None, ax
    elif type(select) is not str:
        print("Please specify select as a string")
        return None, ax
    elif type(groupby) is not str:
        print("Please specify groupby as a string")
        return None, ax
    else: 
        if select == '':
            title = 'Histogram ' + y
        else:
            title = 'Histogram ' + y + '\n(' + select + ')'
        
        # Create axis if not given in the input
        if ax is None:
            fig, ax = plt.subplots(figsize=size)
        
        # Check if bins is number and do the conversion if necessary
        # Estimators: auto, fd, doane, scott, rice, sturges, sqrt (same as in Excel)
        if not bins.strip():
            bins = 'auto'
        elif bins.isnumeric():
            bins = int(bins)
        else:
            # If proper pattern found then calculate range base on given numbers 
            ma = re.match("(-?\d*\.{0,1}\d+):(-?\d*\.{0,1}\d+):(-?\d*\.{0,1}\d+)", bins)
            if ma:
                try:
                    bin_min = float(ma.group(1))
                    bin_step = float(ma.group(2))
                    bin_max = float(ma.group(3))
                    bins = np.arange(bin_min, bin_max + bin_step, bin_step)
                except ValueError:    
                    bins = 'auto'
            else:
                if bins not in ['auto', 'fd', 'doane', 'scott', 'rice', 'sturges', 'sqrt']:
                    bins = 'auto'

        for i in range(len(Y)):
            if Y[i].dtype.name not in ['object', 'datetime64[ns]']: #TODO: check with categorical data - convert object to category? #di = Y[i].dropna().astype('category')
                di = Y[i].dropna()
                # Plot histogram
                #ax.hist(Y[i].dropna(), bins=bins, range=None, label=labels[i], cumulative=cumulative, normed=percentage, histtype=histtype)
                if percentage:
                    # If normed then draw KDE
                    sns.distplot(di, bins=bins, label=labels[i], hist_kws={"histtype": histtype, "linewidth": 3, "cumulative":cumulative, "normed":percentage}, kde_kws={"cumulative":cumulative}, ax=ax)
                else:
                    sns.distplot(di, bins=bins, label=labels[i], kde=False, hist_kws={"histtype": histtype, "linewidth": 3, "cumulative":cumulative, "normed":percentage}, kde_kws={"cumulative":cumulative}, ax=ax)
            
        ax.grid(True)
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel(ylabel)
    
        if xlim is not None:
            ax.xlim(xlim)
    
        if ylim is not None:
            ax.xlim(ylim)
        
        if percentage:
            # Change normed values in range 0-1 to 0-100% using proper formatter
            if cumulative:
                formatter = FuncFormatter(lambda v, pos: "{:3.0f}%".format(v * 100))
            else:
                formatter = FuncFormatter(lambda v, pos: "{:3.3f}%".format(v * 100))  
            ax.yaxis.set_major_formatter(formatter)

        plt.show();
        
        # Tests
        tests = pd.DataFrame(columns=['', 'test name', 'test statistic', 'p-value', 'conclusion'])
        for i in range(len(Y)):
            di = Y[i].dropna()
            if di.dtype.name not in ['object', 'datetime64[ns]'] and di.count() >= 3:
                stat, pvalue = scipy.stats.shapiro(di)
                # https://plot.ly/python/normality-test/
                # Shapiro-Wilk normality test
                if pvalue >= 0.05:
                    conclusion = 'Normal distribution'
                else:
                    # 95% confidence the data does not fit the normal distribution
                    conclusion = 'Not normal distribution'
                # Append results    
                tests = tests.append({'': labels[i], 'test name':'Shapiro-Wilk normality test', 'test statistic': stat, 'p-value': pvalue, 'conclusion': conclusion}, ignore_index=True)
        # Do the t-test if there are 2 independent scores
        if len(Y) == 2:
            # Calculates the t-test for the means of two independent samples of scores
            d1 = Y[0].dropna()
            d2 = Y[1].dropna()
            if d1.dtype.name not in ['object', 'datetime64[ns]'] and d2.dtype.name not in ['object', 'datetime64[ns]']:
                r = scipy.stats.ttest_ind(d1, d2)
                if r[1] >= 0.05:
                    conclusion = 'Mean of the two distributions are not different from each other'
                else:
                   conclusion = 'Mean of the two distributions are different and statistically significant'
            # Append results
            tests = tests.append({'': y, 'test name':'t-test', 'test statistic': r.statistic, 'p-value': r.pvalue, 'conclusion': conclusion}, ignore_index=True)
        # Use first column as index
        tests = tests.set_index('')
        # Return stats and tests
        return stats, tests
#%%
def graph_xy(y, x='', select='', groupby='',
             bissectrice=False, regression=False, constant=None,
             xlim=None, ylim=None, size=(10, 6), ax=None):
    """ Color modifying function
    
     Args:
        color_tuple (str): An RGB or RGBA tuple/array of float values
        factor (float): Factor to change color brightness (darker<1<lighter).
            
    Returns:
        tuple: An RGB tuple of float values after modification
        
    Examples:
    """

    Y, X, X_reg, xlabel, ylabel, labels, stats = graph(y, x, select, groupby)
    info = str(stats)

    if select == '':
        title = ylabel + ' vs. ' + xlabel
    else:
        title = ylabel + ' vs. ' + xlabel + '\n(' + select + ')'

    # Create axis if not given in the input
    if ax is None:
        fig, ax = plt.subplots(figsize=size)

    rlabel = []
    models = []
    
    # Determine min and max x value for bissectrice
    X_min = min(X[0])
    X_max = max(X[0])
    for i in range(len(X)):
        X_min = min(X_min, min(X[i]))
        X_max = max(X_max, max(X[i]))
    
    if not X_reg:
        # Only 1 x variable selected
        for i in range(len(Y)):
            # Plot the points
            pc = ax.scatter(y=Y[i].values, x=X[i].values, label=labels[i])
            # Add regression
            if regression:
                # Select only rows where x and y are not null
                s = (~Y[i].isnull()) & (~X[i].isnull())
                ys = Y[i][s]
                xs = X[i][s]
                # Regression with constant or through 0 ?
                if constant is None:
                    Xs = sm.add_constant(xs)
                else:
                    Xs = sm.add_constant(xs, has_constant=constant)
                # Calculate regression
                model = sm.OLS(ys, Xs).fit()
                # Determine label
                rlabel_str = '{:g} + {:g} * {:s}'.format(model.params[0], model.params[1], xlabel)
                rlabel_str = rlabel_str.replace(' + -', ' - ')
                rlabel.append(rlabel_str)
                # Summary table with all influence and outlier measures
                st, data, ss2 = summary_table(model, alpha=0.05) # 95% confidence
                fittedvalues = data[:,2]
                predict_mean_se  = data[:,3]
                predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
                predict_ci_low, predict_ci_upp = data[:,6:8].T
                # Color of the points marker
                scatter_color = pc.get_facecolor()[-1]
                # Uses slightly diffent color for the regression line
                new_color  = _mod_color_tuple(scatter_color, 0.7)
                # Plot regression line
                #ax.plot(xs, model.predict(Xs), label=rlabel[i])
                ax.plot(xs, fittedvalues, color=new_color, label=rlabel[i])
                # Upper and lower 95% prediction limits
                if len(Y) == 1: # too many lines on the chart if there is more than one function
                    xp_s, yp_s = zip(*sorted(zip(xs, predict_ci_low))) # sort in order to create nice line
                    if '95% prediction limits' in ax.get_legend_handles_labels()[1]: # don't add same labels to legend
                        ax.plot(xp_s, yp_s, 'r:')
                    else:
                        ax.plot(xp_s, yp_s, 'r:', label='95% prediction limits')
                    xp_s, yp_s = zip(*sorted(zip(xs, predict_ci_upp)))
                    ax.plot(xp_s, yp_s, 'r:')
                    # Upper and lower 95% confidence bounds
                    xp_s, yp_s = zip(*sorted(zip(xs, predict_mean_ci_low)))
                    if '95% confidence bounds' in ax.get_legend_handles_labels()[1]:
                        ax.plot(xp_s, yp_s, 'r--')
                    else:
                        ax.plot(xp_s, yp_s, 'r--', label='95% confidence bounds')
                    xp_s, yp_s = zip(*sorted(zip(xs, predict_mean_ci_upp)))
                    ax.plot(xp_s, yp_s, 'r--')
                # Add regression info to output
                models.append(model)
                info = info + '\n\n\n' + 'Regression formula for ' + labels[i] + ', coefficients available in models[' + str(i) + '].params' + '\n' + str(model.summary(xname=['const', xlabel]))
    else:
        # Multiple x variables selected - calculate multiple regression as Y[0] ~ X_reg
        # if more than 1 x specified -> multiple regression : predicted value on x axis
        x_names = [x.strip() for x in xlabel.split(',')]
        xlabel = 'predicted ' + ylabel
        
        X_df = pd.concat(X_reg, axis=1)
        s = (~Y[0].isnull()) & (~X_df.isnull().any(axis=1))
        Ys = Y[0][s]
        if constant is None:
            Xs = sm.add_constant(X_df[s])
        else:
            Xs = sm.add_constant(X_df[s], has_constant=constant)
        # Calculate regression
        model = sm.OLS(Ys, Xs).fit()
        # Add additional label to list if there is itercept in the list of params
        if model.params.index[0] == 'const':
            x_names = ['const'] + x_names
        # Determine label
        rlabel_str = ''
        for i in range(len(model.params)):
            if i == 0:
                if model.params.index[0] == 'const':
                    # First param is intercept
                    rlabel_str += '{:g}'.format(model.params[0])
                else:
                    rlabel_str += '{:g}*{:s}'.format(model.params[0], x_names[0])
            else:
                rlabel_str += ' + {:g}*{:s}'.format(model.params[i], x_names[i]) #model.params.index[i]
        rlabel_str = rlabel_str.replace(' + -', ' - ')
        rlabel.append(rlabel_str)
        # Plot points
        ax.scatter(x=model.predict(Xs), y=Ys, label=rlabel_str, marker='x')
        # Summary table with all influence and outlier measures
        st, data, ss2 = summary_table(model, alpha=0.05) # 95% confidence
        fittedvalues = data[:,2]
        predict_mean_se  = data[:,3]
        predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
        predict_ci_low, predict_ci_upp = data[:,6:8].T
        # Upper and lower 95% prediction limits
        xp_s, yp_s = zip(*sorted(zip(fittedvalues, predict_ci_low))) # sort in order to create nice line
        if '95% prediction limits' in ax.get_legend_handles_labels()[1]: # don't add same labels to legend
            ax.plot(xp_s, yp_s, 'r:')
        else:
            ax.plot(xp_s, yp_s, 'r:', label='95% prediction limits')
        xp_s, yp_s = zip(*sorted(zip(fittedvalues, predict_ci_upp)))
        ax.plot(xp_s, yp_s, 'r:')
        # Upper and lower 95% confidence bounds
        xp_s, yp_s = zip(*sorted(zip(fittedvalues, predict_mean_ci_low)))
        if '95% confidence bounds' in ax.get_legend_handles_labels()[1]:
            ax.plot(xp_s, yp_s, 'r--')
        else:
            ax.plot(xp_s, yp_s, 'r--', label='95% confidence bounds')
        xp_s, yp_s = zip(*sorted(zip(fittedvalues, predict_mean_ci_upp)))
        ax.plot(xp_s, yp_s, 'r--')
        # Add regression info to output
        models.append(model)
        info = info + '\n\n\n' + 'Regression formula for ' + labels[0] + ', coefficients available in models[0].params' + '\n' + str(model.summary(xname=x_names))
        # Determine min and max x value for bissectrice
        X_min = min(fittedvalues)
        X_max = max(fittedvalues)
    # Add bissectrice
    if bissectrice:
        # Plot bissectrice
        #plt.plot([X_min, X_max], [X_min, X_max], 'k', ax=ax)
        ax.plot([X_min, X_max], [X_min, X_max], 'k')
    
    ax.grid(True)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_xlim(ylim)

    plt.show();

    return stats, models, info
#%%
def graph_trend(y, x='', select='', groupby='', mf=False, mf_type='mean', mf_window=50, mf_with_data=False, constant=None,
             xlim=None, ylim=None, size=(10, 6), ax=None):
    """ stats = graph_trend(y, x=None, select=None, groupby=None, mf=False, mf_type='mean', mf_window=50, mf_with_data=False, constant=None,
                               xlim=None, ylim=None, size=(10, 6), ax=None)

    Examples: am.graph.graph_trend('d.C_tap_r', 'd.C_tap_c')
              am.graph.graph_trend('d.C_tap_r', 'd.C_tap_c', , 'd.C_tap_c', '(d.C_tap_r < 2000)')
              am.graph.graph_trend('d.C_tap_r', 'd.C_tap_c', ylim=[-1000,15000])

    2018-02-16 - luc.vandeputte@arcelormittal.com
    """

    Y, X, X_reg, xlabel, ylabel, labels, stats = graph(y, x, select, groupby)

    if select == '':
        title = ylabel + ' vs. ' + xlabel
    else:
        title = ylabel + ' vs. ' + xlabel + '\n(' + select + ')'

    # Create axis if not given in the input
    if ax is None:
        fig, ax = plt.subplots(figsize=size)

    for i in range(len(Y)):
        # Plot the points
        if mf:
            # Calculate data for given rolling function 
            if mf_type == 'mean': 
                rolling = Y[i].rolling(window=mf_window).mean()
            elif mf_type == 'median':
                rolling = Y[i].rolling(window=mf_window).median()
            elif mf_type == 'std':
                rolling = Y[i].rolling(window=mf_window).std()
            elif mf_type == 'sum':
                rolling = Y[i].rolling(window=mf_window).sum()
            # Plot moving function    
            p = ax.plot(X[i], rolling, label='rolling {}'.format(labels[i])) #, color='r'
            hex_color = p[-1].get_color()
            new_color = _mod_color_hex(hex_color, 0.6) #lighter color
            if mf_with_data:
                if X[i].dtype.name == 'datetime64[ns]' or Y[i].dtype.name == 'datetime64[ns]':
                    ax.plot_date(y=Y[i], x=X[i], label=labels[i], marker='o', ms=1, color=new_color)
                else:
                    ax.scatter(y=Y[i].values, x=X[i].values, label=labels[i], s=1, color=new_color)
        else:
            ax.plot(X[i], Y[i], label=labels[i])
    
    ax.grid(True)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_xlim(ylim)

    plt.show();

    return stats
#%%
def graph_boxplot(y, select='', groupby='', constant=None,
             xlim=None, ylim=None, size=(10, 6), ax=None, srt=None):
    """ stats = graph_boxplot(y, x=None, select=None, groupby=None, constant=None,
                               xlim=None, ylim=None, size=(10, 6), ax=None)

    Examples: am.graph.graph_boxplot('d.C_tap_r')

    2018-02-16 - luc.vandeputte@arcelormittal.com
    """

    Y, X, X_reg, xlabel, ylabel, labels, stats = graph(y, '', select, groupby)
    
    # Remove nans from data in Y list
    Y = [y.dropna() for y in Y]

    if select == '':
        title = ylabel
    else:
        title = ylabel + '\n(' + select + ')'
		
    if groupby != '':
       title = title + ' (grouped by ' + groupby + ')'

    # Create axis if not given in the input
    if ax is None:
        fig, ax = plt.subplots(figsize=size)

    # Box properties
    boxprops = dict(color='#0B0BFF', linewidth=1)
    # Whisker properties
    whiskerprops = dict(color='#0B0BFF', linewidth=1)
    # Flier properties
    flierprops = dict(marker='+', markersize=10, markerfacecolor='r', linestyle='none', markeredgecolor='r', alpha=0.5)
    # Cap properties
    capprops = dict(color='#0B0BFF', linewidth=1)
    # Median properties
    medianprops = dict(color='#FF1010', linewidth=1)
    
    #auxiliary vars for sorting
    cnt = stats['count']
    mns = stats['50%']
    #makes sorting
    if srt is not None:
        if srt:
            mns, cnt, Y, labels = zip(*sorted(zip( stats['50%'], stats['count'],  Y, labels), reverse=True))
                       
    # required for the boxplot (drop the indexes)
    Y = [series.values for series in Y]
    # Plot the points
    ax.boxplot(Y, notch=True, patch_artist=False, labels=labels, 
                    boxprops=boxprops, whiskerprops=whiskerprops, flierprops=flierprops, capprops=capprops, medianprops=medianprops)
    # Set grid on
    ax.grid(True)
    #ax.legend()
    ax.set_title(title)
    #ax.set_xlabel(xlabel)
    #ax.set_ylabel(ylabel)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_xlim(ylim)

    # add means and counts directly on the graph
    pos = np.arange( len(Y) ) + 1
    upperLabels = [str(int(s)) for s in cnt]
    meanLabels  = [s for s in mns]
    y_min, y_max = ax.get_ylim()
    delta = y_max - y_min
    for tick, label in zip(range( len(Y) ), ax.get_xticklabels()):
        txt1 = ax.text(pos[tick], y_min, upperLabels[tick],
             horizontalalignment='center', size='x-small', weight='bold',
             color='darkgreen')
        txt1.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
        txt2 = ax.text(pos[tick]+0.25, meanLabels[tick]+delta*0.01, format(meanLabels[tick],'.5g'),
             horizontalalignment='center', size='x-small', weight='bold',
             color='darkgreen')
        txt2.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

    plt.show();

    return stats

#%%
def graph_mono_boxplot(y, select='', bins='auto', constant=None,
             xlim=None, ylim=None, title='', size=(10, 6), ax=None):
    
    # initial data processing
    Y0, X0, xreg, xlabel, ylabel, labels0, stats0 = graph(y, '', select, '')
    # security check - only 1 variable allowed
    if( len(Y0) > 1 ):
        print( "monoboxplot mode requires 1 column to be chosen !!!" )
        return None
    # Remove nans from data in Y list
    Y0 = [y.dropna() for y in Y0]
    # Create binning for the boxplot
    low = stats0['min']
    high= stats0['max']
    nBins = float('nan')
    jmp = float('nan')
    ma = re.match("(\d+):(\d+):(\d+)", bins)
    # 1st option - number of bins
    if( bins.isnumeric() ):
         nBins = int(bins)
         jmp = int( float(high - low) / nBins )
    # 2nd option - exact bins
    elif( ma ):
        low = int(ma.group(1))
        jmp = int(ma.group(2))
        high = int(ma.group(3))
        nBins= int( (high-low)/jmp )
    # 3-rd option - fully automat
    else:
        nBins = int((high - low) * ( float(stats0['count'])**(1./3.)) / (3.45 * float(stats0['std'])))
        jmp = int( (high - low) / nBins )
    # final creation of bins
    BINS = list(np.arange(low,high,jmp))
    # Create axis if not given in the input
    if ax is None:
        fig, ax = plt.subplots(figsize=size)
    if xlim is not None:
        ax.xlim(xlim)
    if ylim is not None:
        ax.xlim(ylim)
    if select == '':
        title = ylabel
    else:
        title = ylabel + '\n(' + select + ')'  
    
    boxprops = dict(color='#0B0BFF', linewidth=1)       # Box properties
    whiskerprops = dict(color='#0B0BFF', linewidth=1)   # Whisker properties
    flierprops = dict(marker='+', markersize=10, markerfacecolor='r', linestyle='none', markeredgecolor='r', alpha=0.5) # Flier properties
    capprops = dict(color='#0B0BFF', linewidth=1)       # Cap properties
    medianprops = dict(color='#FF1010', linewidth=1)    # Median properties
    # set axis
    ax.grid(True)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(ylabel)
    # create empty and fill group-by-group
    yy    = []
    stats = pd.DataFrame([], columns=['count','mean','std','min','25%','50%','75%','max'])
    labels= []
    for idx, bbb in enumerate(BINS):
        selct = '(' + y + '>' + str(int(bbb)) + ') & ('+ y + '<' + str(int(bbb+jmp)) + ')'
        Y, X, xreg, xlabel1, ylabel1, labels1, stat = graph(y, '', selct, '')
        # in case bin is not empty
        #print(stat['count'].values[0])
        if( stat['count'].values[0] >0 ):
            lab = format( float(bbb), '.5g') + ' - ' + format( float(bbb+jmp), '.5g') 
            yy.append( Y[0] )
            labels.append( lab )
            stats = stats.append( stat )      
    #check data exception problem
    yy = [ series.values for series in yy]
    ax.boxplot(yy, notch=True, patch_artist=False, labels=labels, 
               boxprops=boxprops, whiskerprops=whiskerprops, flierprops=flierprops, capprops=capprops, medianprops=medianprops)
    # make x tick well visible
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("left")
        tick.set_rotation(-45)
        
    # add means and counts directly on the graph
    pos = np.arange( len(yy) ) + 1 
    lowerLabels = [str(int(s)) for s in stats['count']]
    meanLabels  = [s for s in stats['50%']]
    y_min, y_max = ax.get_ylim()
    delta = y_max - y_min
    for tick, label in zip(range( len(yy) ), ax.get_xticklabels()):
        txt1 = ax.text(pos[tick], y_min, lowerLabels[tick],
             horizontalalignment='center', size='x-small', weight='bold',
             color='darkgreen')
        txt1.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
        txt2 = ax.text(pos[tick]+0.25, meanLabels[tick]+delta*0.01, format(float(meanLabels[tick]),'.5g'),
             horizontalalignment='center', size='x-small', weight='bold',
             color='darkgreen')
        txt2.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
        pass

    plt.show();
    return stats

#%%
def graph_violinplot(y, select='', groupby='', constant=None,
             xlim=None, ylim=None, size=(10, 6), ax=None):
    """ stats = graph_violinplot(y, select=None, groupby=None, constant=None,
                               xlim=None, ylim=None, size=(10, 6), ax=None)

    Examples: am.graph.graph_violinplot('d.C_tap_r')

    2018-02-16 - luc.vandeputte@arcelormittal.com
    """

    Y, X, X_reg, xlabel, ylabel, labels, stats = graph(y, '', select, groupby)
    
    # Remove nans from Y list
    Y = [y.dropna() for y in Y]

    if select == '':
        title = ylabel
    else:
        title = ylabel + '\n(' + select + ')'

    # Create axis if not given in the input
    if ax is None:
        fig, ax = plt.subplots(figsize=size)
    # Plot graph
    ax.violinplot(Y, showmeans=False, showmedians=True)
    
    # Set grid on
    ax.grid(True)
    #ax.legend()
    ax.set_title(title)
    #ax.set_xlabel(xlabel)
    #ax.set_ylabel(ylabel)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_xlim(ylim)

    plt.show();

    return stats
#%% 
def graph_bar(y, x='', select='', groupby='', constant=None,
             xlim=None, ylim=None, size=(10, 6), ax=None):
    """ stats = graph_bar(y, x=None, select=None, groupby=None, constant=None,
                               xlim=None, ylim=None, size=(10, 6), ax=None)

    Examples: am.graph.graph_bar('d.C_tap_r')

    2018-02-16 - luc.vandeputte@arcelormittal.com
    """

    Y, X, X_reg, xlabel, ylabel, labels, stats = graph(y, x, select, groupby)
    
    # Remove nans from Y list
    Y = [y.dropna() for y in Y]

    if select == '':
        title = ylabel
    else:
        title = ylabel + '\n(' + select + ')'

    # Create axis if not given in the input
    if ax is None:
        fig, ax = plt.subplots(figsize=size)

    # Box properties
    boxprops = dict(color='#0B0BFF', linewidth=1)
    # Whisker properties
    whiskerprops = dict(color='#0B0BFF', linewidth=1)
    # Flier properties
    flierprops = dict(marker='+', markersize=10, markerfacecolor='r', linestyle='none', markeredgecolor='r', alpha=0.5)
    # Cap properties
    capprops = dict(color='#0B0BFF', linewidth=1)
    # Median properties
    medianprops = dict(color='#FF1010', linewidth=1)
    # Plot the points
    #ax.boxplot(Y, notch=True, patch_artist=False, labels=labels, 
    #                boxprops=boxprops, whiskerprops=whiskerprops, flierprops=flierprops, capprops=capprops, medianprops=medianprops)
    
    X = [ series.values for series in X]
    Y = [ series.values for series in Y]
    
    # case 1 y and 0 x
    if( x=='' and (',' not in y) ):
        Y = Y[0]
        X = range(len(X[0]))
        ax.bar(X, Y)
    # case 1 y and 1 x
    elif( (',' not in x) and (',' not in y) ):
        Y = Y[0]
        X = X[0]
        ax.bar(X, Y)
    # case N y and 1 x
    #elif( (',' not in x) and (',' in y) ):
        
    # Set grid on
    ax.grid(True)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_xlim(ylim)

    plt.show();

    return stats

def _mod_color_hex(color_hex, factor):
    """ Color modifying function
    
     Args:
        hex_color (str): Hex color in format #RRGGBB.
        factor (float): Factor to change color brightness. darker<1<lighter
            
    Returns:
        str: Hex string representing color modified by the factor.
        
    Examples:
    """
    r = max(min(255, int(int(color_hex[1:3], 16)*factor)), 0)
    g = max(min(255, int(int(color_hex[3:5], 16)*factor)), 0)
    b = max(min(255, int(int(color_hex[5:7], 16)*factor)), 0)
    return '#{:02X}{:02X}{:02X}'.format(r, g, b)

def _mod_color_tuple(color_tuple, factor):
    """ Color modifying function
    
     Args:
        color_tuple (str): An RGB or RGBA tuple/array of float values
        factor (float): Factor to change color brightness (darker<1<lighter).
            
    Returns:
        tuple: An RGB tuple of float values after modification
        
    Examples:
    """
    r = max(min(1, color_tuple[0]*factor), 0)
    g = max(min(1, color_tuple[1]*factor), 0)
    b = max(min(1, color_tuple[2]*factor), 0)
    return (r, g, b)