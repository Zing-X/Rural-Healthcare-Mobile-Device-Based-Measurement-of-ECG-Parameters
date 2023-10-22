"""
    Computes the outputs for test data
"""

from helper_functions import *
from models import UNet, UNetLite, UNetWide40, UNetWide48, UNetDS64, UNetWide64, MultiResUNet1D, MultiResUNetDS
import os

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import pickle
import json
sns.set()

def predicting_ABP_waveform():
    """
            An interactive way to predict the ABP waveform from PPG signal
            from the test data.
            Ground truth, prediction from approximation network and refinement network
            are presented, and a comparison is also demonstrated
    """

    # loading metadata

    with open('./ABP_data/meta9.p', 'rb') as f:
        dt = pickle.load(f)

    max_ppg = dt['max_ppg']
    min_ppg = dt['min_ppg']
    max_abp = dt['max_abp']
    min_abp = dt['min_abp']

    # loading top data
    with open('./ABP_data/top.p', 'rb') as f:
        top = pickle.load(f)

    # loading precomputed output from approximation network    
    with open('./ABP_data/test_output_approximate.p', 'rb') as f:
        Y_test_pred_approximate = pickle.load(f)

    # taking the actual output, the rest are intermediate ones
    Y_test_pred_approximate = Y_test_pred_approximate[0]

    # loading precomputed output from refinement network
    with open('./ABP_data/test_output.p', 'rb') as f:
        Y_test_pred = pickle.load(f)

    # abp waveform approx.
    abp_signal_pred_approximate = Y_test_pred_approximate[0] * max_abp + min_abp
    # abp waveform predicted
    abp_signal_pred = Y_test_pred[0] * max_abp + min_abp
    # abp waveform ground truth

    print('Approximate Network max: ', max(abp_signal_pred_approximate) - 70) 
    print('Approximate Network min: ', min(abp_signal_pred_approximate) - 0 )

    print('Refinement Network max: ', max(abp_signal_pred) - 70)
    print('Refinement Network min: ', min(abp_signal_pred) - 0 )

    with open("config.json", mode='r') as file:
        data = json.load(file)

    data["Systolic"] = float(max(abp_signal_pred) - 70)
    data["Diastolic"] = float(min(abp_signal_pred) - 0 )

    with open("config.json", mode='w') as file:
        json.dump(data, file)

def predict_test_data():
    """
        Computes the outputs for test data
        and saves them in order to avoid recomputing
    """

    length = 1024               # length of signal

    X_test = pickle.load(open(os.path.join('./ABP_data/','top.p'),'rb'))      # loading test data
    # X_test = dt['X_test']
    # Y_test = dt['Y_test']

    mdl1 = UNetDS64(length)                                             # creating approximation network
    mdl1.load_weights(os.path.join('./models/ApproximateNetwork.h5'))   # loading weights
    
    # temp = np.array(mdl1.predict(X_test, verbose=1))            # predicting approximate abp waveform
    # Y_test_pred_approximate = temp[np.newaxis, ...]

    Y_test_pred_approximate = mdl1.predict(X_test,verbose=1)            # predicting approximate abp waveform

    pickle.dump(Y_test_pred_approximate,open('./ABP_data/test_output_approximate.p','wb')) # saving the approxmiate predictions


    mdl2 = MultiResUNet1D(length)                                       # creating refinement network
    mdl2.load_weights(os.path.join('./models/RefinementNetwork.h5'))    # loading weights

    Y_test_pred = mdl2.predict(Y_test_pred_approximate[0], verbose=1)    # predicting abp waveform

    pickle.dump(Y_test_pred,open('./ABP_data/test_output.p','wb'))                 # saving the predicted abp waeforms


def evaluate_BHS_Standard():
    """
            Evaluates PPG2ABP based on
            BHS Standard Metric
    """

    def newline(p1, p2):
        """
        Draws a line between two points

        Arguments:
                p1 {list} -- coordinate of the first point
                p2 {list} -- coordinate of the second point

        Returns:
                mlines.Line2D -- the drawn line
        """
        ax = plt.gca()
        xmin, xmax = ax.get_xbound()

        if(p2[0] == p1[0]):
            xmin = xmax = p1[0]
            ymin, ymax = ax.get_ybound()
        else:
            ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
            ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

        l = mlines.Line2D([xmin, xmax], [ymin, ymax],
                          linewidth=1, linestyle='--')
        ax.add_line(l)
        return l

    def BHS_metric(err):
        """
        Computes the BHS Standard metric

        Arguments:
                err {array} -- array of absolute error

        Returns:
                tuple -- tuple of percentage of samples with <=5 mmHg, <=10 mmHg and <=15 mmHg error
        """

        leq5 = 0
        leq10 = 0
        leq15 = 0

        for i in range(len(err)):

            if(abs(err[i]) <= 5):
                leq5 += 1
                leq10 += 1
                leq15 += 1

            elif(abs(err[i]) <= 10):
                leq10 += 1
                leq15 += 1

            elif(abs(err[i]) <= 15):
                leq15 += 1

        return (leq5*100.0/len(err), leq10*100.0/len(err), leq15*100.0/len(err))

    def calcError(Ytrue, Ypred, max_abp, min_abp, max_ppg, min_ppg):
        """
        Calculates the absolute error of sbp,dbp,map etc.

        Arguments:
                Ytrue {array} -- ground truth
                Ypred {array} -- predicted
                max_abp {float} -- max value of abp signal
                min_abp {float} -- min value of abp signal
                max_ppg {float} -- max value of ppg signal
                min_ppg {float} -- min value of ppg signal

        Returns:
                tuple -- tuple of abs. errors of sbp, dbp and map calculation
        """

        sbps = []
        dbps = []
        maps = []
        maes = []
        gt = []

        hist = []

        for i in (range(len(Ytrue))):
            y_t = Ytrue[i].ravel()
            y_p = Ypred[i].ravel()

            dbps.append(max_abp*abs(min(y_t)-min(y_p)))
            sbps.append(max_abp*abs(max(y_t)-max(y_p)))
            maps.append(max_abp*abs(np.mean(y_t)-np.mean(y_p)))

        return (sbps, dbps, maps)

    # loading test data
    dt = pickle.load(open(os.path.join('./ABP_data/test.p'), 'rb'))
    X_test = dt['X_test']
    Y_test = dt['Y_test']

    # loading meta data
    dt = pickle.load(open(os.path.join('./ABP_data/meta9.p'), 'rb'))
    max_ppg = dt['max_ppg']
    min_ppg = dt['min_ppg']
    max_abp = dt['max_abp']
    min_abp = dt['min_abp']

    Y_pred = pickle.load(open('./ABP_data/test_output.p', 'rb'))						# loading prediction

    (sbps, dbps, maps) = calcError(Y_test, Y_pred, max_abp, min_abp, max_ppg, min_ppg)   # compute errors

    print('----------------------------')
    print('SBP\'s MAE: ',  np.array(sbps).mean())
    print('DBP\'s MAE: ',  np.array(dbps).mean())
    print('----------------------------')

    sbp_percent = BHS_metric(sbps)											# compute BHS metric for sbp
    dbp_percent = BHS_metric(dbps)											# compute BHS metric for dbp
    map_percent = BHS_metric(maps)											# compute BHS metric for map

    print('----------------------------')
    print('|        BHS-Metric        |')
    print('----------------------------')

    print('----------------------------------------')
    print('|     | <= 5mmHg | <=10mmHg | <=15mmHg |')
    print('----------------------------------------')
    print('| DBP |  {} %  |  {} %  |  {} %  |'.format(
        round(dbp_percent[0], 1), round(dbp_percent[1], 1), round(dbp_percent[2], 1)))
    print('| MAP |  {} %  |  {} %  |  {} %  |'.format(
        round(map_percent[0], 1), round(map_percent[1], 1), round(map_percent[2], 1)))
    print('| SBP |  {} %  |  {} %  |  {} %  |'.format(
        round(sbp_percent[0], 1), round(sbp_percent[1], 1), round(sbp_percent[2], 1)))
    print('----------------------------------------')

    '''
		Plot figures
	'''

    ## SBPS ##

    fig = plt.figure(figsize=(18, 4), dpi=120)
    ax1 = plt.subplot(1, 3, 1)
    ax2 = ax1.twinx()
    sns.distplot(sbps, bins=100, kde=False, rug=False, ax=ax1)
    sns.distplot(sbps, bins=100, kde=False, rug=False, ax=ax2)
    ax2.set_yticklabels(['0 \%', '3.67 \%', '7.34 \%',
                         '11.01 \%', '14.67 \%', '18.34 \%', '22.01 \%'])
    ax1.set_xlabel(r'$|$'+'Error'+r'$|$' + ' (mmHg)', fontsize=14)
    ax1.set_ylabel('Number of Samples', fontsize=14)
    ax2.set_ylabel('Percentage of Samples', fontsize=14)
    plt.title('Absolute Error in SBP Prediction', fontsize=18)
    plt.xlim(xmax=60.0, xmin=0.0)
    plt.xticks(np.arange(0, 60+1, 5))
    p1 = [5, 0]
    p2 = [5, 10000]
    newline(p1, p2)
    p1 = [10, 0]
    p2 = [10, 10000]
    newline(p1, p2)
    p1 = [15, 0]
    p2 = [15, 10000]
    newline(p1, p2)
    plt.tight_layout()

    ## DBPS ##

    ax1 = plt.subplot(1, 3, 2)
    ax2 = ax1.twinx()
    sns.distplot(dbps, bins=100, kde=False, rug=False, ax=ax1)
    sns.distplot(dbps, bins=100, kde=False, rug=False, ax=ax2)
    ax2.set_yticklabels(['0 \%', '7.34 \%', '14.67 \%',
                         '22.01 \%', '29.35 \%', '36.68 \%', '44.02 \%'])
    ax1.set_xlabel(r'$|$'+'Error'+r'$|$' + ' (mmHg)', fontsize=14)
    ax1.set_ylabel('Number of Samples', fontsize=14)
    ax2.set_ylabel('Percentage of Samples', fontsize=14)
    plt.title('Absolute Error in DBP Prediction', fontsize=18)
    plt.xlim(xmax=60.0, xmin=0.0)
    plt.xticks(np.arange(0, 60+1, 5))
    p1 = [5, 0]
    p2 = [5, 10000]
    newline(p1, p2)
    p1 = [10, 0]
    p2 = [10, 10000]
    newline(p1, p2)
    p1 = [15, 0]
    p2 = [15, 10000]
    newline(p1, p2)
    plt.tight_layout()

    ## MAPS ##

    ax1 = plt.subplot(1, 3, 3)
    ax2 = ax1.twinx()
    sns.distplot(maps, bins=100, kde=False, rug=False, ax=ax1)
    sns.distplot(maps, bins=100, kde=False, rug=False, ax=ax2)
    ax2.set_yticklabels(['0 \%', '7.34 \%', '14.67 \%', '22.01 \%',
                         '29.35 \%', '36.68 \%', '44.02 \%', '51.36 \%'])
    ax1.set_xlabel(r'$|$'+'Error'+r'$|$' + ' (mmHg)', fontsize=14)
    ax1.set_ylabel('Number of Samples', fontsize=14)
    ax2.set_ylabel('Percentage of Samples', fontsize=14)
    plt.title('Absolute Error in MAP Prediction', fontsize=18)
    plt.xlim(xmax=60.0, xmin=0.0)
    plt.xticks(np.arange(0, 60+1, 5))
    p1 = [5, 0]
    p2 = [5, 10000]
    newline(p1, p2)
    p1 = [10, 0]
    p2 = [10, 10000]
    newline(p1, p2)
    p1 = [15, 0]
    p2 = [15, 10000]
    newline(p1, p2)
    plt.tight_layout()

    plt.show()


def evaluate_AAMI_Standard():
    """
            Evaluate PPG2ABP using AAMI Standard metric	
    """

    def calcErrorAAMI(Ypred, Ytrue, max_abp, min_abp, max_ppg, min_ppg):
        """
        Calculates error of sbp,dbp,map for AAMI standard computation

        Arguments:
                Ytrue {array} -- ground truth
                Ypred {array} -- predicted
                max_abp {float} -- max value of abp signal
                min_abp {float} -- min value of abp signal
                max_ppg {float} -- max value of ppg signal
                min_ppg {float} -- min value of ppg signal

        Returns:
                tuple -- tuple of errors of sbp, dbp and map calculation
        """

        sbps = []
        dbps = []
        maps = []

        for i in (range(len(Ytrue))):
            y_t = Ytrue[i].ravel()
            y_p = Ypred[i].ravel()

            dbps.append(max_abp*(min(y_p)-min(y_t)))
            sbps.append(max_abp*(max(y_p)-max(y_t)))
            maps.append(max_abp*(np.mean(y_p)-np.mean(y_t)))

        return (sbps, dbps, maps)

    # loading test data
    dt = pickle.load(open(os.path.join('./ABP_data/test.p'), 'rb'))
    X_test = dt['X_test']
    Y_test = dt['Y_test']

    # loading metadata
    dt = pickle.load(open(os.path.join('./ABP_data/meta9.p'), 'rb'))
    max_ppg = dt['max_ppg']
    min_ppg = dt['min_ppg']
    max_abp = dt['max_abp']
    min_abp = dt['min_abp']

    Y_pred = pickle.load(open('./ABP_data/test_output.p', 'rb'))						# loading prediction

    (sbps, dbps, maps) = calcErrorAAMI(Y_test, Y_pred,
                                       max_abp, min_abp, max_ppg, min_ppg)		# compute error

    print('---------------------')
    print('|   AAMI Standard   |')
    print('---------------------')

    print('-----------------------')
    print('|     |  ME   |  STD  |')
    print('-----------------------')
    print('| DBP | {} | {} |'.format(
        round(np.mean(dbps), 3), round(np.std(dbps), 3)))
    print('| MAP | {} | {} |'.format(
        round(np.mean(maps), 3), round(np.std(maps), 3)))
    print('| SBP | {} | {} |'.format(
        round(np.mean(sbps), 3), round(np.std(sbps), 3)))
    print('-----------------------')

    '''
		Plotting figures
	'''

    ## DBPS ##

    fig = plt.figure(figsize=(18, 4), dpi=120)
    ax1 = plt.subplot(1, 3, 1)
    ax2 = ax1.twinx()
    sns.distplot(dbps, bins=100, kde=False, rug=False, ax=ax1)
    sns.distplot(dbps, bins=100, kde=False, rug=False, ax=ax2)
    ax2.set_yticklabels(['0 \%', '7.34 \%', '14.67 \%',
                         '22.01 \%', '29.35 \%', '36.68 \%', '44.02 \%'])
    ax1.set_xlabel('Error (mmHg)', fontsize=14)
    ax1.set_ylabel('Number of Samples', fontsize=14)
    ax2.set_ylabel('Percentage of Samples', fontsize=14)
    plt.title('Error in DBP Prediction', fontsize=18)
    plt.xlim(xmax=50.0, xmin=-50.0)
    #plt.xticks(np.arange(0, 60+1, 5))
    plt.tight_layout()

    ## MAPS ##

    #fig = plt.figure(figsize=(6,4), dpi=120)
    ax1 = plt.subplot(1, 3, 2)
    ax2 = ax1.twinx()
    sns.distplot(maps, bins=100, kde=False, rug=False, ax=ax1)
    sns.distplot(maps, bins=100, kde=False, rug=False, ax=ax2)
    ax2.set_yticklabels(['0 \%', '7.34 \%', '14.67 \%', '22.01 \%',
                         '29.35 \%', '36.68 \%', '44.02 \%', '51.36 \%'])
    ax1.set_xlabel('Error (mmHg)', fontsize=14)
    ax1.set_ylabel('Number of Samples', fontsize=14)
    ax2.set_ylabel('Percentage of Samples', fontsize=14)
    plt.title('Error in MAP Prediction', fontsize=18)
    plt.xlim(xmax=50.0, xmin=-50.0)
    #plt.xticks(np.arange(0, 60+1, 5))
    plt.tight_layout()

    ## SBPS ##

    ax1 = plt.subplot(1, 3, 3)
    ax2 = ax1.twinx()
    sns.distplot(sbps, bins=100, kde=False, rug=False, ax=ax1)
    sns.distplot(sbps, bins=100, kde=False, rug=False, ax=ax2)
    ax2.set_yticklabels(['0 \%', '3.67 \%', '7.34 \%',
                         '11.01 \%', '14.67 \%', '18.34 \%', '22.01 \%'])
    ax1.set_xlabel('Error (mmHg)', fontsize=14)
    ax1.set_ylabel('Number of Samples', fontsize=14)
    ax2.set_ylabel('Percentage of Samples', fontsize=14)
    plt.title('Error in SBP Prediction', fontsize=18)
    plt.xlim(xmax=50.0, xmin=-50.0)
    #plt.xticks(np.arange(0, 60+1, 5))
    plt.tight_layout()

    plt.show()


def main():
    predict_test_data()     # predicts and stores the outputs of test data to avoid recomputing

    predicting_ABP_waveform()

    # evaluate_BHS_Standard()			# evaluates under BHS standard

    # evaluate_AAMI_Standard()		# evaluates under AAMI standard

if __name__ == '__main__':
    main()