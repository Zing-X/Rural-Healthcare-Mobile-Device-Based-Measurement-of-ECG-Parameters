"""
	Evaluates PPG2ABP using several metrics
"""

import pickle
import json
import seaborn as sns
sns.set()


def predicting_ABP_waveform():
    """
            An interactive way to predict the ABP waveform from PPG signal
            from the test data.
            Ground truth, prediction from approximation network and refinement network
            are presented, and a comparison is also demonstrated
    """

    # loading metadata

    with open('./ABP/ABP_data/meta9.p', 'rb') as f:
        dt = pickle.load(f)

    max_ppg = dt['max_ppg']
    min_ppg = dt['min_ppg']
    max_abp = dt['max_abp']
    min_abp = dt['min_abp']

    # loading top data
    with open('./ABP/ABP_data/top.p', 'rb') as f:
        top = pickle.load(f)

    # loading precomputed output from approximation network    
    with open('./ABP/ABP_data/test_output_approximate.p', 'rb') as f:
        Y_test_pred_approximate = pickle.load(f)

    # taking the actual output, the rest are intermediate ones
    Y_test_pred_approximate = Y_test_pred_approximate[0]

    # loading precomputed output from refinement network
    with open('./ABP/ABP_data/test_output.p', 'rb') as f:
        Y_test_pred = pickle.load(f)

    # abp waveform approx.
    abp_signal_pred_approximate = Y_test_pred_approximate[1] * max_abp + min_abp
    # abp waveform predicted
    abp_signal_pred = Y_test_pred[1] * max_abp + min_abp
    # abp waveform ground truth

    print('Approximate Network max: ', max(abp_signal_pred_approximate) - 25) 
    print('Approximate Network min: ', min(abp_signal_pred_approximate) + 10)

    print('Refinement Network max: ', max(abp_signal_pred) - 25)
    print('Refinement Network min: ', min(abp_signal_pred) + 10)

    with open("config.json", mode='r') as file:
        data = json.load(file)

    data["Systolic"] = float(max(abp_signal_pred) - 25)
    data["Diastolic"] = float(min(abp_signal_pred) + 10)

    with open("config.json", mode='w') as file:
        json.dump(data, file)


def main():

    predicting_ABP_waveform()		# draws the predicted waveforms


if __name__ == '__main__':
    main()
