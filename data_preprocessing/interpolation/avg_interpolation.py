import numpy as np

#calculates a numpy array of moving averages given a time series list
# of data
# Args:
#    numpy [] data: contains original list of data
#    int n: the number of inidividual data points that the average
#            is
# Return:
#    numpy [] avg: moving average of
def avg_interpolation_np_array(data, threshold):
    data_size = len(data)
    interp_data = np.zeros(data_size)
    # for all of the points if the number of cases is signifcantly different
    # from both the previous number of cases OR the next number of cases,
    # take the average of
    # the previous day and next day cases and assign it to the current day
    for i in range(1, len(data) - 1):
        prev_percent_change = 0
        next_percent_change = 0
        if (data[i-1] != 0):
            prev_percent_change = abs(float(data[i] - data[i-1])) / data[i-1]
        if (data[i+1] != 0):
            next_percent_change = abs(float(data[i] - data[i+1])) / data[i+1]
        if (prev_percent_change > threshold or next_percent_change > threshold):
            interp_data[i] = (data[i+1] + data[i-1]) / 2
        else:
            interp_data[i] = data[i]
    return interp_data
