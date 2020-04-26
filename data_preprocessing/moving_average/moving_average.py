import numpy as np

#calculates a numpy array of moving averages given a time series list
# of data
# Args:
#    numpy [] data: contains original list of data
#    int n: the number of inidividual data points that the average
#            is
# Return:
#    numpy [] avg: moving average of
def moving_avg_np_array(data, n):
    data_size = len(data)
    avg = np.zeros(data_size)
    moving_sum = 0;
    # for the first  n-1 points, just copy the data over to the
    # moving avg list
    for i in range(n-1):
        avg[i] = data[i]
        moving_sum += data[i] / n
    # for the rest of the points sum over the previous n data points
    #  (including current data point)
    for i in range(n-1, len(data)):
        moving_sum += data[i] / n
        avg[i] = moving_sum
        moving_sum -= data[i - (n-1)] / n
    return avg

#calculates a list of moving averages given a time series list
# of data
# Args:
#    list data: contains original list of data
#    int n: the number of inidividual data points that the average
#            is
# Return:
#    list avg: moving average of
def moving_avg_list(data, n):
    avg = []
    moving_sum = 0;
    # for the first  n-1 points, just copy the data over to the
    # moving avg list
    for i in range(n-1):
        avg.append(data[i])
        moving_sum += data[i] / n
    # for the rest of the points sum over the previous n data points
    #  (including current data point)
    for i in range(n-1, len(data)):
        moving_sum += data[i] / n
        avg.append(moving_sum)
        moving_sum -= data[i - (n-1)] / n
    return avg
