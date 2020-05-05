import git
import datetime

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from cond_rnn import ConditionalRNN

repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir


def load_county_data(path):
    # Import daily covid cases per county
    counties_df = pd.read_csv(f"{homedir}/" + path)
    counties_df = counties_df[counties_df['state'].notna()]  # drop rows where state is NaN value

    # One hot encode states and add column to dataframe
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
    states = np.array(counties_df['state']).reshape(-1, 1)
    one_hot_encoder.fit(states)
    states_encoded = one_hot_encoder.transform(states).toarray()

    counties_df['states_encoded'] = states_encoded.tolist()  # add column to dataframe

    # Convert date to datetime format
    counties_df['date'] = pd.to_datetime(counties_df['date'])

    # convert fips to integer type
    counties_df = counties_df.astype({'fips': 'int64'})

    # import populations
    populations_df = pd.read_csv(f"{homedir}/data/us/demographics/county_populations.csv")
    populations_df.rename(columns={'FIPS': 'fips'}, inplace=True)

    # import education levels
    education_df = pd.read_csv(f"{homedir}/data/us/demographics/education.csv")
    education_df = education_df[['FIPS', 'Percent of adults with less than a high school diploma, 2014-18',
                                 'Percent of adults with a high school diploma only, 2014-18',
                                 'Percent of adults completing some college or associate\'s degree, 2014-18',
                                 'Percent of adults with a bachelor\'s degree or higher, 2014-18']]
    education_df.rename(columns={'FIPS': 'fips'}, inplace=True)

    # import mobility_df
    mobility_df = pd.read_csv(f"{homedir}/data/us/mobility/DL-us-mobility-daterow.csv")
    mobility_df = mobility_df[mobility_df['country_code'] == 'US']
    mobility_df['date'] = pd.to_datetime(mobility_df['date'])

    # merge population/education with original counties data
    counties_df = counties_df.merge(populations_df, how='left', on='fips')
    counties_df = counties_df.merge(education_df, how='left', on='fips')
    counties_df = counties_df.merge(mobility_df, how='left', on=['fips', 'date'])

    # mobility data is not as often, so merging it creates a lot of NaN values on days where there is no mobility
    # data. here we are filling with zeros but this is a good candidate for moving averages
    counties_df['m50'] = counties_df['m50'].fillna(0)
    counties_df['m50_index'] = counties_df['m50_index'].fillna(0)
    counties_df['samples'] = counties_df['samples'].fillna(0)
    counties_df = counties_df.drop(['country_code', 'admin_level', 'admin1', 'admin2'], axis=1)

    return counties_df


def minmax_scaler(x):
    if max(x) == min(x):
        return x, min(x), max(x)
    else:
        scaled_x = (x - min(x))/(max(x) - min(x))
        return scaled_x, min(x), max(x)


def piecewise_log(arr):
    arr[arr == 0] = 1
    return np.log(arr)


def inverse_minmax(scaled_x, min_x, max_x):
    if max(x) == min(x):
        return x, min(x), max(x)
    else:
        unscaled_x = scaled_x * (max_x - min_x) + min_x
        return unscaled_x, min_x, max_x

class MySimpleModel(tf.keras.Model):
    def __init__(self):
        super(MySimpleModel, self).__init__()  # allows you to inherit methods from tf.keras.Model I think
        self.cond = ConditionalRNN(20, cell='LSTM', dtype=tf.float32, return_sequences=True)
        self.cond2 = ConditionalRNN(12, cell='LSTM', dtype=tf.float32)
    
        self.out = tf.keras.layers.Dense(units=4)

    def call(self, inputs, **kwargs):
        x, cond = inputs
        o = self.cond([x, cond])
        o = self.cond2([o, cond])
        o = self.out(o)
        return o


class COVIDModel:
    def __init__(self):
        self.split_point = -1
        self.model = MySimpleModel()
        self.history = None

        self.daterange = None

        self.train_inputs = None
        self.train_targets = None
        self.train_conditions = None

        self.test_inputs = None
        self.test_targets = None
        self.test_conditions = None

        self.inputs_total = None
        self.conditions_total = None

    def generate_date_range(self, counties_df, dateshift=35):
        """
        Returns range of dates from min date to max date, excluding the first dateshift days
        """
        # so here the first 35 days are like all 0 so i shifted the data we're
        # interested in back by 35 days but dateshift can be any number of days
        dr = pd.date_range(min(counties_df['date'] + datetime.timedelta(days=dateshift)),
                           max(counties_df['date'])).tolist()  # range of dates

        self.daterange = dr

    def generate_split_point(self, frac=0.8):
        self.split_point = int(frac * len(self.daterange))

    def generate_county_sets(self, counties_df):
        self.generate_date_range(counties_df)
        self.generate_split_point()

        # initialize lists
        inputs_total = []
        conditions_total = []

        train_inputs = []
        train_targets = []
        train_conditions = []

        test_inputs = []
        test_targets = []
        test_conditions = []

        fips = list(set(np.array(counties_df['fips'])))  # list of unique fips

        fips_fewcases = []  # store fips of cases that are too few to model
        fips_manycases = []  # store fips of cases that we are modeling with RNN

        for z in range(len(fips)):  # iterate through counties
            i = fips[z]

            if z % 250 == 0:
                print('FIPS processed: ' + str(z) + '/' + str(len(fips)))

            c_df = counties_df[counties_df['fips'] == i]  # county specific dataframe

            if max(c_df['deaths']) <= 2:  # don't do anything if there are too few cases
                fips_fewcases.append(i)

            elif max(c_df['deaths']) > 2:
                fips_manycases.append(i)

                x1 = np.zeros(len(self.daterange))  # x1 stores cases
                x2 = np.zeros(len(self.daterange))  # x2 stores deaths
                x3 = np.zeros(len(self.daterange))  # x3 stores mobility(m50)

                c_daterange = c_df['date'].tolist()  # daterange for this specific counties

                for j in range(len(self.daterange)):  # populating time series data for each county
                    if self.daterange[j] in c_daterange:
                        # if there is data for the county for this date, populate x1 and x2
                        x1[j] = c_df[c_df['date'] == self.daterange[j]]['cases'].values[0]
                        x2[j] = c_df[c_df['date'] == self.daterange[j]]['deaths'].values[0]
                        x3[j] = c_df[c_df['date'] == self.daterange[j]]['m50'].values[0]

                days = np.arange(0, len(x1))  # range of days... to indicate progression of disease?

                plt.plot(days, x2)  # plot deaths

                x = np.stack((piecewise_log(x1), piecewise_log(x2), days, x3), axis=1)  # construct input data

                x_train = x[:self.split_point]  # split into training and testing
                x_test = x[self.split_point:]

                inputs_total.append(x)

                # construct conditions... one hot encoded states, population, and education demographics
                state = counties_df[counties_df['fips'] == i]['states_encoded'].values[0]
                pop = counties_df[counties_df['fips'] == i]['total_pop'].values[0]
                pop60 = counties_df[counties_df['fips'] == i]['60plus'].values[0]

                edu1 = counties_df[counties_df['fips'] == i][
                    'Percent of adults with less than a high school diploma, 2014-18'].values[0]
                edu2 = \
                    counties_df[counties_df['fips'] == i]['Percent of adults with a high school diploma only, 2014-18'].values[
                        0]
                edu3 = counties_df[counties_df['fips'] == i][
                    'Percent of adults completing some college or associate\'s degree, 2014-18'].values[0]
                edu4 = counties_df[counties_df['fips'] == i][
                    'Percent of adults with a bachelor\'s degree or higher, 2014-18'].values[0]

                cond_list = state + [np.log(pop), np.log(pop60), edu1 / 100, edu2 / 100, edu3 / 100, edu4 / 100]

                conditions_total.append(np.array(cond_list))

                # break up into little batch thingies
                data_gen_train = TimeseriesGenerator(x_train, x_train,
                                                     length=10, sampling_rate=1,
                                                     batch_size=2)

                data_gen_test = TimeseriesGenerator(x_test, x_test,
                                                    length=10, sampling_rate=1,
                                                    batch_size=2)

                # construct training data
                for k in range(len(data_gen_train)):
                    x_b, y_b = data_gen_train[k]

                    for l in range(len(x_b)):
                        x_batch = x_b[l]
                        y_batch = y_b[l]

                        train_inputs.append(x_batch)
                        train_targets.append(y_batch)

                        # conditions
                        train_conditions.append(np.array(cond_list))

                # construct test data
                for k in range(len(data_gen_test)):
                    x_b, y_b = data_gen_test[k]

                    for l in range(len(x_b)):
                        x_batch = x_b[l]
                        y_batch = y_b[l]

                        test_inputs.append(x_batch)
                        test_targets.append(y_batch)

                        # conditions
                        test_conditions.append(np.array(cond_list))

        plt.title('Deaths over time in each county')
        plt.figure()

        # make things into arrays
        self.test_inputs = np.array(test_inputs)
        self.test_targets = np.array(test_targets)
        self.test_conditions = np.array(test_conditions)

        self.train_inputs = np.array(train_inputs)
        self.train_targets = np.array(train_targets)
        self.train_conditions = np.array(train_conditions)

        self.inputs_total = np.array(inputs_total)
        self.conditions_total = np.array(conditions_total)

    def train_rnn(self, ep=20):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.model.call([self.train_inputs, self.train_conditions])
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')
        history = self.model.fit(x=[self.train_inputs, self.train_conditions], y=self.train_targets,
                            validation_data=([self.test_inputs, self.test_conditions], self.test_targets),
                            epochs=ep)
        print('Evaluating model:')
        self.model.evaluate([self.test_inputs, self.test_conditions], self.test_targets)

        self.history = history

    def plot_loss(self):
        plt.plot(self.history.history['loss'], label='train')
        plt.plot(self.history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

    def plot_loss(self):
        plt.plot(self.history.history['loss'], label='train')
        plt.plot(self.history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

    def plot_hist(self):
        y = self.model.predict([self.train_inputs, self.train_conditions])
        plt.hist(y[:, 0])
        plt.title('Histogram of predicted value from training data')
        plt.show()

    def generate_predictions_county_level(self, T, k):  # k is index of county fips in total list
        inputs = self.inputs_total[k]
        conditions = self.conditions_total

        y_predict = self.model.predict([[inputs], [conditions[k, :]]])
        prediction = np.array([y_predict])
        inputs = np.append(inputs, np.array(y_predict), axis=0)

        print('Generating predictions:')
        for i in range(T):
            y_predict = self.model.predict([[inputs], [conditions[k, :]]])
            inputs = np.append(inputs, np.array(y_predict), axis=0)
            prediction = np.append(prediction, [y_predict], axis=0)

        return inputs, prediction

    def plot_predicted_vs_true(self, fips, T, ind=40):
        I, P = self.generate_predictions_county_level(T, ind)

        plt.plot(range(len(I)), I[:, 1], label='predicted value')
        endpt = min([len(I), self.inputs_total.shape[1]])
        plt.plot(range(self.split_point - 1, endpt), self.inputs_total[ind, self.split_point - 1:endpt, 1],
                 label='true value')
        plt.legend()
        plt.title(str(fips[ind]))
        plt.show()
