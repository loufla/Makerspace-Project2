## Functions that generate feature values are listed first,
# then helper functions that multiprocess feature functions, convert mp outputs to dicts and add mp outputs to dfs

import numpy as np
import pandas as pd
import multiprocess as mp
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
from datetime import datetime

#need to add dfs to list depending on data you want to process (I've been using one df per day's data)
df_list=[]


def stu_time_marc(df):
    student_list = ['conner', 'rhea', 'yani', 'aashna', 'sara', 'chali', 'natalie', 'rachel', 'xiaoyi', 'hoa',
                    'melissa', 'denise',
                    'rui', 'ji su', 'juan', 'rebecca', 'miaoya', 'helen']
    stu_time_w_marc = {}
    for student in student_list:
        # get df for each student
        df_stu = df.loc[(df['student_id'] == student)]
        df_stu.drop_duplicates(subset=['timestamp'], inplace=True)
        df_stu = df_stu.reset_index(drop=True)

        stu_times = df_stu['timestamp'].tolist()

        # get df for Marc
        df_instructor_temp = df.loc[(df['student_id'] == 'marc') & (df['timestamp'].isin(stu_times))]
        df_instructor_temp.drop_duplicates(subset=['timestamp'], inplace=True)
        df_instructor_temp = df_instructor_temp.reset_index(drop=True)
        instructor_times_temp = df_instructor_temp['timestamp'].tolist()

        df_stu_temp = pd.DataFrame(df_stu.loc[df_stu['timestamp'].isin(instructor_times_temp)])
        df_stu_temp = df_stu_temp.reset_index(drop=True)

        # check proximity of student & Marc
        df_stu_temp['x_compare'] = np.where((df_stu_temp['0_x'] + 0.5 > df_instructor_temp['0_x']) & (
                    df_instructor_temp['0_x'] >= df_stu_temp['0_x'] - 0.5), True, False)
        df_stu_temp['y_compare'] = np.where((df_stu_temp['0_y'] + 0.5 > df_instructor_temp['0_y']) & (
                    df_instructor_temp['0_y'] >= df_stu_temp['0_y'] - 0.5), True, False)

        # get student & instructor shoulders' slopes
        df_stu_temp['slope'] = (df_stu_temp['6_y'] - df_stu_temp['5_y']) / (df_stu_temp['6_x'] - df_stu_temp['5_x'])
        df_instructor_temp['slope'] = (df_instructor_temp['6_y'] - df_instructor_temp['5_y']) / (
                    df_instructor_temp['6_x'] - df_instructor_temp['5_x'])

        df_stu_temp['turned'] = False

        # comparing the slopes of students & instructors
        for row in df_instructor_temp.itertuples():
            df_stu_temp['0_x'].iloc[row.Index]
            x_dist = abs((df_stu_temp.at[row.Index, '0_x']) - (df_instructor_temp.at[row.Index, '0_x']))
            y_dist = abs((df_stu_temp.at[row.Index, '0_y']) - (df_instructor_temp.at[row.Index, '0_y']))
            # checking whether people are adjacent along x or y axis
            if x_dist > y_dist:
                # checking who is on the left side of the other
                if df_stu_temp.at[row.Index, '0_x'] < df_instructor_temp.at[row.Index, '0_x']:
                    # check if turned to each other
                    if (df_stu_temp.at[row.Index, 'slope'] < 0) and (df_instructor_temp.at[row.Index, 'slope'] > 0):
                        df_stu_temp.at[row.Index, 'turned'] = True
                    else:
                        df_stu_temp.at[row.Index, 'turned'] = False
                else:
                    if (df_stu_temp.at[row.Index, 'slope'] > 0) and (df_instructor_temp.at[row.Index, 'slope'] < 0):
                        df_stu_temp.at[row.Index, 'turned'] = True
                    else:
                        df_stu_temp.at[row.Index, 'turned'] = False
            else:
                # checking who is on the left side of the other
                if df_stu_temp.at[row.Index, '0_y'] < df_instructor_temp.at[row.Index, '0_y']:
                    # check if turned to each other
                    if (df_stu_temp.at[row.Index, 'slope'] > 0) and (df_instructor_temp.at[row.Index, 'slope'] < 0):
                        df_stu_temp.at[row.Index, 'turned'] = True
                    else:
                        df_stu_temp.at[row.Index, 'turned'] = False
                else:
                    if (df_stu_temp.at[row.Index, 'slope'] < 0) and (df_instructor_temp.at[row.Index, 'slope'] > 0):
                        df_stu_temp.at[row.Index, 'turned'] = True
                    else:
                        df_stu_temp.at[row.Index, 'turned'] = False

        # getting total no. of timestamps for which student & Marc are collaborating
        time_counts = len(df_stu_temp.loc[(df_stu_temp['x_compare'] == True) & (df_stu_temp['y_compare'] == True) & (
                    df_stu_temp['turned'] == True)])
        # multiplying total by length of time for each timestamp - .67s, adding this to dict
        stu_time_w_marc[student] = time_counts * 0.67
    return stu_time_w_marc

def student_availability(df):
    student_list = ['conner', 'rhea', 'yani', 'aashna', 'sara', 'chali', 'natalie', 'rachel', 'xiaoyi', 'hoa',
                    'melissa', 'denise',
                    'rui', 'ji su', 'juan', 'rebecca', 'miaoya', 'helen']
    student_appearances = {}
    for student in student_list:
        arrival_counter = 0
        curr_time = 0
        last_time = 0
        # creating df for each student
        df_stu = df.loc[(df['student_id'] == student)]
        df_stu.drop_duplicates(subset=['timestamp'], inplace=True)
        df_stu = df_stu.reset_index(drop=True)
        df_stu['timestamp'] = df_stu['timestamp'].map(lambda x: x[11:19])

        for row in df_stu.to_dict('records'):
            curr_time = row['timestamp']
            if arrival_counter == 0:
                arrival_counter = arrival_counter + 1
            if last_time != 0:
                if ((datetime.strptime(curr_time, '%H:%M:%S') - datetime.strptime(last_time,
                                                                                  '%H:%M:%S')) / 3600).seconds >= 2:
                    arrival_counter = arrival_counter + 1
            last_time = curr_time
        student_appearances[student] = arrival_counter

    return student_appearances

#HELPER FUNCTIONS
def multiprocess_functions(function,df_list):
    function = function
    args = df_list
    times=[]

    # figure out how many cores we need
    num_cores = min(mp.cpu_count()-3, len(args))

    # multiprocess the arguments using the function defined above
    with mp.Pool(num_cores) as pool:
        #get list of time dicts
        times=pool.map(function, args)
    return times

#convert list of time dicts into summed final dict that can become new column in df
def time_list_to_dict(time_list,student_list):
    times_final = {'conner': 0,
                   'rhea': 0,
                   'yani': 0,
                   'aashna': 0,
                   'sara': 0,
                   'chali': 0,
                   'natalie': 0,
                   'rachel': 0,
                   'xiaoyi': 0,
                   'hoa': 0,
                   'melissa': 0,
                   'denise': 0,
                   'rui': 0,
                   'ji su': 0,
                   'juan': 0,
                   'rebecca': 0,
                   'miaoya': 0,
                   'helen': 0}
    for student in student_list:
        for dict in time_list:
            times_final[student]=times_final[student]+dict[student]

    return times_final

def dict_to_col(df,time_dict, col_name):
    df[col_name]=pd.Series([])
    df[col_name]=df[col_name].fillna(df['student_id'].map(time_dict))