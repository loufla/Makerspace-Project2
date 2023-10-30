# libraries
from datetime import datetime, timedelta
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#import moviepy.editor as mpy
import icalendar
import pytz

##########
# helper #
##########

def create_dir(target_dir):
  if not os.path.exists(target_dir):
    try:
        os.makedirs(target_dir)
    except:
        pass

##############
# timestamps #
##############

def return_unix_time(curr_time, scale=1000):
    ''' converts a string value into a unix timestamp (millisecond) 
        this code works for the following formats:
            - 2022-03-05 12:42:40.133000-05:00
            - 2022-03-23T14:59:56.533Z '''
    if 'T' in curr_time: curr_time = curr_time.replace('T', ' ')
    if 'Z' in curr_time: curr_time = curr_time.replace('Z', '')
    if '-05:00' in curr_time: curr_time = curr_time.replace('-05:00', '') 
    try:    
        curr_time = datetime.strptime(curr_time, "%Y-%m-%d %H:%M:%S.%f")
        return int(curr_time.timestamp() * scale)
    except Exception as e:
        curr_time = datetime.strptime(curr_time, "%Y-%m-%d %H:%M:%S")
        return int(curr_time.timestamp() * scale)

def add_unix_time_to_df(df, scale='millisecond'): 
    ''' add a column  that represents the unix time of a dataframe '''
    multiplier = 1
    if scale == 'millisecond': multiplier = 1000
    df['unix'] = df.apply(lambda x: return_unix_time(x['timestamp'],scale=multiplier), axis=1)

def which_hour(curr_time):
    ''' converts a string value into a unix timestamp (millisecond) 
        this code works for the following formats:
            - 2022-03-05 12:42:40.133000-05:00
            - 2022-03-23T14:59:56.533Z '''
    if 'T' in curr_time: curr_time = curr_time.replace('T', ' ')
    if 'Z' in curr_time: curr_time = curr_time.replace('Z', '')
    if '-05:00' in curr_time: curr_time = curr_time.replace('-05:00', '') 
    try:    
        curr_time = datetime.strptime(curr_time, "%Y-%m-%d %H:%M:%S.%f")
        return curr_time.hour
    except Exception as e:
        curr_time = datetime.strptime(curr_time, "%Y-%m-%d %H:%M:%S")
        return curr_time.hour

def add_aoi_to_df(df):
    df['aoi'] = df.apply(lambda x: determine_activity((x['0_x'],x['0_y'])), axis=1)

def add_hour_to_df(df):
    df['hour'] = df.apply(lambda x: which_hour(x['timestamp']), axis=1)

def convert_datetime(time_str):
  time_zone = time_str.split('-')[-1]
  time_zone = time_zone.split(':')[0] + time_zone.split(':')[1]
  time_date = time_str.split(' ')[0]
  time_time = time_str.split(' ')[1].split('-')[0]
  reformat_time = '{} {} -{}'.format(time_date,time_time,time_zone)

  try:
    return datetime.strptime(reformat_time, '%Y-%m-%d %H:%M:%S.%f %z')
  except:
    return datetime.strptime(reformat_time, '%Y-%m-%d %H:%M:%S %z')

def obtain_target_times(in_df_student,in_every_sec_freq):
  
  # obtain starting df_freq
  df_freq = in_df_student.copy()
  df_freq.sort_values(by=['timestamp'],inplace=True)
  df_freq.reset_index(inplace=True,drop=True)

  # obtain start_time, end_time and first target_time
  start_time = df_freq.at[0,'timestamp']
  end_time = df_freq.at[len(df_freq)-1,'timestamp']
  target_time = start_time + timedelta(seconds=in_every_sec_freq)

  # obtain target_times
  target_times = [start_time]

  def determine_time_diff(input_time):
    return (input_time - target_time).total_seconds()

  while target_time < end_time:
    df_freq['time_diff'] = df_freq['timestamp'].apply(determine_time_diff)
    df_freq = df_freq[df_freq['time_diff']>=0]
    potential_time = df_freq.at[df_freq['time_diff'].idxmin(),'timestamp']
    if (potential_time - target_time).total_seconds() < in_every_sec_freq:
      target_times.append(potential_time)
    
    target_time = target_time + timedelta(seconds=in_every_sec_freq)

  return target_times

############
# calendar #
############
cal_path = '/content/drive/Shareddrives/2020-Makerspace-tracking/Data/2022-Spr-T519/calendar/Teaching (T519)_harvard.edu_20ijup1o4a8bbbbolk31c7f8jk@group.calendar.google.com.ics'
spr22_start = datetime(2022,1,27,0,0,0)
spr22_end = datetime(2022,5,4,23,59,59)

def clean_calendar(in_cal_path,in_target_start,in_target_end):
    
    # import calendar
    df_calendar = pd.DataFrame(columns=['summary','description','location','dtstart','dtend'])
    cal =  open(in_cal_path, 'r')
    gcal = icalendar.Calendar.from_ical(cal.read())
    for component in gcal.walk():
      if component.name == "VEVENT":
        df_calendar = df_calendar.append({'summary':component.get('summary'),
                                          'description':component.get('description'),
                                          'location':component.get('location'),
                                          'dtstart':component.get('dtstart').dt,
                                          'dtend':component.get('dtend').dt},ignore_index=True)
        
    # remove events that are outside of target time periods
    tz_ny = pytz.timezone('America/New_York')
    target_start_tz = tz_ny.localize(in_target_start)
    target_end_tz = tz_ny.localize(in_target_end)
    
    def test_time_range(test_time,lower_limit,upper_limit):
      if lower_limit <= test_time <= upper_limit:
        return '1'
      else:
        return '0'
    
    def convert_pd_time(pd_time):
      out_time = pd.Timestamp(pd_time).to_pydatetime()
      if out_time.tzinfo!=None:
        return out_time
      else:
        return tz_ny.localize(out_time)
    
    df_calendar['dtstart'] = df_calendar['dtstart'].apply(convert_pd_time)
    df_calendar['dtend'] = df_calendar['dtend'].apply(convert_pd_time)
    df_calendar['sem_ind'] = df_calendar['dtstart'].apply(test_time_range,args=(target_start_tz,target_end_tz))
    df_calendar = df_calendar[df_calendar['sem_ind']=='1']
    df_calendar.reset_index(drop=True,inplace=True)

    
    # remove uncessary info
    df_calendar.drop(index=[0],inplace=True)
    df_calendar = df_calendar[['summary','dtstart','dtend']]
    df_calendar.reset_index(inplace=True,drop=True)
    df_calendar.sort_values(by=['dtstart'],inplace=True)
    
    return df_calendar

#################
# data cleaning #
#################

# define drives
pri_data_dir = '/content/drive/Shared drives/2020-Makerspace-tracking/Data/2022-Spr-T519/poseconnect/'
data_dir = pri_data_dir + 'poseconnect_cleaned/'

# target files
files = ['df_pose_2d.json', 'poses_3d_interpolated.csv']

# variables
tz_ny = pytz.timezone('America/New_York')

def clean_2d_csv(input_2d,input_3d,input_id_map):
    
    # attach student ids
    input_3d['student_id'] = input_3d['pose_track_3d_id'].map(input_id_map)
    
    # remove unidentified
    input_3d = input_3d[input_3d['student_id']!='unidentified']
    
    # remove nan
    input_3d.dropna(how='any',axis=0,inplace=True)
    
    # obtain needed 2d ids
    try:
      input_3d['pose_2d_ids'] = input_3d['pose_2d_ids'].apply(eval)
    except:
      pass
    all_pose_2d_ids = list(input_3d['pose_2d_ids'])
    flat_pose_2d_ids = [item for sublist in all_pose_2d_ids for item in sublist]
    pose_2d_ids = list(set(flat_pose_2d_ids))
    pose_2d_ids_dict = dict(zip(pose_2d_ids,pose_2d_ids))
    
    # start cleaning input_2d
    df_processed = input_2d[['timestamp','pose_2d_id','keypoint_coordinates_2d']]
    
    # remove unneeded pose_2d_ids
    df_processed['pose_2d_id_ind'] = df_processed['pose_2d_id'].map(pose_2d_ids_dict)
    df_processed.dropna(how='any',axis=0,inplace=True)
    df_processed.reset_index(inplace=True,drop=True)
    df_processed = df_processed[['timestamp','pose_2d_id','keypoint_coordinates_2d']]
    
    # extract more info
    def get_camera_id(pose_2d_id):
      return '{}_{}'.format(pose_2d_id.split('_')[0],pose_2d_id.split('_')[1])
    
    def get_video_id(pose_2d_id):
      return '{}_{}_{}'.format(pose_2d_id.split('_')[0],pose_2d_id.split('_')[1],pose_2d_id.split('_')[2])
    
    def get_image_id(pose_2d_id):
      return pose_2d_id.split('_')[3]
    
    df_processed['camera_id'] = df_processed['pose_2d_id'].apply(get_camera_id)
    df_processed['video_id'] = df_processed['pose_2d_id'].apply(get_video_id)
    df_processed['image_id'] = df_processed['pose_2d_id'].apply(get_image_id)
    
    # process keypoints
    def process_keypoints(input_coord_list,target_keypoint_no,target_xyz_no):
      return input_coord_list[target_keypoint_no][target_xyz_no]
    
    # get individual 2d coordinates
    try:
      df_processed['keypoint_coordinates_2d'] = df_processed['keypoint_coordinates_2d'].apply(eval)
    except:
      pass
    df_processed.reset_index(inplace=True,drop=True)
    no_keypoints = len(df_processed.at[0,'keypoint_coordinates_2d'])
    
    for keypoint_no in range(no_keypoints):
      df_processed['{}_x'.format(keypoint_no)] = df_processed['keypoint_coordinates_2d'].apply(process_keypoints,args=(keypoint_no,0))
      df_processed['{}_y'.format(keypoint_no)] = df_processed['keypoint_coordinates_2d'].apply(process_keypoints,args=(keypoint_no,1))
    
    # remove unneeded 2d columns and sort rows
    df_processed = df_processed[['camera_id', 'video_id', 'image_id', 'timestamp', 'pose_2d_id', 
                                '0_x','0_y', '1_x', '1_y', '2_x', '2_y', '3_x', '3_y', '4_x', '4_y', 
                                '5_x','5_y', '6_x', '6_y', '7_x', '7_y', '8_x', '8_y', '9_x', '9_y', 
                                '10_x','10_y', '11_x', '11_y', '12_x', '12_y', '13_x', '13_y', 
                                '14_x', '14_y','15_x', '15_y', '16_x', '16_y']]
    df_processed.sort_values(by=['timestamp','video_id'],inplace=True)
    df_processed.reset_index(inplace=True,drop=True)
    
    return df_processed

def clean_3d_csv(input_3d,input_id_map,input_period_option,input_period_start=None,input_period_end=None,input_period=None):
  
    # attach student ids
    input_3d['student_id'] = input_3d['pose_track_3d_id'].map(input_id_map)
    
    # remove unidentified
    input_3d = input_3d[input_3d['student_id']!='unidentified']
    
    # remove nan
    input_3d.dropna(how='any',axis=0,inplace=True)
    
    # obtain relevant columns
    df_processed = input_3d[['student_id','timestamp','pose_2d_ids','pose_track_3d_id','keypoint_coordinates_3d']]
    df_processed.reset_index(inplace=True,drop=True)
    
    # convert timestamp 
    def convert_timestamp(timestamp_str):
      return tz_ny.localize(datetime.strptime(timestamp_str,'%Y-%m-%dT%H:%M:%S.%fZ'))
    
    df_processed['timestamp'] = df_processed['timestamp'].apply(convert_timestamp)
    
    # Option 1 - add period info
    if input_period_option == 1:  
      
      # test period
      def test_period_range(test_time,lower_limit,upper_limit,period_name):
        if lower_limit <= test_time <= upper_limit:
          return [period_name]
        else:
          return []
    
      df_processed['period_ind'] = df_processed['timestamp'].apply(test_period_range,args=(input_period_start,input_period_end,input_period))
    
      # add period info
      try:
        df_processed['period_info'] = df_processed['period_info'].apply(eval)  
      except:
        df_processed['period_info'] = '[]'
        df_processed['period_info'] = df_processed['period_info'].apply(eval)
      
      df_processed['period_info'] = df_processed['period_info'] + df_processed['period_ind']
    
    # Option 2 - add empty period info
    elif input_period_option == 2: 
      try:
        df_processed['period_info'] = df_processed['period_info'].apply(eval)
      except:
        df_processed['period_info'] = '[]'
        df_processed['period_info'] = df_processed['period_info'].apply(eval)
    
    # process keypoints
    def process_keypoints(input_coord_list,target_keypoint_no,target_xyz_no):
      return input_coord_list[target_keypoint_no][target_xyz_no]
    
    # process 3D info
    df_processed = df_processed[['student_id', 'timestamp', 'period_info', 'pose_2d_ids', 'pose_track_3d_id', 'keypoint_coordinates_3d']]
    
    # get 3d individual coordinates
    try:
      df_processed['keypoint_coordinates_3d'] = df_processed['keypoint_coordinates_3d'].apply(eval)
    except:
      pass
    df_processed.reset_index(inplace=True,drop=True)
    no_keypoints = len(df_processed.at[0,'keypoint_coordinates_3d'])
    
    for keypoint_no in range(no_keypoints):
      df_processed['{}_x'.format(keypoint_no)] = df_processed['keypoint_coordinates_3d'].apply(process_keypoints,args=(keypoint_no,0))
      df_processed['{}_y'.format(keypoint_no)] = df_processed['keypoint_coordinates_3d'].apply(process_keypoints,args=(keypoint_no,1))
      df_processed['{}_z'.format(keypoint_no)] = df_processed['keypoint_coordinates_3d'].apply(process_keypoints,args=(keypoint_no,2))
    
    # remove unneeded 3d columns and sort rows
    df_processed = df_processed[['student_id', 'timestamp', 'period_info', 'pose_2d_ids', 'pose_track_3d_id', 
                                '0_x', '0_y', '0_z', '1_x', '1_y', '1_z', '2_x', '2_y', '2_z', 
                                '3_x', '3_y', '3_z', '4_x', '4_y', '4_z', '5_x', '5_y', '5_z',
                                '6_x', '6_y', '6_z', '7_x', '7_y', '7_z', '8_x', '8_y', '8_z', 
                                '9_x', '9_y', '9_z', '10_x', '10_y', '10_z', '11_x', '11_y', '11_z', 
                                '12_x', '12_y', '12_z', '13_x', '13_y', '13_z', '14_x', '14_y', '14_z', 
                                '15_x', '15_y', '15_z', '16_x', '16_y', '16_z']]
    df_processed.sort_values(by=['timestamp','student_id'],inplace=True)
    df_processed.reset_index(inplace=True,drop=True)
    
    return df_processed

def clean_2d_date(in_target_date):
    
    # import identification data
    id_csv = pd.read_csv('/content/drive/Shareddrives/2020-Makerspace-tracking/Data/2022-Spr-T519/labelled_tracks/{}.csv'.format(in_target_date))
    id_map = dict(zip(list(id_csv['pose_track_3d_id']),list(id_csv['person_identity'])))
    assert len(id_map) == len(id_csv)
    
    # define folders
    target_data_dir = pri_data_dir + '{}/'.format(in_target_date)
    
    # glob video times
    target_time_list = glob.glob(target_data_dir + '*')
    target_time_list = sorted([target_time.split('/')[-1] for target_time in target_time_list])
    
    # combine datasets
    target_cleaned_dir = data_dir + '{}/'.format(in_target_date)
    create_dir(target_cleaned_dir)
    output_csv = target_cleaned_dir + '2d_{}.csv'.format(in_target_date)
    
    # create headers
    cols_2d = ['camera_id', 'video_id', 'image_id', 'timestamp', 'pose_2d_id', 
          '0_x','0_y', '1_x', '1_y', '2_x', '2_y', '3_x', '3_y', '4_x', '4_y', 
          '5_x','5_y', '6_x', '6_y', '7_x', '7_y', '8_x', '8_y', '9_x', '9_y', 
          '10_x','10_y', '11_x', '11_y', '12_x', '12_y', '13_x', '13_y', 
          '14_x', '14_y','15_x', '15_y', '16_x', '16_y']
    
    try:
      chunk = pd.read_csv(output_csv)
    except:
      chunk = pd.DataFrame(columns=cols_2d)
      chunk = chunk.append(dict(zip(cols_2d,cols_2d)),ignore_index=True)
      chunk.to_csv(output_csv, mode="a", header=False, index=False)

    for target_time in tqdm(target_time_list,total=len(target_time_list)):
        target_time_folder = target_data_dir + '{}/'.format(target_time)
        
        print('Reading csv for {}...'.format(target_time))
        target_2d = pd.read_json(target_time_folder + files[0])
        target_3d = pd.read_csv(target_time_folder + files[1])
        
        print('Cleaning csv for {}'.format(target_time))
        
        try:
            cleaned_csv = clean_2d_csv(target_2d,target_3d,id_map)
            print('Saving csv for {}...'.format(target_time))
            cleaned_csv.to_csv(output_csv, mode="a", header=False, index=False)
        except:
            continue

def clean_3d_date(in_target_date):
    
    # import calendar
    df_calendar = pd.read_csv('/content/drive/Shareddrives/2020-Makerspace-tracking/Data/2022-Spr-T519/calendar/cleaned_calendar.csv')
    
    df_calendar['dtstart'] = df_calendar['dtstart'].apply(datetime.fromisoformat)
    df_calendar['dtend'] = df_calendar['dtend'].apply(datetime.fromisoformat)
    
    # obtain relevant calendar info
    target_date_start = tz_ny.localize(datetime.strptime(in_target_date,'%Y-%m-%d'))
    target_date_end = target_date_start + datetime.timedelta(minutes=1439)
    
    def test_time_range(test_time,lower_limit,upper_limit):
        if lower_limit <= test_time <= upper_limit:
            return '1'
        else:
            return '0'
    
    df_calendar['target_date_ind'] = df_calendar['dtstart'].apply(test_time_range,args=(target_date_start,target_date_end))
    
    df_calendar = df_calendar[df_calendar['target_date_ind']=='1']
    df_calendar.reset_index(drop=True,inplace=True)
    
    if len(df_calendar) != 0:
        period_option = 1
        # input period info
        select_idx = 0
        period = df_calendar.at[select_idx,'summary']
        period_start = df_calendar.at[select_idx,'dtstart']
        period_end = df_calendar.at[select_idx,'dtend']
    else:
        period_option = 2
        period = None
        period_start = None
        period_end = None
      
     # import identification data
    id_csv = pd.read_csv('/content/drive/Shareddrives/2020-Makerspace-tracking/Data/2022-Spr-T519/labelled_tracks/{}.csv'.format(in_target_date))
    id_map = dict(zip(list(id_csv['pose_track_3d_id']),list(id_csv['person_identity'])))
    assert len(id_map) == len(id_csv)
    
    
    # define folders
    target_data_dir = pri_data_dir + '{}/'.format(in_target_date)
    
    # glob video times
    target_time_list = glob.glob(target_data_dir + '*')
    target_time_list = sorted([target_time.split('/')[-1] for target_time in target_time_list])
    
    # combine datasets
    target_cleaned_dir = data_dir + '{}/'.format(in_target_date)
    create_dir(target_cleaned_dir)
    output_csv = target_cleaned_dir + '3d_{}.csv'.format(in_target_date)
    
    # create headers
    cols_3d = ['student_id', 'timestamp', 'period_info', 'pose_2d_ids', 'pose_track_3d_id',
               '0_x', '0_y', '0_z', '1_x', '1_y', '1_z', '2_x', '2_y', '2_z', 
               '3_x', '3_y', '3_z', '4_x', '4_y', '4_z', '5_x', '5_y', '5_z',
               '6_x', '6_y', '6_z', '7_x', '7_y', '7_z', '8_x', '8_y', '8_z', 
               '9_x', '9_y', '9_z', '10_x', '10_y', '10_z', '11_x', '11_y', '11_z', 
               '12_x', '12_y', '12_z', '13_x', '13_y', '13_z', '14_x', '14_y', '14_z', 
               '15_x', '15_y', '15_z', '16_x', '16_y', '16_z']
    
    try:
      chunk = pd.read_csv(output_csv)
    except:
      chunk = pd.DataFrame(columns=cols_3d)
      chunk = chunk.append(dict(zip(cols_3d,cols_3d)),ignore_index=True)
      chunk.to_csv(output_csv, mode="a", header=False, index=False)
    
    for target_time in tqdm(target_time_list,total=len(target_time_list)):
      target_time_folder = target_data_dir + '{}/'.format(target_time)
    
      print('Reading csv for {}...'.format(target_time))
      target_3d = pd.read_csv(target_time_folder + files[1])
      
      print('Cleaning csv for {}'.format(target_time))
      try:
        cleaned_csv = clean_3d_csv(target_3d,id_map,period_option,input_period_start=period_start,input_period_end=period_end,input_period=period)
        print('Saving csv for {}...'.format(target_time))
        cleaned_csv.to_csv(output_csv, mode="a", header=False, index=False)
      except:
        continue
    
##########
# Labels #
##########

corners =[(7.75, 12.55),(7.75, 0),(0, 0),(0, 12.55)] # topleft, topright, bottomright, bottomleft

no_vertical = 12
no_horizontal = 25

total_vertical = abs(corners[3][0] - corners[0][0])
total_horizontal = abs(corners[1][1] - corners[0][1])

square_vertical = total_vertical / no_vertical
square_horizontal = total_horizontal / no_horizontal

def get_partition(start,end,separation):
    current = start
    out_list = [current]
    while current > end:
        current-=separation
        out_list.append(current)
    return out_list

first_row = get_partition(corners[0][1],corners[1][1],square_horizontal)
first_row = [point for point in first_row if point>corners[1][1]]

first_col = get_partition(corners[0][0],corners[3][0],square_vertical)
first_col = [point for point in first_col if point>corners[3][0]]

def get_box(top_left,in_vertical,in_horizontal):
    x, y = top_left
    return [top_left,(x,y-in_horizontal),(x-in_vertical,y-in_horizontal),(x-in_vertical,y)] # topleft, topright, bottomright, bottomleft

# obtain grid
grid = []
for point_horizontal in first_row:
    for point_vertical in first_col:
        box = get_box((point_vertical,point_horizontal),square_vertical,square_horizontal)
        if box[3][0] >= (corners[3][0] - square_vertical/2) and box[1][1] >= (corners[1][1] - square_horizontal/2):
          grid.append(box)
          
def determine_within_frame(input_time,frame_start,frame_end):
  if frame_start <= input_time <= frame_end:
    return 1
  else:
    return 0

def determine_within_plot(input_time,plot_times):
  if input_time in plot_times:
    return 1
  else:
    return 0

def determine_box_indicator(in_df_student_frame,in_grid):
  out_df_student_frame = in_df_student_frame.copy()
  out_df_student_frame['box_indicator'] = 0

  for i, row_i in out_df_student_frame.iterrows():

    nose = [row_i['{}_x'.format(keypoint_dict['nose'])],row_i['{}_y'.format(keypoint_dict['nose'])]]

    for box_i, target_box in enumerate(in_grid):
      if within_area(nose,target_box):
        out_df_student_frame.at[i,'box_indicator'] = box_i + 1

  # raise exception for positions with no box
  df_no_box = out_df_student_frame[out_df_student_frame['box_indicator']==0]
  if len(df_no_box) != 0:
    print(df_no_box)
    raise Exception('There are positions with no box!')
  else:
    return out_df_student_frame

def determine_label_indicator(in_df_student_frame):
  out_df_student_frame = in_df_student_frame.copy()
  out_df_student_frame['label_indicator'] = 0
  
  unique_boxes = list(set(out_df_student_frame['box_indicator']))
  
  for unique_box in unique_boxes:
      
    df_box = out_df_student_frame[out_df_student_frame['box_indicator']==unique_box]
    
    unique_identities = list(set(df_box['student_id']))
    
    for unique_identity in unique_identities:
        
      df_identity = df_box[df_box['student_id']==unique_identity]
      
      first_row_id = df_identity.index[0]
      
      out_df_student_frame.at[first_row_id,'label_indicator'] = 1
      
  return out_df_student_frame

########
# AOIs #
########

# define AOIs
laser_position=[(7.5, 8.5),(7.5, 3.5),(6, 3.5),(6, 8.5)] 
soldering_position=[(7.5, 11.5),(7.5, 8.5),(6, 8.5),(6, 11.5)]
sewing_position=[(6.5, 2),(6.5, 0.2),(5.5, 0.2),(5.5, 2)] 
printer_position=[(3, 1.8),(3, 0.2),(0.05, 0.2),(0.05, 1.8)] 
collaboration_position=[(5.5, 10.5),(5.5, 2),(1.7, 2),(1.7, 10.5)] 
tool_position=[(1.4, 8),(1.4, 4.3),(0.2, 4.3),(0.2, 8)] 
office_position=[(1.4, 11.5),(1.4, 8),(0.2, 8),(0.2, 11.5)]

def within_area(test_position,target_area):

  # transform coordinates
  x, y = transform_coordinates(test_position,scale_x,scale_y,translate_x,translate_y)
  
  transformed_area = []
  for target_position in target_area:
    transformed_area.append(transform_coordinates(target_position,scale_x,scale_y,translate_x,translate_y))
  
  
  target_xs = [target[0] for target in transformed_area]
  target_ys = [target[1] for target in transformed_area]

  min_x = min(target_xs)
  max_x = max(target_xs)
  min_y = min(target_ys)
  max_y = max(target_ys)
  if min_x <= x <= max_x and min_y <= y <= max_y:
    return True
  else:
    return False

def determine_activity(student_position):

  activities_booleans_dict = {}
  for activity_idx, activity in enumerate(activities):
    if activity == 'nothing':
      pass
    else:
      target_position = eval('{}_position'.format(activity))
      activities_booleans_dict[activity] = within_area(student_position,target_position)
  
  true_activities = []
  for activity, true_false_value in activities_booleans_dict.items():
    if true_false_value == True:
      true_activities.append(activity)
  
  if len(true_activities) == 0:
    return 'nothing'
  elif len(true_activities) == 1:
    return true_activities[0]
  else:
    return true_activities[0]
    # raise Exception('More than one activity found:' + str(true_activities))
    
########
# Draw #
########

# define drives
pri_frames_dir = '/content/drive/Shared drives/2020-Makerspace-tracking/Analysis/2022-Spr-Week7/student_frames/'
pri_vid_dir = '/content/drive/Shared drives/2020-Makerspace-tracking/Analysis/2022-Spr-Week7/student_videos/'

# draw variables
figsize = (15,15)
floorplan_path = '/content/drive/Shared drives/2020-Makerspace-tracking/Development/Visualization/outputs/EIS-floorplan.png'
w,h = (1154,732)
scale_x = -0.95*w
scale_y = -0.9*h
translate_x = w
translate_y = h

keypoint_dict = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# class variables
dates = ['2022-03-03','2022-03-04','2022-03-05','2022-03-06','2022-03-07','2022-03-08','2022-03-09']
activities = ['laser','soldering','sewing','printer','collaboration','tool','office','nothing']
students = ['conner','denise','juan','chali','melissa','ji su','rachel','hoa','aashna',
            'rhea','yani','helen','natalie','rebecca','sara','miaoya','xiaoyi','rui']
instructors = ['bertrand','iulian','marc','daniel','alaa']

# plot variables
position_size = 1
position_opacity = 200 # 0 to 255

tri_extension_factor = 5
tri_opacity = 25 # 0 to 255

plot_frequency = 5
frame_frequency = 5*60
label_dist_thres = 100
include_activities = False

def create_color_palette(palette_name,target_list):
  
  # check color palette is of suitable length
  orig_color_palette=sns.color_palette(palette_name)
  if len(orig_color_palette) < len(target_list):
    print(len(orig_color_palette),len(target_list))
    raise Exception('Color palette is not long enough!')
  
  # create color palette and dict
  color_palette = sns.color_palette(palette_name, len(target_list)) 
  color_dict = dict(zip(target_list,color_palette))

  return color_palette, color_dict

# activity palette
if include_activities:  
  activity_color_palette, activity_color_dict = create_color_palette("husl",activities)

# students palette
students_detailed = students + ['instructor']
students_detailed_color_palette, students_detailed_color_dict = create_color_palette("tab20",students_detailed)


def create_new_image(floorplan_path, w, h):
    ''' create a new image with the floorplan '''
    lab = Image.open(floorplan_path)
    img = Image.new('RGB', (w,h), (255, 255, 255))
    img.paste(lab, (0,0), mask=lab)
    return img, ImageDraw.Draw(img, 'RGBA')

def transform_coordinates(input_coords,scale_x,scale_y,translate_x,translate_y):
  output_y = abs(input_coords[0])/7 * scale_y + translate_y
  output_x = abs(input_coords[1])/12 * scale_x + translate_x

  return int(output_x), int(output_y)


def draw_positions(in_img,in_draw,student_position,colour,opacity,position_size,student_identity=None,display=False):
  
  # define variables
  pos_x, pos_y = transform_coordinates(student_position,scale_x,scale_y,translate_x,translate_y)

  # draw the student position
  student_ellipse = (pos_x-position_size,pos_y-position_size,pos_x+position_size,pos_y+position_size)
  in_draw.ellipse(student_ellipse, fill=tuple([int(v*255) for v in colour]+[opacity]))

  # label the student identity
  if student_identity != None:
    font = ImageFont.load_default()
    in_draw.text((pos_x-position_size/2, pos_y-position_size/2), student_identity , font=font, fill='black')

  # display
  if display:
    fig = plt.figure(figsize=figsize)
    plt.imshow(np.array(in_img))
    plt.axis('off')
    plt.show()

  return in_img,in_draw

def draw_activities(in_img,in_draw,student_position,colour,opacity,position_size,student_identity=None,student_activity=None,display=False):
  
  # define variables
  pos_x, pos_y = transform_coordinates(student_position,scale_x,scale_y,translate_x,translate_y)

  # draw the student position
  student_ellipse = (pos_x-position_size,pos_y-position_size,pos_x+position_size,pos_y+position_size)
  in_draw.ellipse(student_ellipse, fill=tuple([int(v*255) for v in colour]+[opacity]))

  # label the student identity
  if student_identity != None:
    font = ImageFont.load_default()
    in_draw.text((pos_x-position_size/2, pos_y-position_size/2), student_identity , font=font, fill='black')

  # label the student activity
  if student_activity != None:
    font = ImageFont.load_default()
    in_draw.text((pos_x-position_size/2, pos_y-position_size/2), student_activity , font=font, fill='black')

  # display
  if display:
    fig = plt.figure(figsize=figsize)
    plt.imshow(np.array(in_img))
    plt.axis('off')
    plt.show()

  return in_img,in_draw
    

def draw_equipment(draw,equipment_positions,equipment_name,display=False):
  
  for equipment_position in equipment_positions:

    # define variables
    transformed_position = transform_coordinates(equipment_position,scale_x,scale_y,translate_x,translate_y)
   
    # draw the equipment position
    r = 4
    equipment_ellipse = (transformed_position[0]-r,transformed_position[1]-r,transformed_position[0]+r,transformed_position[1]+r)
    draw.ellipse(equipment_ellipse, fill=(255,0,0), outline='black')
    font = ImageFont.load_default()

    # label the equipment position
    font = ImageFont.load_default()
    draw.text((transformed_position[0]-r/2, transformed_position[1]-r/2), equipment_name , font=font, fill='black')


def draw_dandelions(in_img,in_draw,nose,left_shoulder,right_shoulder,colour,opacity,tri_extension_factor,display=False):
  
  # coordinate transformation
  nose = transform_coordinates(nose,scale_x,scale_y,translate_x,translate_y)
  left_shoulder = transform_coordinates(left_shoulder,scale_x,scale_y,translate_x,translate_y)
  right_shoulder = transform_coordinates(right_shoulder,scale_x,scale_y,translate_x,translate_y)
  
  # determine mid shoulder point
  mid_shoulder_y = (left_shoulder[1]+right_shoulder[1])/2
  mid_shoulder_x = (left_shoulder[0]+right_shoulder[0])/2
  mid_shoulder = (int(mid_shoulder_x),int(mid_shoulder_y))

  # determine triangle
  tri_left = tuple((np.array(nose)-np.array(mid_shoulder))*tri_extension_factor+np.array(left_shoulder))
  tri_right = tuple((np.array(nose)-np.array(mid_shoulder))*tri_extension_factor+np.array(right_shoulder))

  # draw the student orientation triangle
  in_draw.polygon([mid_shoulder, tri_left, tri_right], fill=tuple([int(v*255) for v in colour]+[opacity]))

  # display
  if display:
    fig = plt.figure(figsize=figsize)
    plt.imshow(np.array(in_img))
    plt.axis('off')
    plt.show()

  return in_img,in_draw

def draw_single_frame(in_img,in_draw,in_df_students_frame,in_target_time,display=True,in_save_dir=None):
  
  plotted_students_color_dict = {}

  for i, row_i in tqdm(in_df_students_frame.iterrows(),total=len(in_df_students_frame)):
    
    # get the joints
    nose = [row_i['{}_x'.format(keypoint_dict['nose'])],row_i['{}_y'.format(keypoint_dict['nose'])]]
    left_shoulder = [row_i['{}_x'.format(keypoint_dict['left_shoulder'])],row_i['{}_y'.format(keypoint_dict['left_shoulder'])]]
    right_shoulder = [row_i['{}_x'.format(keypoint_dict['right_shoulder'])],row_i['{}_y'.format(keypoint_dict['right_shoulder'])]]

    # get identity
    student_identity = row_i['student_id']

    # get color
    if student_identity in instructors:
      student_color = students_detailed_color_dict['instructor']
      plotted_students_color_dict['instructor'] = student_color
    elif student_identity in students:
      student_color = students_detailed_color_dict[student_identity]
      plotted_students_color_dict[student_identity] = student_color
    else:
      raise Exception('{} not in known identities!'.format(student_identity))

    if include_activities:
      # determine activity
      student_activity = determine_activity(nose)
      student_color = activity_color_dict[student_activity]
      plotted_students_color_dict[student_identity] = student_color
    
    # draw
    if row_i['label_indicator'] == 1:
        in_img,in_draw = draw_activities(in_img,in_draw,nose,student_color,position_opacity,position_size,student_identity=student_identity)
    else:
      in_img,in_draw = draw_activities(in_img,in_draw,nose,student_color,position_opacity,position_size)
    in_img,in_draw = draw_dandelions(in_img,in_draw,nose,left_shoulder,right_shoulder,student_color,tri_opacity,tri_extension_factor)
  
  # create legend
  legend_handles = []

  for student, color in plotted_students_color_dict.items():
    create_patch = mpatches.Patch(color=color, label=student)
    legend_handles.append(create_patch)

  # display / save
  fig = plt.figure(figsize=figsize)
  plt.imshow(np.array(in_img))
  plt.axis('off')
  plt.title('{}'.format(in_target_time),fontsize=16,color='blue',fontweight='bold')
  plt.legend(handles=legend_handles,bbox_to_anchor=(1.05, 1.0), loc='upper left',prop = {'size' : 25})
  plt.tight_layout()
  if display:
    plt.show()
  if in_save_dir!=None:
    fig.savefig(in_save_dir+'{}.jpg'.format(str(in_target_time).split('.')[0]), dpi=fig.dpi, bbox_inches='tight')

  return in_img,in_draw

def draw_date(in_target_date,in_save_indicator):
  
  # obtain relevant data
  df_data = pd.read_csv(data_dir+'{}/3d_{}.csv'.format(in_target_date,in_target_date))

  # convert datetime
  df_data['timestamp'] = df_data['timestamp'].apply(convert_datetime)
  df_data.sort_values(by=['timestamp'],inplace=True)
  df_data.reset_index(inplace=True,drop=True)

  # obtain frames times
  frame_times = obtain_target_times(df_data,frame_frequency)
  print('{} Frame Times: {}'.format(in_target_date,len(frame_times)))
  print(frame_times)

  for i in range(len(frame_times)-1):
    frame_start = frame_times[i]
    frame_end = frame_times[i+1]
    df_data_frame = df_data.copy()
    df_data_frame['frame_indicator'] = df_data_frame['timestamp'].apply(determine_within_frame,args=(frame_start,frame_end))
    df_data_frame = df_data_frame[df_data_frame['frame_indicator']==1]
    df_data_frame.sort_values(by=['timestamp'],inplace=True)
    df_data_frame.reset_index(inplace=True,drop=True)
    
    # obtain plot times
    plot_times = obtain_target_times(df_data_frame,plot_frequency)
    
    # obtain data for plotting
    df_data_frame['plot_indicator'] = df_data_frame['timestamp'].apply(determine_within_plot,args=(plot_times,))
    df_data_frame = df_data_frame[df_data_frame['plot_indicator']==1]
    df_data_frame.sort_values(by=['timestamp'],inplace=True)
    df_data_frame.reset_index(inplace=True,drop=True)
    df_data_frame = determine_box_indicator(df_data_frame,grid)
    df_data_frame = determine_label_indicator(df_data_frame)

    # load the background image / font; create a new image
    img, draw = create_new_image(floorplan_path, w, h)

    if in_save_indicator:
      # define save dir
      create_dir(pri_frames_dir)
      date_dir = pri_frames_dir + '{}/'.format(in_target_date)
      create_dir(date_dir)
      save_dir = date_dir + 'all/'
      create_dir(save_dir)
    else:
      save_dir = None

    # draw
    print('Drawing for all at {}'.format(frame_start))
    out_img,out_draw = draw_single_frame(img,draw,df_data_frame,frame_start,display=True,in_save_dir=save_dir)
    
def create_student_video(in_fps, in_target_date, in_target_student):

  # define variables
  target_frames_dir = pri_frames_dir + '{}/{}/'.format(in_target_date,in_target_student)
  target_date_dir = pri_vid_dir + '{}/'.format(in_target_date)
  create_dir(target_date_dir)
  target_output_vid = target_date_dir + '{}.mp4'.format(in_target_student)

  # obtain all frames
  frames = sorted(glob.glob(target_frames_dir+'*.jpg'))

  # creating slide for each image
  print('Creating video for {} on {}'.format(in_target_student,in_target_date))
  img_clips = []
  for frame in tqdm(frames,total=len(frames)):
    slide = mpy.ImageClip(frame,duration=1/in_fps)
    img_clips.append(slide)

  # concatenating slides
  video_slides = mpy.concatenate_videoclips(img_clips, method='compose')

  # exporting final video
  video_slides.write_videofile(target_output_vid, fps=in_fps)
    

##########
# Social #
##########

def add_social_interactions(df, verbose=False):
    ''' this function adds three states to a dataframe:
    
        solo: student working alone
        cooperative: student working next to someone (close proximity)
        collaborative: student working with someone (lines of sight intersect)'''
    
    # add the AOI
    add_aoi_to_df(df)
    add_movement(df)
    df['aoi_bin'] = df.apply(lambda x:'middle' \
                             if x['aoi']=='collaboration' \
                             else 'outter', axis=1)
    df['state'] = 'solo' + df['aoi_bin']
    df['is_with'] = ''
    df['jva'] = 0
    df['observed'] = ''

    # go through the data using numpy arrays
    persons = df['person_identity'].to_numpy()
    lshoulderx = df['left_shoulder_x'].to_numpy()
    lshouldery = df['left_shoulder_y'].to_numpy()
    rshoulderx = df['right_shoulder_x'].to_numpy()
    rshouldery = df['right_shoulder_y'].to_numpy()
    nosex = df['nose_x'].to_numpy()
    nosey = df['nose_y'].to_numpy()
    time = df['datetime'].to_numpy()
    aois = df['aoi']
    is_moving = df['is_moving']

    # go through the data
    for i in range(0,df.shape[0]-1):

        # skip if we have already checked it
        if df.at[i,'state'] == 'collaboration': continue

        # get the data
        t = time[i]
        if np.isnan([nosex[i],nosey[i]]).any(): continue
        hx = (lshoulderx[i]+rshoulderx[i])/2.0
        hy = (lshouldery[i]+rshouldery[i])/2.0
        head = transform_coordinate(hx,hy)
        nose = transform_coordinate(nosex[i],nosey[i])

        # check the next entries
        j = i+1
        while(time[j] == time[i] and j < df.shape[0]-1):

            # check if we have data
            if not np.isnan([nosex[j],nosey[j]]).any(): 
                nosej = transform_coordinate(nosex[j],nosey[j])
                
                # close proximity
                if dist_two_points(nose,nosej) < THRESHOLD: 
                    
                    # create a column to keep track of the other person
                    df.at[i,'is_with'] = persons[j]
                    
                    # if instructor, we tag this as help
                    if status_participant(persons[i]) == 'teacher' or \
                       status_participant(persons[j]):
                        if persons[i] in instructors and is_moving[i] == 1: 
                            df.at[i,'state'] = 'help-giving'
                        if persons[i] in instructors and is_moving[i] == 0: 
                            df.at[i,'state'] = 'help-requested'
                        elif persons[j] in instructors and is_moving[i] == 1: 
                            df.at[i,'state'] = 'help-seeking'
                        elif persons[j] in instructors and is_moving[i] == 0: 
                            df.at[i,'state'] = 'help-receiving'
                    
                    else: # test if collaboration or coop
                        # in or outgroup interactions
                        #in_or_out = 'in' if same_group(persons[i], persons[j]) else 'out'
                        hxj = (lshoulderx[j]+rshoulderx[j])/2.0
                        hyj = (lshouldery[j]+rshouldery[j])/2.0
                        headj = transform_coordinate(hxj,hyj)
                        gaze1 = compute_gaze(head,nose)
                        gaze2 = compute_gaze(headj,nosej)
                        if intersect(gaze1,gaze2) != None: 
                            df.at[i,'state'] = 'collaboration'#+in_or_out
                            df.at[i,'jva'] = 1
                        else: df.at[i,'state'] = 'cooperation'#+in_or_out
                
                # not close proximity, but check observation
                else: 
                    lshoulderj = transform_coordinate(lshoulderx[j],lshouldery[j])
                    rshoulderj = transform_coordinate(rshoulderx[j],rshouldery[j])
                    gaze1_extended = compute_gaze(head,nose,scalar=10)
                    shoulderj = [lshoulderj,rshoulderj]
                    if intersect(gaze1_extended, shoulderj):
                        df.at[i,'observed'] = persons[j]
                    
            # check the next row
            j += 1

        # print progress
        if i % 10000 == 0 and verbose:
            sys.stdout.write("\r")
            sys.stdout.write(str(i) + "/"+str(df.shape[0]))
            sys.stdout.flush()
