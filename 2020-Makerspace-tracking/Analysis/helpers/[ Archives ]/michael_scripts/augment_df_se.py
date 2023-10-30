#!/usr/bin/env python
# coding: utf-8

## Libraries and Installs
import os
import collections
import numpy as np
import pandas as pd
import ast
from datetime import datetime, timedelta
import icalendar
import tqdm
import scipy

# base folder path
base_path = ''

# if we are on google colab, we mount the drive
if 'google.colab' in str(get_ipython()):
  from google.colab import drive
  drive.mount('/content/drive')
  base_path = './drive/Shareddrives/2020-Makerspace-tracking'

# if we are running it locally, we use the standard gdrive path
# (you will have to update this path)
else: base_path = '/Users/MikeH/Library/CloudStorage/GoogleDrive-michaelhu@college.harvard.edu/Shared drives/2020-Makerspace-tracking/'

# folders we'll be working with
agg_path = os.path.join(base_path, 'Data', '2022-Spr-T519', 'aggregated')
data_path = os.path.join(base_path, 'Data', '2022-Spr-T519', 'poseconnect')
analysis_path = os.path.join(base_path, 'Analysis', '2022-Spr-Week7')

scores_path = os.path.join(agg_path, 'participants_scores.csv')
scores_df = pd.read_csv(scores_path)
scores_df.head()


# Sensor data
# load the script
script = os.path.join(base_path, 'Analysis', 'scripts', 'augment_df.py')
get_ipython().run_line_magic('run', '"$script"')


# Grab Data
def grab_data(dataset):
    folder = os.path.join(data_path, 'poseconnect_cleaned')
    scores_df = pd.read_csv(scores_path)
    for dir in os.listdir(folder):
        if '2022' in dir:
            subfolder = os.path.join(folder, dir)
            for subfile in os.listdir(subfolder):
                # we only care about the 3d reconstructed data
                if subfile.endswith('.csv') and '3d' in subfile and "summary" not in subfile:
                    print(subfile)
                    path = os.path.join(subfolder, subfile)
                        
                    # we read the data and add AOI columns
                    data = pd.read_csv(path)
                    dataset[subfile] = data
                    
dataset = dict()
grab_data(dataset)


## Helpers ##
joint_names = {
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

def get_joint(row, joint) -> tuple():
    if joint not in joint_names:
        raise ValueError("Invalid joint name.")
    joint_ind = joint_names[joint]
    return (row[f'{joint_ind}_x'], row[f'{joint_ind}_y'], row[f'{joint_ind}_z'])

def next_weekday(d, weekday):
    days_ahead = weekday - d.weekday()
    if days_ahead <= 0: # Target day already happened this week
        days_ahead += 7
    return d + timedelta(days_ahead)

def csv_path(file_name):
    return 'csv_files/' + file_name

aois = {'laser': [(7.5, 8.5),(7.5, 3.5),(6, 3.5),(6, 8.5)], 
        'soldering': [(7.5, 11.5),(7.5, 8.5),(6, 8.5),(6, 11.5)], 
        'sewing': [(6.5, 2),(6.5, 0.2),(5.5, 0.2),(5.5, 2)],
        'printer': [(3, 1.8),(3, 0.2),(0.05, 0.2),(0.05, 1.8)], 
        'collaboration': [(5.5, 10.5),(5.5, 2),(1.7, 2),(1.7, 10.5)],
        'tool': [(1.4, 8),(1.4, 4.3),(0.2, 4.3),(0.2, 8)], 
        'office': [(1.4, 11.5),(1.4, 8),(0.2, 8),(0.2, 11.5)]}

def in_workspace(center, threshold):
    for name in aois:
        workspace = aois[name]
        if (center[0] > workspace[0][0] - threshold and center[0] < workspace[1][0] + threshold and
            center[1] > workspace[2][1] - threshold and center[1] < workspace[0][1] + threshold):
            return name
    return "nothing"
    

teachers = set(['bertrand', 'marc', 'iulian', 'daniel', 'alaa'])

def near_who(center, student_id, last_positions, threshold):
    if student_id not in last_positions:
        last_positions[student_id] = center
        return None
    
    nearest_id = None
    nearest_distance = None
    
    for id, position in last_positions.items():
        if id == student_id:
            continue
        
        distance = ((center[0] - position[0]) ** 2 + (center[1] - position[1]) ** 2) ** 0.5
        
        if distance <= threshold and (nearest_distance is None or distance < nearest_distance):
            nearest_id = id
            nearest_distance = distance
            
    last_positions[student_id] = center
    
    if nearest_id in teachers:
        return "TEACHERS"
    else: 
        return "STUDENTS"


# ## Correlations

def hours_til_thursday(time_string):
    # Parse the input time string
    dt = datetime.fromisoformat(time_string)

    # Find the next Thursday
    next_thursday = next_weekday(dt, 3)

    # Calculate the number of hours until the next Thursday
    hours_until_thursday = (next_thursday - dt).total_seconds() / 3600

    # Calculate the multiplier based on the number of days until the next Thursday
    return hours_until_thursday

def write_early():
    print("Writing early bird data...")
    earliness_dict = collections.defaultdict(int)
    
    for filename in dataset:
        data = dataset[filename]
        
        for index, row in data.iterrows():
            student = row['student_id']
            time = row['timestamp']
            
            earliness_dict[student] += hours_til_thursday(time) / 24
                    
    df = pd.DataFrame.from_dict(earliness_dict, orient='index', columns=['earliness'])
    df.to_csv(csv_path('early_bird.csv'), index=True, index_label='student_id')

def write_appearances():
    print("Writing appearances data...")
    counts = collections.defaultdict(int)

    # Loop over each dataframe and count the number of times each student appears
    for filename in dataset:
        data = dataset[filename]
        # Get the counts for this dataframe
        student_counts = data['student_id'].value_counts().to_dict()
        # Add the counts to the dictionary
        for student_id, count in student_counts.items():
            counts[student_id] += count

    # Convert the counts to a dataframe
    counts_df = pd.DataFrame.from_dict(counts, orient='index', columns=['count'])
    counts_df.to_csv(csv_path('counts.csv'), index=True, index_label='student_id')


def shake_knees(knees, hips, student_id, last_position, lower_threshold, upper_threshold, hip_threshold) -> bool:
    """ Detect the shaking of the knees. """
    if student_id not in last_position:
        last_position[student_id] = knees + hips
        return False
    
    delta_knees = np.linalg.norm(np.array(last_position[student_id][:2]) - np.array(knees))
    delta_hips = np.linalg.norm(np.array(last_position[student_id][2:]) - np.array(hips))
    last_position[student_id] = knees + hips
    
    return (lower_threshold < delta_knees < upper_threshold and delta_hips < hip_threshold)

def write_knee_shakes(weighed_by_time=True, weighed_by_work=True, weighed_by_surroundings=True):
    print("Writing knee shaking data...")
    nervousness_dict = collections.defaultdict(int)
    nervousness_dict_time = collections.defaultdict(int)
    nervousness_dicts_work = {'nothing': collections.defaultdict(int), 'work': collections.defaultdict(int)}
    for aoi in aois:
        nervousness_dicts_work[aoi] = collections.defaultdict(int)
    nervousness_dict_peers = collections.defaultdict(int)
    nervousness_dict_instructors = collections.defaultdict(int)
    
    last_position = dict()
    center_positions = dict()

    for filename in dataset:
        data = dataset[filename]
        
        for _, row in data.iterrows():
            student = row['student_id']
            right_knee, left_knee = get_joint(row, 'right_knee'), get_joint(row, 'left_knee')
            right_hip, left_hip = get_joint(row, 'right_hip'), get_joint(row, 'left_hip')
            center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
            
            if shake_knees((right_knee, left_knee), (right_hip, left_hip), student, last_position, 0.1, 0.2, 0.05):
                nervousness_dict[student] += 1
                if weighed_by_time:
                    nervousness_dict_time[student] += 1 / hours_til_thursday(row['timestamp'])
                if weighed_by_work:
                    workspace = in_workspace(center, 0.1)
                    nervousness_dicts_work[workspace][student] += 1
                    if workspace != 'nothing':
                        nervousness_dicts_work['work'][student] += 1
                if weighed_by_surroundings:
                    near = near_who(center, student, center_positions, 0.5)
                    if near == "TEACHERS":
                        nervousness_dict_instructors[student] += 1
                    elif near == "STUDENTS":
                        nervousness_dict_peers[student] += 1
    
    if weighed_by_time:
        df_early = pd.DataFrame.from_dict(nervousness_dict_time, orient='index', columns=['knee_shakes_time'])
        df_early.to_csv(csv_path('knee_shakes_time.csv'), index=True, index_label='student_id')
        
    if weighed_by_work:
        for key in nervousness_dicts_work:
            df_work = pd.DataFrame.from_dict(nervousness_dicts_work[key], orient='index', columns=[f'knee_shakes_{key}'])
            df_work.to_csv(csv_path(f'knee_shakes_{key}.csv'), index=True, index_label='student_id')
        
    if weighed_by_surroundings:
        df_peers = pd.DataFrame.from_dict(nervousness_dict_peers, orient='index', columns=['knee_shakes_peers'])
        df_peers.to_csv(csv_path('knee_shakes_peers.csv'), index=True, index_label='student_id')
        
        df_instructors = pd.DataFrame.from_dict(nervousness_dict_instructors, orient='index', columns=['knee_shakes_instructors'])
        df_instructors.to_csv(csv_path('knee_shakes_instructors.csv'), index=True, index_label='student_id')
    
    df = pd.DataFrame.from_dict(nervousness_dict, orient='index', columns=['knee_shakes'])
    df.to_csv(csv_path('knee_shakes.csv'), index=True, index_label='student_id')


def detect_hand_near_face(hand_joint, nose_joint, threshold):
    hand_face_distance = np.sqrt(np.sum([(hand_joint[i] - nose_joint[i])**2 for i in range(3)]))

    return hand_face_distance < threshold

def write_hand_near_face(weighed_by_time=True, weighed_by_work=True, weighed_by_surroundings=True):
    print("Writing hand near face data...")
    nervousness_dict = collections.defaultdict(int)
    nervousness_dict_time = collections.defaultdict(int)
    nervousness_dicts_work = {'nothing': collections.defaultdict(int), 'work': collections.defaultdict(int)}
    for aoi in aois:
        nervousness_dicts_work[aoi] = collections.defaultdict(int)
    nervousness_dict_peers = collections.defaultdict(int)
    nervousness_dict_instructors = collections.defaultdict(int)
    
    center_positions = dict()

    for filename in dataset:
        data = dataset[filename]
        
        for _, row in data.iterrows():
            student = row['student_id']
            right_hand, left_hand = get_joint(row, 'right_wrist'), get_joint(row, 'left_wrist')
            nose = get_joint(row, 'nose')
            
            if detect_hand_near_face(right_hand, nose, 0.1) or detect_hand_near_face(left_hand, nose, 0.1):
                nervousness_dict[student] += 1
                if weighed_by_time:
                    nervousness_dict_time[student] += 1 / hours_til_thursday(row['timestamp'])
                left_hip, right_hip = get_joint(row, 'left_hip'), get_joint(row, 'right_hip')
                
                center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
                if weighed_by_work:
                    workspace = in_workspace(center, 0.1)
                    nervousness_dicts_work[workspace][student] += 1
                    if workspace != 'nothing':
                        nervousness_dicts_work['work'][student] += 1
                if weighed_by_surroundings:
                    near = near_who(center, student, center_positions, 0.5)
                    if near == "TEACHERS":
                        nervousness_dict_instructors[student] += 1
                    elif near == "STUDENTS":
                        nervousness_dict_peers[student] += 1
                    
    if weighed_by_time:
        df_early = pd.DataFrame.from_dict(nervousness_dict_time, orient='index', columns=['hand_near_face_time'])
        df_early.to_csv(csv_path('hand_near_face_time.csv'), index=True, index_label='student_id')
        
    if weighed_by_work:
        for key in nervousness_dicts_work:
            df_work = pd.DataFrame.from_dict(nervousness_dicts_work[key], orient='index', columns=[f'hand_near_face_{key}'])
            df_work.to_csv(csv_path(f'hand_near_face_{key}.csv'), index=True, index_label='student_id')
        
    if weighed_by_surroundings:
        df_peers = pd.DataFrame.from_dict(nervousness_dict_peers, orient='index', columns=['hand_near_face_peers'])
        df_peers.to_csv(csv_path('hand_near_face_peers.csv'), index=True, index_label='student_id')
        
        df_instructors = pd.DataFrame.from_dict(nervousness_dict_instructors, orient='index', columns=['hand_near_face_instructors'])
        df_instructors.to_csv(csv_path('hand_near_face_instructors.csv'), index=True, index_label='student_id')
    
    df = pd.DataFrame.from_dict(nervousness_dict, orient='index', columns=['hand_near_face'])
    df.to_csv(csv_path('hand_near_face.csv'), index=True, index_label='student_id')


def feet_taps(ankles, knees, student_id, lower_threshold, upper_threshold, knee_threshold, last_position) -> bool:
    right_ankle, left_ankle = ankles
    right_knee, left_knee = knees
    
    if student_id not in last_position:
        last_position[student_id] = right_ankle, left_ankle, right_knee, left_knee
        return False
    else:
        last_right_ankle, last_left_ankle, last_right_knee, last_left_knee = last_position[student_id]
        last_position[student_id] = right_ankle, left_ankle, right_knee, left_knee
        
        delta_rankle = [last_right_ankle[i] - right_ankle[i] for i in range(3)]
        delta_lankle = [last_left_ankle[i] - left_ankle[i] for i in range(3)]
        delta_rknee = [last_right_knee[i] - right_knee[i] for i in range(3)]
        delta_lknee = [last_left_knee[i] - left_knee[i] for i in range(3)]
    
    # Calculate the magnitude of the difference between the knee positions
    delta_rankle_magnitude = np.sqrt(delta_rankle[0]**2 + delta_rankle[1]**2 + delta_rankle[2]**2)
    delta_lankle_magnitude = np.sqrt(delta_lankle[0]**2 + delta_lankle[1]**2 + delta_lankle[2]**2)
    delta_rknee_magnitude = np.sqrt(delta_rknee[0]**2 + delta_rknee[1]**2 + delta_rknee[2]**2)
    delta_lknee_magnitude = np.sqrt(delta_lknee[0]**2 + delta_lknee[1]**2 + delta_lknee[2]**2)
    
    return (lower_threshold < delta_rankle_magnitude < upper_threshold and delta_rknee_magnitude < knee_threshold) or \
           (lower_threshold < delta_lankle_magnitude < upper_threshold and delta_lknee_magnitude < knee_threshold)


def write_feet_taps(weighed_by_time=True, weighed_by_work=True, weighed_by_surroundings=True):
    print("Writing feet taps data...")
    nervousness_dict = collections.defaultdict(int)
    nervousness_dict_time = collections.defaultdict(int)
    nervousness_dicts_work = {'nothing': collections.defaultdict(int), 'work': collections.defaultdict(int)}
    for aoi in aois:
        nervousness_dicts_work[aoi] = collections.defaultdict(int)
    nervousness_dict_peers = collections.defaultdict(int)
    nervousness_dict_instructors = collections.defaultdict(int)
    
    last_position = dict()
    center_positions = dict()

    for filename in dataset:
        data = dataset[filename]
        
        for _, row in data.iterrows():
            student = row['student_id']
            right_ankle, left_ankle = get_joint(row, 'right_ankle'), get_joint(row, 'left_ankle')
            right_knee, left_knee = get_joint(row, 'right_knee'), get_joint(row, 'left_knee')
            
            if feet_taps((right_ankle, left_ankle), (right_knee, left_knee), student, 0.1, 0.2, 0.05, last_position):
                nervousness_dict[student] += 1
                if weighed_by_time:
                    nervousness_dict_time[student] += 1 / hours_til_thursday(row['timestamp'])
                left_hip, right_hip = get_joint(row, 'left_hip'), get_joint(row, 'right_hip')
                
                center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
                if weighed_by_work:
                    workspace = in_workspace(center, 0.1)
                    nervousness_dicts_work[workspace][student] += 1
                    if workspace != 'nothing':
                        nervousness_dicts_work['work'][student] += 1
                if weighed_by_surroundings:
                    near = near_who(center, student, center_positions, 0.5)
                    if near == "TEACHERS":
                        nervousness_dict_instructors[student] += 1
                    elif near == "STUDENTS":
                        nervousness_dict_peers[student] += 1
                
    if weighed_by_time:
        df_early = pd.DataFrame.from_dict(nervousness_dict_time, orient='index', columns=['feet_taps_time'])
        df_early.to_csv(csv_path('feet_taps_time.csv'), index=True, index_label='student_id')
        
    if weighed_by_work:
        for key in nervousness_dicts_work:
            df_work = pd.DataFrame.from_dict(nervousness_dicts_work[key], orient='index', columns=[f'feet_taps_{key}'])
            df_work.to_csv(csv_path(f'feet_taps_{key}.csv'), index=True, index_label='student_id')
        
    if weighed_by_surroundings:
        df_peers = pd.DataFrame.from_dict(nervousness_dict_peers, orient='index', columns=['feet_taps_peers'])
        df_peers.to_csv(csv_path('feet_taps_peers.csv'), index=True, index_label='student_id')
        
        df_instructors = pd.DataFrame.from_dict(nervousness_dict_instructors, orient='index', columns=['feet_taps_instructors'])
        df_instructors.to_csv(csv_path('feet_taps_instructors.csv'), index=True, index_label='student_id')
    
    df = pd.DataFrame.from_dict(nervousness_dict, orient='index', columns=['feet_taps'])
    df.to_csv(csv_path('feet_taps.csv'), index=True, index_label='student_id')


def leaning_backwards(hips, shoulders, knees) -> bool:
    left_hip, right_hip = hips
    left_shoulder, right_shoulder = shoulders
    right_knee, left_knee = knees
    
    # Calculate the center of the shoulders
    center_shoulder = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2, (left_shoulder[2] + right_shoulder[2]) / 2)
    
    # Calculate the center of the hips
    center_hip = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2, (left_hip[2] + right_hip[2]) / 2)
    
    # Determine which way the person is facing
    if right_knee[2] < right_hip[2] and left_knee[2] < left_hip[2]:
        # Facing forwards
        return center_hip[2] < center_shoulder[2]
    elif right_knee[2] > right_hip[2] and left_knee[2] > left_hip[2]:
        # Facing backwards
        return center_hip[2] > center_shoulder[2]
    # Unknown facing direction
    return False


def write_leaning_backwards(weighed_by_time=True, weighed_by_work=True, weighed_by_surroundings=True):
    print("Writing leaning backwards data...")
    not_working_dict = collections.defaultdict(int)
    not_working_dict_time = collections.defaultdict(int)
    nervousness_dicts_work = {'nothing': collections.defaultdict(int), 'work': collections.defaultdict(int)}
    for aoi in aois:
        nervousness_dicts_work[aoi] = collections.defaultdict(int)
    not_working_dict_peers = collections.defaultdict(int)
    not_working_dict_instructors = collections.defaultdict(int)
    
    center_positions = dict()

    for filename in dataset:
        data = dataset[filename]
        
        for _, row in data.iterrows():
            student = row['student_id']
            left_hip, right_hip = get_joint(row, 'left_hip'), get_joint(row, 'right_hip')
            left_shoulder, right_shoulder = get_joint(row, 'left_shoulder'), get_joint(row, 'right_shoulder')
            left_knee, right_knee = get_joint(row, 'left_knee'), get_joint(row, 'right_knee')
            center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
            
            if leaning_backwards((left_hip, right_hip), (left_shoulder, right_shoulder), (left_knee, right_knee)):
                not_working_dict[student] += 1
                if weighed_by_time:
                    not_working_dict_time[student] += 1 / hours_til_thursday(row['timestamp'])
                if weighed_by_work:
                    workspace = in_workspace(center, 0.1)
                    nervousness_dicts_work[workspace][student] += 1
                    if workspace != 'nothing':
                        nervousness_dicts_work['work'][student] += 1
                if weighed_by_surroundings:
                    near = near_who(center, student, center_positions, 0.5)
                    if near == "TEACHERS":
                        not_working_dict_instructors[student] += 1
                    elif near == "STUDENTS":
                        not_working_dict_peers[student] += 1
                
    if weighed_by_time:
        df_early = pd.DataFrame.from_dict(not_working_dict_time, orient='index', columns=['leaning_backwards_time'])
        df_early.to_csv(csv_path('leaning_backwards_time.csv'), index=True, index_label='student_id')
        
    if weighed_by_work:
        for key in nervousness_dicts_work:
            df_work = pd.DataFrame.from_dict(nervousness_dicts_work[key], orient='index', columns=[f'leaning_backwards_{key}'])
            df_work.to_csv(csv_path(f'leaning_backwards_{key}.csv'), index=True, index_label='student_id')
        
    if weighed_by_surroundings:
        df_peers = pd.DataFrame.from_dict(not_working_dict_peers, orient='index', columns=['leaning_backwards_peers'])
        df_peers.to_csv(csv_path('leaning_backwards_peers.csv'), index=True, index_label='student_id')
        
        df_instructors = pd.DataFrame.from_dict(not_working_dict_instructors, orient='index', columns=['leaning_backwards_instructors'])
        df_instructors.to_csv(csv_path('leaning_backwards_instructors.csv'), index=True, index_label='student_id')
    
    df = pd.DataFrame.from_dict(not_working_dict, orient='index', columns=['leaning_backwards'])
    df.to_csv(csv_path('leaning_backwards.csv'), index=True, index_label='student_id')


def hands_on_the_side(wrists, shoulders, table_height) -> bool:
    left_wrist, right_wrist = wrists
    left_shoulder, right_shoulder = shoulders
    
    # See if the wrists are between the shoulders
    return not ((left_shoulder[0] < left_wrist[0] < right_shoulder[0]) \
           and (left_shoulder[0] < right_wrist[0] < right_shoulder[0])) or \
           (left_wrist[2] < table_height and right_wrist[2] < table_height)


def write_hands_on_the_side(weighed_by_time=True, weighed_by_work=True, weighed_by_surroundings=True):
    print("Writing hands on the side data...")
    not_working_dict = collections.defaultdict(int)
    not_working_dict_time = collections.defaultdict(int)
    nervousness_dicts_work = {'nothing': collections.defaultdict(int), 'work': collections.defaultdict(int)}
    for aoi in aois:
        nervousness_dicts_work[aoi] = collections.defaultdict(int)
    not_working_dict_peers = collections.defaultdict(int)
    not_working_dict_instructors = collections.defaultdict(int)
    
    center_positions = dict()

    for filename in dataset:
        data = dataset[filename]
        
        for _, row in data.iterrows():
            student = row['student_id']
            left_wrist, right_wrist = get_joint(row, 'left_wrist'), get_joint(row, 'right_wrist')
            left_shoulder, right_shoulder = get_joint(row, 'left_shoulder'), get_joint(row, 'right_shoulder')
            
            if hands_on_the_side((left_wrist, right_wrist), (left_shoulder, right_shoulder), 1):
                not_working_dict[student] += 1
                if weighed_by_time:
                    not_working_dict_time[student] += 1 / hours_til_thursday(row['timestamp'])
                    
                left_hip, right_hip = get_joint(row, 'left_hip'), get_joint(row, 'right_hip')
                center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
                if weighed_by_work:
                    workspace = in_workspace(center, 0.1)
                    nervousness_dicts_work[workspace][student] += 1
                    if workspace != 'nothing':
                        nervousness_dicts_work['work'][student] += 1
                if weighed_by_surroundings:
                    near = near_who(center, student, center_positions, 0.5)
                    if near == "TEACHERS":
                        not_working_dict_instructors[student] += 1
                    elif near == "STUDENTS":
                        not_working_dict_peers[student] += 1
                    
                
    if weighed_by_time:
        df_early = pd.DataFrame.from_dict(not_working_dict, orient='index', columns=['hands_on_the_side_time'])
        df_early.to_csv(csv_path('hands_on_the_side_time.csv'), index=True, index_label='student_id')
        
    if weighed_by_work:
        for key in nervousness_dicts_work:
            df_work = pd.DataFrame.from_dict(nervousness_dicts_work[key], orient='index', columns=[f'hands_on_the_side_{key}'])
            df_work.to_csv(csv_path(f'hands_on_the_side_{key}.csv'), index=True, index_label='student_id')
        
    if weighed_by_surroundings:
        df_peers = pd.DataFrame.from_dict(not_working_dict_peers, orient='index', columns=['hands_on_the_side_peers'])
        df_peers.to_csv(csv_path('hands_on_the_side_peers.csv'), index=True, index_label='student_id')
        
        df_instructors = pd.DataFrame.from_dict(not_working_dict_instructors, orient='index', columns=['hands_on_the_side_instructors'])
        df_instructors.to_csv(csv_path('hands_on_the_side_instructors.csv'), index=True, index_label='student_id')
    
    df = pd.DataFrame.from_dict(not_working_dict, orient='index', columns=['hands_on_the_side'])
    df.to_csv(csv_path('hands_on_the_side.csv'), index=True, index_label='student_id')
