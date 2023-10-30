# libraries
import os
import sys
import glob
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tqdm import tqdm
from math import acos, degrees
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont


####################
# Global variables #
####################

# size of the floorplan / video
VIDEO_W,VIDEO_H = (1154,732)

# AOIs of the space
activities = ['laser','soldering','sewing','printer','collaboration','tool','office','nothing']

# define AOIs
laser_position=[(7.5, 8.5),(7.5, 3.5),(6, 3.5),(6, 8.5)] 
soldering_position=[(7.5, 11.5),(7.5, 8.5),(6, 8.5),(6, 11.5)]
sewing_position=[(6.5, 2),(6.5, 0.2),(5.5, 0.2),(5.5, 2)] 
printer_position=[(3, 1.8),(3, 0.2),(0.05, 0.2),(0.05, 1.8)] 
collaboration_position=[(5.5, 10.5),(5.5, 2),(1.7, 2),(1.7, 10.5)] 
tool_position=[(1.4, 8),(1.4, 4.3),(0.2, 4.3),(0.2, 8)] 
office_position=[(1.4, 11.5),(1.4, 8),(0.2, 8),(0.2, 11.5)]


###############
# Math helper #
###############

def intersect(line1, line2):
    p1,p2 = line1
    p3,p4 = line2
    x1,y1 = p1
    x2,y2 = p2
    x3,y3 = p3
    x4,y4 = p4
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == 0: return None # parallel
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua < 0 or ua > 1: return None # out of range
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub < 0 or ub > 1: return None # out of range
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)
    return (x,y)

def dist_two_points(pt1,pt2):
    return math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)

def dist_two_points_3d(pt1,pt2):
    return math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2 + (pt2[2] - pt1[2])**2)

def compute_gaze(head,nose,scalar=2):
    vector = (nose[0]-head[0])*2, (nose[1]-head[1])*2
    gaze = vector[0]*scalar, vector[1]*scalar
    gaze = [head,(gaze[0]+head[0],gaze[1]+head[1])]
    return gaze


########
# AOIs #
########

def within_area(test_position,target_area):

    # transform coordinates
    x, y = transform_coordinates(test_position[0],test_position[1])

    transformed_area = []
    for target_position in target_area:
        transformed_area.append(transform_coordinates(target_position[0],target_position[1]))

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
        if activity == 'nothing': pass
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
    

def add_aoi_to_df(df):
    df['aoi'] = df.apply(lambda x: determine_activity((x['nose_x'],x['nose_y'])), axis=1)

    
#########
# Poses #
#########

def dist(a,b):
    ''' compute the distance betweeen two 3D points '''
    p = np.array((float(a[0]),float(a[1]),float(a[2])))
    q = np.array((float(b[0]),float(b[1]),float(b[2])))
    return np.linalg.norm(p-q)

def angle(p1, p2, p3):
    ''' compute the angle between three points '''
    A = dist(p1, p2)
    B = dist(p1, p3)
    C = dist(p2, p3)
    return degrees(acos((A * A + C * C - B * B)/(2.0 * A * C)))

def angle_df(df, j1, j2, j3):
    ''' add angle data to the entire dataframe ''' 
    df[j2+'_angle'] = df.apply(lambda x: angle((x[j1+'_x'],x[j1+'_y'],x[j1+'_z']),\
                                               (x[j2+'_x'],x[j2+'_y'],x[j2+'_z']),\
                                               (x[j3+'_x'],x[j3+'_y'],x[j3+'_z'])), axis=1)

def add_neck_angle(df): 
    ''' add neck and sacrum coordinates; compute neck angle '''

    # add neck
    df['neck_x'] = df[['right_shoulder_x', 'left_shoulder_x']].mean(axis=1)
    df['neck_y'] = df[['right_shoulder_y', 'left_shoulder_y']].mean(axis=1)
    df['neck_z'] = df[['right_shoulder_z', 'left_shoulder_z']].mean(axis=1)
    
    # add sacrum
    df['sacrum_x'] = df[['right_hip_x', 'left_hip_x']].mean(axis=1)
    df['sacrum_y'] = df[['right_hip_y', 'left_hip_y']].mean(axis=1)
    df['sacrum_z'] = df[['right_hip_z', 'left_hip_z']].mean(axis=1)
    
    # compute angles
    angle_df(df, 'nose', 'neck', 'sacrum')

def compute_joints_angle(df):
    ''' add columns to compute the angle between different joints to a df '''
    
    add_neck_angle(df)
    angle_df(df, 'left_shoulder', 'right_shoulder', 'right_elbow')
    angle_df(df, 'right_shoulder', 'left_shoulder', 'left_elbow')
    angle_df(df, 'left_shoulder', 'left_elbow', 'left_wrist')
    angle_df(df, 'right_shoulder', 'right_elbow', 'right_wrist')
    angle_df(df, 'neck', 'left_hip', 'left_knee')
    angle_df(df, 'neck', 'right_hip', 'right_knee')
    angle_df(df, 'left_hip', 'left_knee', 'left_ankle')
    angle_df(df, 'right_hip', 'right_knee', 'right_ankle')

def generate_angle_file(filename): 
    if not filename.endswith('.csv'): return
    path = os.path.join(downsampled_path, filename)
    output = os.path.join('./angle_data/', filename.replace('.csv','_angle.csv'))
    if not os.path.isfile(output):
        try: 
            df = pd.read_csv(path)
            compute_joints_angle(df)
            df_angles = df[['person_identity','datetime'] + [x for x in df.columns if 'angle' in x]]
            df_angles.to_csv(output, index=False)
        except Exception as e: 
            return 'failed because ' + str(e)


def add_standing(df):
    ''' tells you if a person is standing (=1)'''

    # compute knee angles
    if 'left_knee_angle' not in df.columns: 
        angle_df(df, 'left_hip', 'left_knee', 'left_ankle')
        angle_df(df, 'right_hip', 'right_knee', 'right_ankle')

    # a person is standing if they are at a certain height and
    # their knee angles are larger than 125 degrees
    def is_standing(row):
        if row.right_knee_angle > 125 and row.left_knee_angle > 125 and row.nose_z > 1.1:
            return 1

    df['standing'] = df.apply(lambda x: is_standing(x), axis=1)


########
# Draw #
########

def create_new_image(floorplan_path):
    ''' create a new image with the floorplan '''
    lab = Image.open(floorplan_path)
    img = Image.new('RGB', (VIDEO_W,VIDEO_H), (255, 255, 255))
    img.paste(lab, (0,0), mask=lab)
    return img, ImageDraw.Draw(img, 'RGBA')


def transform_coordinates(x,y):
    ''' scales and moves the coordinates to be aligned on the floorplan '''
    if np.isnan(x) or math.isnan(x): return np.nan,np.nan
    new_y = ((x * 90) - VIDEO_W) * -1 - 450
    new_x = ((y * 90) - VIDEO_H) * -1 + 410
    return int(new_x),int(new_y)


def draw_ellipse(draw, joint, r, color, scale=1):
    ''' PIL helper to draw an ellipse for a given joint'''
    x,y = joint
    x = int(abs(x))
    y = int(abs(y))
    joint_ellipse = (x-r,y-r,x+r,y+r)
    draw.ellipse(joint_ellipse, fill=color, outline='black')


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

    coordinates = []

    for equipment_position in equipment_positions:

        # define variables
        transformed_position = transform_coordinates(*equipment_position)

        # draw the equipment position
        r = 4
        equipment_ellipse = (transformed_position[0]-r,transformed_position[1]-r,transformed_position[0]+r,transformed_position[1]+r)
        draw.ellipse(equipment_ellipse, fill=(255,0,0), outline='black')

        # label the equipment position
        font = ImageFont.load_default()

        coordinates += [transformed_position]

    draw.rectangle([coordinates[1], coordinates[3]], outline='black', fill=(255,255,255,100))
    center = (coordinates[1][0]+coordinates[3][0])/2, (coordinates[1][1]+coordinates[3][1])/2
    draw.text(center, equipment_name , font=font, fill='black')