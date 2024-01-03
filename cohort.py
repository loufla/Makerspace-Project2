import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class Cohort:
    def __init__(self, folder):
        ''' creates class variables '''
        self.sections = []
        self.office_hours = []
        self.name = os.path.basename(folder)
        self.people = self.create_people(folder)
        self.groups = self.create_groups(folder)
        self.calendar = self.create_calendar(folder)
        
    def create_people(self, folder):
        ''' creates a Person instance for each participant '''
        dic_participants = {}
        csv_participants = os.path.join(folder, 'participants', 'participants.csv')
        # if the file doesn't exist, we throw an exception
        if not os.path.isfile(csv_participants):
            raise Exception("Couldn't find ", csv_participants)
        df_participants = pd.read_csv(csv_participants)
        for index, row in df_participants.iterrows():
            fname,lname = row['first_name'],row['last_name']
            dic_participants[fname.lower()] = Person(fname,lname,row['email'],row['role'])
        return dic_participants
    
    def create_groups(self, folder):
        ''' retrives the groups that worked together on different projects '''
        csv_groups = os.path.join(folder, 'participants', 'groups_by_week.csv')
        # if the file doesn't exist, we throw an exception
        if not os.path.isfile(csv_groups):
            raise Exception("Couldn't find ", csv_groups)
        return pd.read_csv(csv_groups)
    
    def create_calendar(self, folder):
        ''' retrieves the calendar for the semester '''
        csv_calendar = os.path.join(folder, 'calendar', 'cleaned_calendar.csv')
        # if the file doesn't exist, we throw an exception
        if not os.path.isfile(csv_calendar):
            raise Exception("Couldn't find ", csv_calendar)
        # turn office hours and sections into a list for easier access
        sections_df = pd.read_csv(csv_calendar)
        sections = []
        for index, row in sections_df.iterrows():
            start,end = row['dtstart'][:-6],row['dtend'][:-6]
            start = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
            end = datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
            # FIX TIMEZONE ISSUES
            add_hours = 0
            if '+00:00' in row['dtstart']: 
                if end < datetime(2022, 4, 13): add_hours = -5
                elif start > datetime(2022, 4, 13): add_hours = -4
            start = start + timedelta(hours=add_hours)
            end = end + timedelta(hours=add_hours)
            # fill data in the dataframe
            if 'section' in row['summary'].lower():
                self.sections.append([start,end])
            if 'office' in row['summary'].lower():
                self.office_hours.append([start,end])
        
    def same_group(self, student1, student2, date):
        ''' determines if two students were working on the same project '''
        # get and convert the data
        groups_df = self.groups
        groups_df['from'] = pd.to_datetime(groups_df['from'], format='%m/%d/%y')
        groups_df['to'] = pd.to_datetime(groups_df['to'], format='%m/%d/%y')
        # select subset based on date and check if students are there
        subdf = groups_df.loc[(date >= groups_df['from']) & (date < groups_df['to'])]
        if student1 in subdf.student1.values and student2 in subdf.student2.values: return True
        if student2 in subdf.student1.values and student1 in subdf.student2.values: return True
        return False  
    
    def is_section(self, time):
        ''' check if a time is a section '''
        for start,end in self.sections:
            if time >= start and time <= end:
                return 1
        return ''
    
    def is_office(self, time):
        ''' check if a time is an office hours '''
        for start,end in self.office_hours:
            if time >= start and time <= end:
                return 1
        return ''
            
        
class Person:
    def __init__(self, fname, lname, email, role):
        self.fname = fname
        self.lname = lname
        self.email = email
        self.role = role