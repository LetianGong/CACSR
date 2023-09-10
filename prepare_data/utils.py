import numpy as np
import copy
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
from datetime import timedelta

def toTimedelta(x):
    return timedelta(minutes=x)

def toDatetime(x):
    try:
        return datetime.strptime(x.replace('+0000 ', ''), "%a %b %d %H:%M:%S %Y")
    except:
        return None

def toDatetime_sy(x):
    try:
        return datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    except:
        return None

def distance(lon1, lat1, lon2, lat2):  
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
   
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    if lon1 == 0 or lat1 ==0 or lon2==0 or lat2==0:
        return 0
    # haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  
    return c * r  


def construct_spatial_matrix_accordingDistance(distance_theta, venue_cnt, venue_lng, venue_lat, gaussian_beta=None):
    SS_distance = np.zeros((venue_cnt, venue_cnt))  
    SS_gaussian_distance = np.zeros((venue_cnt, venue_cnt))  
    SS_proximity = np.zeros((venue_cnt, venue_cnt))  
    for i in range(venue_cnt):
        for j in range(venue_cnt):
            distance_score = distance(venue_lng[i], venue_lat[i], venue_lng[j], venue_lat[j])
            SS_distance[i, j] = distance_score  
            if gaussian_beta is not None:
                distance_gaussian_score = np.exp(-gaussian_beta * distance_score) 
                SS_gaussian_distance[i, j] = distance_gaussian_score  
            if SS_distance[i, j] < distance_theta:  
                SS_proximity[i, j] = 1
    return SS_distance, SS_proximity, SS_gaussian_distance


def delta_minutes(ori, cp):
    delta = (ori.timestamp() - cp.timestamp())/60
    if delta < 0:
        delta = 1
    return delta


def get_relativeTime(arrival_times): 
    first_time_list = [arrival_times[0] for _ in range(len(arrival_times))]
    return list(map(delta_minutes, arrival_times, first_time_list))


def get_delta(arrival_times):

    copy_times = copy.deepcopy(arrival_times)
    copy_times.insert(0, copy_times[0]) 
    copy_times.pop(-1)
    return list(map(delta_minutes, arrival_times, copy_times))


def split_sampleSeq2sessions(sampleSeq_delta_times, min_session_mins):

    sessions = []  
    split_index = []  #
    sessions_lengths = []

    for i in range(1, len(sampleSeq_delta_times)):
        if sampleSeq_delta_times[i] >= min_session_mins:
            split_index.append(i)
    # print('split_index:', split_index)
    if len(split_index) == 0:  
        sessions.append(sampleSeq_delta_times)
        sessions_lengths.append(len(sampleSeq_delta_times))
        # print('sessions:', sessions)
        # print('sessions_lengths:', sessions_lengths)
        return sessions, split_index, sessions_lengths
    else:
        start_index = 0
        for i in range(0, len(split_index)):
            split = split_index[i]
            if split-start_index > 1:  
                sampleSeq_delta_times[start_index] = 0  
                sessions.append(sampleSeq_delta_times[start_index:split])
                sessions_lengths.append(len(sampleSeq_delta_times[start_index:split]))
            start_index = split
        if len(sampleSeq_delta_times[split_index[-1]:]) > 1: 
            sampleSeq_delta_times[split_index[-1]] = 0
            sessions.append(sampleSeq_delta_times[split_index[-1]:])
            sessions_lengths.append(len(sampleSeq_delta_times[split_index[-1]:]))
        # print('sessions:', sessions)
        # print('sessions_lengths:', sessions_lengths)
        return sessions, split_index, sessions_lengths  



def splitSeq_basedonSessions(seq, split_index):

    sessions = []
    if len(split_index) == 0:
        sessions.append(seq)
    else:
        start_index = 0
        for i in range(0, len(split_index)):
            split = split_index[i]
            if split-start_index > 1:
                sessions.append(seq[start_index:split])
            start_index = split
        if len(seq[split_index[-1]:]) > 1: 
            sessions.append(seq[split_index[-1]:])
    return sessions


def getTimeDelta_forSTRNN(session_based_sample_arrival_times):
    '''
    :param session_based_sample_arrival_times:  in mins
    :return:
    '''
    time_delta = []
    for i in range(len(session_based_sample_arrival_times)):  
        time_delta.append([(session_based_sample_arrival_times[i][-1].timestamp() - x.timestamp()) / 60 for x in
                        session_based_sample_arrival_times[i]])
    return time_delta


def getDistanceDelta_forSTRNN(SS_distance, session_based_sample_locations):
    distance_delta = []
    for i in range(len(session_based_sample_locations)):
        distance_delta.append(
            [SS_distance[session_based_sample_locations[i][-1], x] for x in session_based_sample_locations[i]])
    return distance_delta

