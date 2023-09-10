import os
import argparse
import configparser
import pandas as pd
import math
import sys
sys.path.append('..')
from utils import *
from sample_v3 import Sample

if __name__ == "__main__":

    # read hyper-param settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='./prepare_4square_nyc.conf', type=str, help="configuration file path")
    parser.add_argument("--dataroot", default='../data/', type=str,
                    help="data root directory")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    print('Read configuration file: %s' % (args.config), flush=True)
    config.read(args.config)
    data_config = config['Data']
    data_root = args.dataroot
    # Data config
    city = data_config['dataset_name']
    country_name = data_config['country_name']
    x1 = float(data_config['x1'])
    x2 = float(data_config['x2'])
    y1 = float(data_config['y1'])
    y2 = float(data_config['y2'])
    max_his_period_days = int(data_config['max_his_period_days'])
    max_merge_seconds_limit = int(data_config['max_merge_seconds_limit'])
    max_delta_mins = int(data_config['max_delta_mins'])
    min_session_mins = int(data_config['min_session_mins'])
    least_disuser_count = int(data_config['least_disuser_count'])
    least_checkins_count = int(data_config['least_checkins_count'])
    latN = int(data_config['latN'])
    lngN = int(data_config['lngN'])
    gaussian_beta = int(data_config['gaussian_beta'])  
    distance_theta = int(data_config['distance_theta']) 
    split_save = bool(int(data_config['split_save']))
    if 'debug' in data_config.keys():
        debug = bool(int(data_config['debug']))
    else:
        debug = False
    
    # dataset
    # filepath = os.path.join(data_root, 'raw_datasets')
    if city == "GOW":
        filepath = os.path.join(data_root, 'raw_gowalla')
    elif city == "NYC" or city == "JKT":
        filepath = os.path.join(data_root, 'raw_4square')
    else:
        raise ValueError(f'Wrong city name: {city}')
    print('file path:', filepath, flush=True)
    if split_save:  #
        save_filename = 'www_' + city + '_' + str(max_his_period_days) + 'H' + str(max_merge_seconds_limit) + 'M' + str(max_delta_mins) + 'd' + str(min_session_mins) + 's' + str(least_disuser_count) + 'P' + str(least_checkins_count) + 'U'
        train_save_filename = os.path.join(data_root + 'new_datasets', 'train_' + save_filename + '.npz')
        val_save_filename = os.path.join(data_root + 'new_datasets', 'val_' + save_filename + '.npz')
        test_save_filename = os.path.join(data_root + 'new_datasets', 'test_' + save_filename + '.npz')
        print('save_filename:', flush=True)
        print(train_save_filename, flush=True)
        print(val_save_filename, flush=True)
        print(test_save_filename, flush=True)
    else:
        save_filename = 'www_' + city + '_' + str(max_his_period_days) + 'H' + str(max_merge_seconds_limit) + 'M' + str(max_delta_mins) + 'd' + str(min_session_mins) + 's' + str(least_disuser_count) + 'P' + str(least_checkins_count) + 'U'
        save_file = os.path.join(data_root + 'new_datasets', save_filename + '.npz')
        print('save_filename:', save_file, flush=True)

    POIs = pd.read_csv(os.path.join(filepath, 'raw_POIs.txt'), sep='\t', header=None)
    POIs.columns = ['venueId', 'latitude', 'longitude', 'category', 'Country']
    POIs_in_Country = POIs[POIs['Country'] == country_name]
    POIs_in_city = POIs_in_Country[(POIs_in_Country['latitude'] >= x1) & (POIs_in_Country['latitude'] <= x2) & (POIs_in_Country['longitude'] >= y1) & (POIs_in_Country['longitude'] <= y2)]
    POIs_in_city = POIs_in_city.drop_duplicates()
    POIs_in_city_set = set(POIs_in_city['venueId'].values)
    print('there are totally %d POIs in %s, %s' % (len(POIs_in_city_set), country_name, city))

    checkins = pd.read_csv(os.path.join(filepath, 'raw_Checkins.txt'), sep='\t', header=None)
    checkins.columns = ['userId', 'venueId', 'utctimestamp', 'Timezone offset in minutes']
    checkins = checkins[checkins['venueId'].isin(POIs_in_city_set)] 
    if city == "GOW":
        checkins['utcdatetime'] = checkins['utctimestamp'].apply(toDatetime_sy)
    elif city == "NYC" or city == "JKT":
        checkins['utcdatetime'] = checkins['utctimestamp'].apply(toDatetime)
    else:
        raise ValueError(f'Wrong city name: {city}')
    checkins = checkins[checkins['utcdatetime'].notnull()].reset_index(drop=True)
    print("checkins' length is " , len(checkins))
    if debug:
        checkins = checkins[:50000]

    # local time
    checkins['local time'] = checkins['utcdatetime'] + checkins['Timezone offset in minutes'].apply(toTimedelta)
    checkins['local weekday'] = checkins['local time'].apply(lambda x: x.weekday())
    checkins['local hour'] = checkins['local time'].apply(lambda x: x.hour)
    checkins['local minute'] = checkins['local time'].apply(lambda x: x.minute)

    # only several columns are useful
    checkins = checkins[['userId', 'venueId', 'local time', 'local weekday', 'local hour', 'local minute']]
    checkins = checkins.drop_duplicates()  

    # user_checkins_map = checkins[['userId', 'venueId']].drop_duplicates() 
    user_checkins_map = checkins[['userId', 'venueId']] 

    venueId_visitedUserCount = user_checkins_map.groupby(['venueId']).count().iloc[:, [0]]
    venueIds = venueId_visitedUserCount[venueId_visitedUserCount['userId'] >= least_disuser_count]
    venueIds = set(venueIds.index.values)  # get location set
    print('after filtering, %d locations are left.' % len(venueIds), flush=True)

    venue_info = POIs_in_city[POIs_in_city['venueId'].isin(venueIds)]
    venue_info = venue_info[['venueId', 'latitude', 'longitude', 'category']].drop_duplicates()
    category_idx = {}
    venueId_lidx = {}
    venue_category = {}
    venue_lat = {}
    venue_lng = {}
    venue_cnt = 0
    category_cnt = 0

    for vid, value in venue_info.groupby(['venueId']):
        print(vid, '-->', venue_cnt, flush=True)
        if(len(value) > 1):
            print(value, flush=True)
        venueId_lidx[vid] = venue_cnt

        if value.iloc[0]['category'] not in category_idx: 
            category_idx[value.iloc[0]['category']] = category_cnt
            venue_category[venue_cnt] = category_cnt
            category_cnt += 1
        else:
            venue_category[venue_cnt] = category_idx[value.iloc[0]['category']]
        venue_lat[venue_cnt] = value['latitude'].mean()
        venue_lng[venue_cnt] = value['longitude'].mean()
        venue_cnt += 1
    print('total category:', len(category_idx))

    SS_distance, SS_proximity, SS_gaussian_distance = construct_spatial_matrix_accordingDistance(distance_theta, venue_cnt, venue_lng, venue_lat, gaussian_beta=gaussian_beta)

    max_lat = max(venue_lat.values())
    min_lat = min(venue_lat.values())
    max_lng = max(venue_lng.values())
    min_lng = min(venue_lng.values())

    lats = []
    lngs = []
    for i in range(venue_cnt):
        lats.append(venue_lat[i])
        lngs.append(venue_lng[i])

    venue_latidx = {}
    venue_lngidx = {}
    for i in range(venue_cnt):
        eps = 1e-7
        latidx = int((venue_lat[i]-min_lat)*latN/(max_lat - min_lat + eps)) 
        lngidx = int((venue_lng[i]-min_lng)*lngN/(max_lng - min_lng + eps))
        venue_latidx[i]= latidx if latidx < latN else (latN-1)
        venue_lngidx[i]= lngidx if lngidx < lngN else (lngN-1)

    feature_category = []
    feature_lat = []
    feature_lng = []
    feature_lat_ori = []
    feature_lng_ori = []
    for i in range(venue_cnt):
        feature_category.append(venue_category[i])
        feature_lat.append(venue_latidx[i])
        feature_lng.append(venue_lngidx[i])
        feature_lat_ori.append(venue_lat[i])
        feature_lng_ori.append(venue_lng[i])


    checkins_invenueIds = checkins[checkins['venueId'].isin(venueIds)]
    user_CheckinCount = checkins_invenueIds.groupby(['userId']).count().iloc[:, [0]]
    userIds = user_CheckinCount[user_CheckinCount['venueId'] >= least_checkins_count]
    userIds = set(userIds.index.values)  
    print('after filtering, %d users are left.' % len(userIds), flush=True)

    checkins_filter = checkins[checkins['userId'].isin(userIds) & checkins['venueId'].isin(venueIds)]
    checkins_filter['timestamp'] = checkins_filter['local time'].apply(lambda x: x.timestamp()) 
    checkins_filter['lid'] = checkins_filter['venueId'].apply(lambda x: venueId_lidx[x]) 
    print('after filtering, %d check-ins points are left.' % len(checkins_filter), flush=True)
    print("checkins_filter's columns: ", checkins_filter.columns, checkins_filter.dtypes)

    sample_constructor = Sample(venueId_lidx, SS_distance=SS_distance, SS_gaussian_distance=SS_gaussian_distance, max_his_period_days=max_his_period_days, max_merge_seconds_limit=max_merge_seconds_limit, max_delta_mins=max_delta_mins, min_session_mins=min_session_mins)
    
    userId_checkins = checkins_filter.groupby('userId')

    all_drops = []
    all_drops_ratio = []
    for userId, checkins in userId_checkins:
        print('userId:', userId, flush=True)
        uid = sample_constructor.user_cnt
        print('uid:', uid, flush=True)
        checkins = checkins.sort_values(by=['timestamp'])  
        checkins = checkins.reset_index(drop=True)  
        
        tmp_len = len(checkins)
        checkins, drops = sample_constructor.deal_cluster_sequence_for_each_user(checkins) 
        all_drops.append(drops) 
        all_drops_ratio.append(drops / tmp_len) 

        total = len(checkins)
        user_lidFreq = (checkins[:].groupby(['lid']).count()).iloc[:, [0]]/total
        lid_visitFreq = venue_cnt * [0]
        for index, row in user_lidFreq.iterrows(): 
            lid_visitFreq[index] = row['userId']

        flag = sample_constructor.construct_sample_seq2seq(checkins, uid)
        if flag:
            sample_constructor.userId2uid[userId] = uid
            sample_constructor.user_cnt += 1
            sample_constructor.user_lidfreq.append(lid_visitFreq)
        else:
            print("drop the checkin data of ", userId)

    print("total drop events for each users: ", all_drops[:10], np.mean(all_drops), np.max(all_drops), np.min(all_drops), np.sum(all_drops))
    print("total drop events ratio for each users: ", all_drops_ratio[:10], np.mean(all_drops_ratio), np.max(all_drops_ratio), np.min(all_drops_ratio))

    print('total train sample cnt:', sample_constructor.trainX_sample_cnt, flush=True)
    print('total validation sample cnt:', sample_constructor.valX_sample_cnt, flush=True)
    print('total test sample cnt:', sample_constructor.testX_sample_cnt, flush=True)

    print('sample example check: Train sample', flush=True)
    print('trainX_target_lengths: ', sample_constructor.trainX_target_lengths[:5], flush=True)
    print('trainX_arrival_times:', sample_constructor.trainX_arrival_times[:5], flush=True)
    print('trainX_session_arrival_times:', sample_constructor.trainX_session_arrival_times[:5], flush=True)
    print('trainX_local_weekdays:', sample_constructor.trainX_local_weekdays[:5], flush=True)
    print('trainX_session_local_weekdays:', sample_constructor.trainX_session_local_weekdays[:5], flush=True)
    print('trainX_local_hours:', sample_constructor.trainX_local_hours[:5], flush=True)
    print('trainX_session_local_hours:', sample_constructor.trainX_session_local_hours[:5], flush=True)
    print('trainX_local_mins:', sample_constructor.trainX_local_mins[:5], flush=True)
    print('trainX_session_local_mins:', sample_constructor.trainX_session_local_mins[:5], flush=True)
    print('trainX_delta_times:', sample_constructor.trainX_delta_times[:5], flush=True)
    print('trainX_duration2first:', sample_constructor.trainX_duration2first[:5], flush=True)
    print('trainX_session_delta_times:', sample_constructor.trainX_session_delta_times[:5], flush=True)
    print('trainX_locations:', sample_constructor.trainX_locations[:5], flush=True)
    print('trainX_session_locations:', sample_constructor.trainX_session_locations[:5], flush=True)
    print('trainX_last_distances:', sample_constructor.trainX_last_distances[:5], flush=True)
    print('trainX_users:', sample_constructor.trainX_users[:5], flush=True)
    print('trainX_lengths:', sample_constructor.trainX_lengths[:5], flush=True)
    print('trainX_session_lengths:', sample_constructor.trainX_session_lengths[:5], flush=True)
    print('trainX_session_num:', sample_constructor.trainX_session_num[:5], flush=True)
    print('trainY_arrival_times:', sample_constructor.trainY_arrival_times[:5], flush=True)
    print('trainY_delta_times:', sample_constructor.trainY_delta_times[:5], flush=True)
    print('trainY_locations:', sample_constructor.trainY_locations[:5], flush=True)

    print('sample example check: Val sample', flush=True)
    print('valX_target_lengths: ', sample_constructor.valX_target_lengths[:5], flush=True)
    print('valX_arrival_times:', sample_constructor.valX_arrival_times[:5], flush=True)
    print('valX_session_arrival_times:', sample_constructor.valX_session_arrival_times[:5], flush=True)
    print('valX_local_weekdays:', sample_constructor.valX_local_weekdays[:5], flush=True)
    print('valX_session_local_weekdays:', sample_constructor.valX_session_local_weekdays[:5], flush=True)
    print('valX_local_hours:', sample_constructor.valX_local_hours[:5], flush=True)
    print('valX_session_local_hours:', sample_constructor.valX_session_local_hours[:5], flush=True)
    print('valX_local_mins:', sample_constructor.valX_local_mins[:5], flush=True)
    print('valX_session_local_mins:', sample_constructor.valX_session_local_mins[:5], flush=True)
    print('valX_delta_times:', sample_constructor.valX_delta_times[:5], flush=True)
    print('valX_duration2first:', sample_constructor.valX_duration2first[:5], flush=True)
    print('valX_session_delta_times:', sample_constructor.valX_session_delta_times[:5], flush=True)
    print('valX_locations:', sample_constructor.valX_locations[:5], flush=True)
    print('valX_session_locations:', sample_constructor.valX_session_locations[:5], flush=True)
    print('valX_last_distances:', sample_constructor.valX_last_distances[:5], flush=True)
    print('valX_users:', sample_constructor.valX_users[:5], flush=True)
    print('valX_lengths:', sample_constructor.valX_lengths[:5], flush=True)
    print('valX_session_lengths:', sample_constructor.valX_session_lengths[:5], flush=True)
    print('valX_session_num:', sample_constructor.valX_session_num[:5], flush=True)
    print('valY_arrival_times:', sample_constructor.valY_arrival_times[:5], flush=True)
    print('valY_delta_times:', sample_constructor.valY_delta_times[:5], flush=True)
    print('valY_locations:', sample_constructor.valY_locations[:5], flush=True)

    print('sample example check: Test sample', flush=True)
    print('testX_target_lengths: ', sample_constructor.testX_target_lengths[:5], flush=True)
    print('testX_arrival_times:', sample_constructor.testX_arrival_times[:5], flush=True)
    print('testX_session_arrival_times:', sample_constructor.testX_session_arrival_times[:5], flush=True)
    print('testX_local_weekdays:', sample_constructor.testX_local_weekdays[:5], flush=True)
    print('testX_session_local_weekdays:', sample_constructor.testX_session_local_weekdays[:5], flush=True)
    print('testX_local_hours:', sample_constructor.testX_local_hours[:5], flush=True)
    print('testX_session_local_hours:', sample_constructor.testX_session_local_hours[:5], flush=True)
    print('testX_local_mins:', sample_constructor.testX_local_mins[:5], flush=True)
    print('testX_session_local_mins:', sample_constructor.testX_session_local_mins[:5], flush=True)
    print('testX_delta_times:', sample_constructor.testX_delta_times[:5], flush=True)
    print('testX_duration2first:', sample_constructor.testX_duration2first[:5], flush=True)
    print('testX_session_delta_times:', sample_constructor.testX_session_delta_times[:5], flush=True)
    print('testX_locations:', sample_constructor.testX_locations[:5], flush=True)
    print('testX_session_locations:', sample_constructor.testX_session_locations[:5], flush=True)
    print('testX_last_distances:', sample_constructor.testX_last_distances[:5], flush=True)
    print('testX_users:', sample_constructor.testX_users[:5], flush=True)
    print('testX_lengths:', sample_constructor.testX_lengths[:5], flush=True)
    print('testX_session_lengths:', sample_constructor.testX_session_lengths[:5], flush=True)
    print('testX_session_num:', sample_constructor.testX_session_num[:5], flush=True)
    print('testY_arrival_times:', sample_constructor.testY_arrival_times[:5], flush=True)
    print('testY_delta_times:', sample_constructor.testY_delta_times[:5], flush=True)
    print('testY_locations:', sample_constructor.testY_locations[:5], flush=True)

    print('user_lidfreq:', sample_constructor.user_lidfreq[:5], flush=True)

    if split_save:
        print('save test data set ...')
        np.savez_compressed(test_save_filename,
                            testX_target_lengths=sample_constructor.testX_target_lengths,
                            testX_arrival_times=sample_constructor.testX_arrival_times,
                            testX_duration2first = sample_constructor.testX_duration2first,
                            testX_session_arrival_times=sample_constructor.testX_session_arrival_times,
                            testX_local_weekdays=sample_constructor.testX_local_weekdays,
                            testX_session_local_weekdays=sample_constructor.testX_session_local_weekdays,
                            testX_local_hours=sample_constructor.testX_local_hours,
                            testX_session_local_hours=sample_constructor.testX_session_local_hours,
                            testX_local_mins=sample_constructor.testX_local_mins,
                            testX_session_local_mins=sample_constructor.testX_session_local_mins,
                            testX_delta_times=sample_constructor.testX_delta_times,
                            testX_session_delta_times=sample_constructor.testX_session_delta_times,
                            testX_locations=sample_constructor.testX_locations,
                            testX_session_locations=sample_constructor.testX_session_locations,
                            testX_last_distances=sample_constructor.testX_last_distances,
                            testX_users=sample_constructor.testX_users, testX_lengths=sample_constructor.testX_lengths,
                            testX_session_lengths=sample_constructor.testX_session_lengths,
                            testX_session_num=sample_constructor.testX_session_num,
                            testY_arrival_times=sample_constructor.testY_arrival_times,
                            testY_delta_times=sample_constructor.testY_delta_times,
                            testY_locations=sample_constructor.testY_locations,

                            us=sample_constructor.us, vs=sample_constructor.vs,

                            feature_category=feature_category, feature_lat=feature_lat, feature_lng=feature_lng,
                            feature_lat_ori=feature_lat_ori, feature_lng_ori=feature_lng_ori,

                            latN=latN, lngN=lngN, category_cnt=category_cnt,

                            user_cnt=sample_constructor.user_cnt, venue_cnt=sample_constructor.venue_cnt,

                            SS_distance=sample_constructor.SS_distance, SS_guassian_distance=sample_constructor.SS_gaussian_distance
                            )

        print('save train data set ...')
        np.savez_compressed(train_save_filename,
                            trainX_target_lengths=sample_constructor.trainX_target_lengths,
                            trainX_arrival_times=sample_constructor.trainX_arrival_times,
                            trainX_duration2first=sample_constructor.trainX_duration2first,
                            trainX_session_arrival_times=sample_constructor.trainX_session_arrival_times,
                            trainX_local_weekdays=sample_constructor.trainX_local_weekdays,
                            trainX_session_local_weekdays=sample_constructor.trainX_session_local_weekdays,
                            trainX_local_hours=sample_constructor.trainX_local_hours,
                            trainX_session_local_hours=sample_constructor.trainX_session_local_hours,
                            trainX_local_mins=sample_constructor.trainX_local_mins,
                            trainX_session_local_mins=sample_constructor.trainX_session_local_mins,
                            trainX_delta_times=sample_constructor.trainX_delta_times,
                            trainX_session_delta_times=sample_constructor.trainX_session_delta_times,
                            trainX_locations=sample_constructor.trainX_locations,
                            trainX_session_locations=sample_constructor.trainX_session_locations,
                            trainX_last_distances=sample_constructor.trainX_last_distances,
                            trainX_users=sample_constructor.trainX_users, trainX_lengths=sample_constructor.trainX_lengths,
                            trainX_session_lengths=sample_constructor.trainX_session_lengths,
                            trainX_session_num=sample_constructor.trainX_session_num,
                            trainY_arrival_times=sample_constructor.trainY_arrival_times,
                            trainY_delta_times=sample_constructor.trainY_delta_times,
                            trainY_locations=sample_constructor.trainY_locations,
                            user_lidfreq=sample_constructor.user_lidfreq)

        print('save val data set ...')
        np.savez_compressed(val_save_filename,
                            valX_target_lengths=sample_constructor.valX_target_lengths,
                            valX_arrival_times=sample_constructor.valX_arrival_times,
                            valX_duration2first=sample_constructor.valX_duration2first,
                            valX_session_arrival_times=sample_constructor.valX_session_arrival_times,
                            valX_local_weekdays=sample_constructor.valX_local_weekdays,
                            valX_session_local_weekdays=sample_constructor.valX_session_local_weekdays,
                            valX_local_hours=sample_constructor.valX_local_hours,
                            valX_session_local_hours=sample_constructor.valX_session_local_hours,
                            valX_local_mins=sample_constructor.valX_local_mins,
                            valX_session_local_mins=sample_constructor.valX_session_local_mins,
                            valX_delta_times=sample_constructor.valX_delta_times,
                            valX_session_delta_times=sample_constructor.valX_session_delta_times,
                            valX_locations=sample_constructor.valX_locations,
                            valX_session_locations=sample_constructor.valX_session_locations,
                            valX_last_distances=sample_constructor.valX_last_distances,
                            valX_users=sample_constructor.valX_users, valX_lengths=sample_constructor.valX_lengths,
                            valX_session_lengths=sample_constructor.valX_session_lengths,
                            valX_session_num=sample_constructor.valX_session_num,
                            valY_arrival_times=sample_constructor.valY_arrival_times,
                            valY_delta_times=sample_constructor.valY_delta_times,
                            valY_locations=sample_constructor.valY_locations)
    else:
        np.savez_compressed(save_file, user_cnt=sample_constructor.user_cnt, venue_cnt=sample_constructor.venue_cnt,

                            trainX_arrival_times=sample_constructor.trainX_arrival_times, trainX_duration2first=sample_constructor.trainX_duration2first, trainX_session_arrival_times=sample_constructor.trainX_session_arrival_times,
                            trainX_local_weekdays=sample_constructor.trainX_local_weekdays,trainX_session_local_weekdays=sample_constructor.trainX_session_local_weekdays,
                            trainX_local_hours=sample_constructor.trainX_local_hours, trainX_session_local_hours=sample_constructor.trainX_session_local_hours,
                            trainX_local_mins=sample_constructor.trainX_local_mins, trainX_session_local_mins=sample_constructor.trainX_session_local_mins,
                            trainX_delta_times=sample_constructor.trainX_delta_times, trainX_session_delta_times=sample_constructor.trainX_session_delta_times,
                            trainX_locations=sample_constructor.trainX_locations, trainX_session_locations=sample_constructor.trainX_session_locations,
                            trainX_last_distances=sample_constructor.trainX_last_distances,
                            trainX_users=sample_constructor.trainX_users,
                            trainX_lengths=sample_constructor.trainX_lengths,
                            trainX_session_lengths=sample_constructor.trainX_session_lengths,
                            trainX_session_num=sample_constructor.trainX_session_num,
                            trainY_arrival_times=sample_constructor.trainY_arrival_times,
                            trainY_delta_times=sample_constructor.trainY_delta_times,
                            trainY_locations=sample_constructor.trainY_locations,
                            trainX_target_lengths = sample_constructor.trainX_target_lengths,
                            
                            user_lidfreq=sample_constructor.user_lidfreq,

                            valX_arrival_times=sample_constructor.valX_arrival_times,
                            valX_duration2first=sample_constructor.valX_duration2first,
                            valX_session_arrival_times=sample_constructor.valX_session_arrival_times,
                            valX_local_weekdays=sample_constructor.valX_local_weekdays,
                            valX_session_local_weekdays=sample_constructor.valX_session_local_weekdays,
                            valX_local_hours=sample_constructor.valX_local_hours,
                            valX_session_local_hours=sample_constructor.valX_session_local_hours,
                            valX_local_mins=sample_constructor.valX_local_mins,
                            valX_session_local_mins=sample_constructor.valX_session_local_mins,
                            valX_delta_times=sample_constructor.valX_delta_times,
                            valX_session_delta_times=sample_constructor.valX_session_delta_times,
                            valX_locations=sample_constructor.valX_locations,
                            valX_session_locations=sample_constructor.valX_session_locations,
                            valX_last_distances=sample_constructor.valX_last_distances,
                            valX_users=sample_constructor.valX_users,
                            valX_lengths=sample_constructor.valX_lengths,
                            valX_session_lengths=sample_constructor.valX_session_lengths,
                            valX_session_num=sample_constructor.valX_session_num,
                            valY_arrival_times=sample_constructor.valY_arrival_times,
                            valY_delta_times=sample_constructor.valY_delta_times,
                            valY_locations=sample_constructor.valY_locations,
                            valX_target_lengths = sample_constructor.valX_target_lengths,

                            testX_arrival_times=sample_constructor.testX_arrival_times,
                            testX_duration2first=sample_constructor.testX_duration2first,
                            testX_session_arrival_times=sample_constructor.testX_session_arrival_times,
                            testX_local_weekdays=sample_constructor.testX_local_weekdays,
                            testX_session_local_weekdays=sample_constructor.testX_session_local_weekdays,
                            testX_local_hours=sample_constructor.testX_local_hours,
                            testX_session_local_hours=sample_constructor.testX_session_local_hours,
                            testX_local_mins=sample_constructor.testX_local_mins,
                            testX_session_local_mins=sample_constructor.testX_session_local_mins,
                            testX_delta_times=sample_constructor.testX_delta_times,
                            testX_session_delta_times=sample_constructor.testX_session_delta_times,
                            testX_locations=sample_constructor.testX_locations,
                            testX_session_locations=sample_constructor.testX_session_locations,
                            testX_last_distances=sample_constructor.testX_last_distances,
                            testX_users=sample_constructor.testX_users,
                            testX_lengths=sample_constructor.testX_lengths,
                            testX_session_lengths=sample_constructor.testX_session_lengths,
                            testX_session_num=sample_constructor.testX_session_num,
                            testY_arrival_times=sample_constructor.testY_arrival_times,
                            testY_delta_times=sample_constructor.testY_delta_times,
                            testY_locations=sample_constructor.testY_locations,
                            testX_target_lengths = sample_constructor.testX_target_lengths,
            
                            us=sample_constructor.us, vs=sample_constructor.vs,

                            feature_category=feature_category, feature_lat=feature_lat, feature_lng=feature_lng, feature_lat_ori=feature_lat_ori, feature_lng_ori=feature_lng_ori,

                            latN=latN, lngN=lngN, category_cnt=category_cnt, SS_distance=sample_constructor.SS_distance, SS_guassian_distance=sample_constructor.SS_gaussian_distance)


