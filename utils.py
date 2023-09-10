import numpy as np
import os
import math
import nni
#import seaborn as sns
#import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def evaluate_location(y, top_k_pred, K=20):
    '''
    get hit ratio, mrr
    :param y: (batch,)
    :param top_k_pred: (batch, num_class)
    :param K
    :return:
    '''
    total_num = top_k_pred.shape[0]
    hit_ratio = np.zeros(K)
    mrr = []
    for i in range(total_num):  
        rank = np.where(top_k_pred[i] == y[i])[0] + 1
        mrr.append(rank)
        for j in range(1, K+1):   
            if y[i] in set(top_k_pred[i, :j]):
                hit_ratio[j-1] = hit_ratio[j-1] + 1
    hit_ratio = hit_ratio/total_num
    # print('mrr:',mrr)
    mrr = (1/np.array(mrr)).mean()
    return hit_ratio, mrr


def get_total_prob_c(loader, model, gts, gss, time_info, week_info, feature_category, feature_lat, feature_lng, feature_lat_ori, feature_lng_ori, save_filename=None, params_path=None, distance=None):
    '''
    calculates the loss, mae and mape for the entire data loader
    :param loader:
    :param save:
    :return:
    '''
    all_topic = []
    all_label = []
    for input in loader:
        topic, gamma_c = model.get_gammac(input, gss, feature_category, feature_lat, feature_lng, feature_lat_ori, feature_lng_ori, gts, time_info, week_info, distance=distance)  # (batch_size,), (batch_size,)
        all_topic.append(topic.detach().cpu().numpy())
        all_label.append(gamma_c.detach().cpu().numpy())
    all_topic = np.concatenate(all_topic)
    all_label = np.concatenate(all_label)
    all_label_index = np.argmax(all_label, axis=1)
    print('all_label:', all_label[:10])
    print('all_label_index:', all_label_index[:10])
    if save_filename is not None:
        filename = os.path.join(params_path, save_filename + '_gammac.npz')
        np.savez(filename, all_topic=all_topic, all_label=all_label, all_label_index=all_label_index)
    return all_topic, all_label, all_label_index


def density_visualization(density, ground_truth, batch_cnt):
    '''
    plot the probability density function
    :param density:
    :param ground_truth:
    :return:
    '''
    n_samples = 1000
    hours = 48
    x = np.linspace(0, hours, n_samples)
    t = ground_truth
    cnt = 0
    length = len(density)
    '''
    for i in range(length):
        y = density[i]
        plt.plot(x, y, "r-", label="STDGN")
        plt.legend()
        plt.xlabel(r"$\tau$", fontdict={'family': 'Times New Roman', 'size':16})
        plt.ylabel(r"p($\tau$)", fontdict={'family': 'Times New Roman', 'size':16})
        plt.yticks(fontproperties = 'Times New Roman', size = 14)
        plt.xticks(fontproperties = 'Times New Roman', size = 14)
        plt.grid()
        # plt.title("the probability density function in JKT dataset", fontdict={'family': 'Times New Roman', 'size':16})
        true_value = round(t[i], 2)
        plt.axvline(x=true_value, ls=":", c="black")
        plt.text(x=true_value + 1, y=1/2*np.max(y), s=r"$\tau_{n+1}$=" + str(true_value), size=16, alpha=0.8)
        plt.legend(prop={'family' : 'Times New Roman', 'size' : 16})
        plt.show()
        pic_name = str(batch_cnt) + '_' + str(cnt)
        cnt += 1
        plt.savefig(f'./data/density/jkt_{pic_name}.png')
        plt.savefig(f'./data/density/jkt_{pic_name}.eps',format='eps', dpi=10000)
        plt.close()
    '''


def softmax(x):
    '''
    self-define softmax operation
    :param x:
    :return:
    '''
    # print("before: ", x)
    x -= np.max(x, axis=1, keepdims=True)  # for stationary computation
    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)  # formula
    # print("after: ", x)
    return x


def rad(d):
    '''
    rad the latitude and longitude
    :param d: latitude or longitude
    :return rad:
    '''
    return d * math.pi / 180.0


def getDistance(lat1, lng1, lat2, lng2):
    '''
    get the distance between two location using their latitude and longitude
    :param lat1:
    :param lng1:
    :param lat2:
    :param lng2:
    :return s:
    '''
    EARTH_REDIUS = 6378.137
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(math.sqrt(math.pow(math.sin(a/2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b/2), 2)))
    s = s * EARTH_REDIUS
    return s


def attention_visualization(batch_attention, events, batch_cnt, batch_size, layers, heads, hierarchical, low_layer, high_layer, dataset_name, category, batch_lat, batch_lng, batch_week, batch_hour, batch_minute):
    '''
    attention visualization
    :param batch_attention:
    :return:
    '''
    '''
    def draw(data, x, y, ax, bar, bar_min, bar_max):
        sns.heatmap(data,
                xticklabels=x, square=True, yticklabels=y, vmin=bar_min-0.01, vmax=bar_max+0.01,
                cbar=bar, ax=ax, cmap='Blues')
    '''
    # color = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgreen', 'lightgray', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen']
    color = ['black', 'blue', 'lightcoral', 'red', 'brown', 'gold', 'darkorange', 'lime', 'yellow', 'magenta', 'olive', 'purple', 'firebrick', 'indigo', 'deeppink', 'darkgreen', 'cyan', 'slategray', 'darkkhaki', 'peru']  # strong color
    cnt = 0

    for sample in range(batch_size):
        # the upper and lower bound of the longitude
        x_num_list = batch_lng[sample]
        max_lng = round((round(max(x_num_list),4)+0.01),4)
        min_lng = round((round(min(x_num_list),4)-0.01),4)
        range_lng = round(max_lng-min_lng,2)*100
        if abs(range_lng) < 0.001:
            range_lng = 5
        # print("range_lng:" , range_lng)

        # the upper and lower bound of latitude
        y_num_list = batch_lat[sample]
        max_lat = round((round(max(y_num_list),4)+0.01),4)
        min_lat = round((round(min(y_num_list),4)-0.01),4)
        range_lat = round(max_lat-min_lat,2)*100
        if abs(range_lat) < 0.001:
            range_lat = 5
        # print("range_lat: ", range_lat)
        '''
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus']=False
        fig = plt.figure(figsize=(range_lng,range_lat))
        plt.xlim(min_lng, max_lng)
        plt.ylim(min_lat, max_lat)
        '''
        # simply process the venue which too close
        locations = dict()
        visit = dict()
        for text, lng, lat in zip(category[sample], batch_lng[sample], batch_lat[sample]):
            if text not in locations.keys():
                locations[text] = [1, [lng, lat]]
            else:
                locations[text] = [locations[text][0] + 1, [lng, lat]]  # repeated venue
            visit[text] = 0
        for key1 in locations.keys():
            if visit[key1] == 0:
                lng = locations[key1][1][0]
                lat = locations[key1][1][1]
                for key2 in locations.keys():
                    if key1 != key2 and visit[key2] == 0:
                        tmp_lng = locations[key2][1][0]
                        tmp_lat = locations[key2][1][1]
                        tmp_cnt = locations[key2][0]
                        # print(lat, lng, tmp_lat, tmp_lng, getDistance(lat, lng, tmp_lat, tmp_lng))
                        if getDistance(lat, lng, tmp_lat, tmp_lng) < 0.2 or abs(lat - tmp_lat) < 0.0015:  # too close or same latitude（km）
                            if lat > tmp_lat:  # move down
                                locations[key2] = [tmp_cnt, [tmp_lng, tmp_lat - 0.002]]
                            else:  # move up
                                locations[key2] = [tmp_cnt, [tmp_lng, tmp_lat + 0.002]]
                visit[key1] = 1

        category_process = []
        batch_lng_process = []
        batch_lat_process = []
        for key in locations.keys():  # remove the repetitive key
            category_process.append(key)
            batch_lng_process.append(locations[key][1][0])
            batch_lat_process.append(locations[key][1][1])

        pic_name = str(batch_cnt) + '_' + str(sample)

        color_cnt = 0
        '''
        for text, lng, lat in zip(category_process, batch_lng_process, batch_lat_process):
            # print(f"{text}: ({lng}, {lat})")
            plt.plot(lng, lat, 'o', color=color[color_cnt])  # different venue use different color
            plt.annotate(text, (lng, lat))
            color_cnt = color_cnt + 1
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # set the edge distance
        plt.savefig(f'./data/attention/jkt_{pic_name}_map.png')
        plt.savefig(f'./data/attention/jkt_{pic_name}_map.eps', format="eps", dpi=1000)
        plt.close()
        '''
        x = []
        y = []
        dict_weeks = {0: 'Sun', 1: 'Mon',  2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat'}
        for tmp_category, week, hour, minute in zip(category[sample], batch_week[sample], batch_hour[sample], batch_minute[sample]):
            time = ''
            reverse_time = ''
            time = time + dict_weeks[week] + " "
            reverse_time = reverse_time + tmp_category + " "
            if hour < 10:
                time = time + "0" + str(hour) + ":"
                reverse_time += "0" + str(hour) + ":"
            else:
                time = time + str(hour) + ":"
                reverse_time += str(hour) + ":"
            if minute < 10:
                time = time + "0" + str(minute)
                reverse_time += "0" + str(minute)
            else:
                time = time + str(minute)
                reverse_time += str(minute)
            time = time + " "  + tmp_category
            reverse_time += " " + dict_weeks[week]
            y.append(time)
            x.append(reverse_time)
            #fig, axs = plt.subplots(layers, heads, figsize=(20, 32))
            if hierarchical:
                low_and_high_layers = [low_layer, high_layer]
                for i in range(len(batch_attention)):
                    for layer in range(low_and_high_layers[i]):
                        for head in range(heads):
                            if i == 0:
                                ax = axs[layer][head]
                            else:
                                ax = axs[layer + low_and_high_layers[0]][head]
                            tmp = softmax(batch_attention[i][sample, layer, head, :events, :events])
                            draw(tmp,
                                x if (i == 1 and layer == low_and_high_layers[i] - 1) else [], y if head == 0 else [], ax=ax, bar=False if head == heads-1 else False, bar_min=np.min(tmp), bar_max=np.max(tmp))
                            ax.tick_params(labelsize=20)

            else:
                for layer in range(layers):
                    for head in range(heads):
                        if layers == 1:
                            ax = axs[head]  # when layers equal to one, there will be a error that *dimension lost*
                        else:
                            ax = axs[layer][head]
                        tmp = softmax(batch_attention[sample, layer, head, :events, :events])
                        draw(tmp,
                            x if (layer == layers - 1) else [], y if head == 0 else [], ax=ax, bar=False, bar_min=np.min(tmp), bar_max=np.max(tmp))
            fig.tight_layout()  # tight layout
            #plt.savefig(f"./data/attention/jkt_{pic_name}_score.png")
            #plt.savefig(f"./data/attention/jkt_{pic_name}_score.eps", format="eps", dpi=1000)
            cnt = cnt + 1
            #plt.close()


def get_s_baselines_total_loss_s_for_CACSR_RNN(loader, model, save_filename=None, params_path=None):
    all_loss_s = []
    all_ground_truth_location = []
    all_predicted_topK = []

    for input in loader:
        s_loss_score, top_k_pred = model(input)
        all_loss_s.append(s_loss_score.detach().cpu().numpy())
        all_ground_truth_location.append(input.Y_location.cpu().numpy())
        all_predicted_topK.append(top_k_pred.cpu().numpy())

    all_loss_s = np.array(all_loss_s)
    all_loss_s = np.mean(all_loss_s)

    all_ground_truth_location = np.concatenate(all_ground_truth_location)
    all_predicted_topK = np.concatenate(all_predicted_topK)

    hit_ratio, mrr = evaluate_location(all_ground_truth_location, all_predicted_topK)

    if save_filename is not None:
        filename = os.path.join(params_path, save_filename + '_results.npz')
        np.savez(filename, all_ground_truth_location=all_ground_truth_location, all_predicted_topK=all_predicted_topK)

    return all_loss_s, hit_ratio, mrr


# for downstream validation bencnmark
def get_s_baselines_total_loss_s_for_CACSR_DEMO_DOWN(loader, model, downstream='POI_RECOMMENDATION', save_filename=None, params_path=None):
    all_loss_s = []
    all_ground_truth_users = []
    all_predicted_topK = []
    for input in loader:
        s_loss_score, top_k_pred = model(input, mode='downstream', downstream=downstream)
        all_loss_s.append(s_loss_score.detach().cpu().numpy())
        if downstream == 'POI_RECOMMENDATION':
            all_ground_truth_users.append(input.Y_location.cpu().numpy())
            # all_ground_truth_users.append(torch.index_select(torch.tensor(input.Y_location), dim=0, index=indice).cpu().numpy())
        elif downstream == 'TUL':
            all_ground_truth_users.append(input.X_users.cpu().numpy())
        else:
            raise ValueError('downstream is not in [POI_RECOMMENDATION, TUL]')

        all_predicted_topK.append(top_k_pred.cpu().numpy())

    all_loss_s = np.array(all_loss_s)
    all_loss_s = np.mean(all_loss_s)

    all_ground_truth_users = np.concatenate(all_ground_truth_users)
    all_predicted_topK = np.concatenate(all_predicted_topK)

    hit_ratio, mrr = evaluate_location(all_ground_truth_users, all_predicted_topK)

    if save_filename is not None:
        filename = os.path.join(params_path, save_filename + '_results.npz')
        np.savez(filename, all_ground_truth_users=all_ground_truth_users, all_predicted_topK=all_predicted_topK)

    return all_loss_s, hit_ratio, mrr


# for downstream validation bencnmark
def get_s_baselines_total_loss_s_for_CACSR_DOWN(loader, model, downstream='POI_RECOMMENDATION', save_filename=None, params_path=None):
    all_loss_s = []
    all_ground_truth_users = []
    all_predicted_topK = []
    for input in loader:
        s_loss_score, top_k_pred, indice = model(input, mode='downstream', downstream=downstream)
        all_loss_s.append(s_loss_score.detach().cpu().numpy())
        if downstream == 'POI_RECOMMENDATION':
            all_ground_truth_users.append(torch.index_select(torch.tensor(input.Y_location), dim=0, index=indice).cpu().numpy())
        elif downstream == 'TUL':
            all_ground_truth_users.append(input.X_users.cpu().numpy())
        else:
            raise ValueError('downstream is not in [POI_RECOMMENDATION, TUL]')

        all_predicted_topK.append(top_k_pred.cpu().numpy())

    all_loss_s = np.array(all_loss_s)
    all_loss_s = np.mean(all_loss_s)

    all_ground_truth_users = np.concatenate(all_ground_truth_users)
    all_predicted_topK = np.concatenate(all_predicted_topK)

    hit_ratio, mrr = evaluate_location(all_ground_truth_users, all_predicted_topK)

    if save_filename is not None:
        filename = os.path.join(params_path, save_filename + '_results.npz')
        np.savez(filename, all_ground_truth_users=all_ground_truth_users, all_predicted_topK=all_predicted_topK)

    return all_loss_s, hit_ratio, mrr


def get_semantic_information(cnt2category, data_root):
    import pickle
    vecpath = data_root + "glove.twitter.27B.50d.pkl"
    pkl_data = open(vecpath, "rb")
    word_vec = pickle.load(pkl_data)
    for word in word_vec.keys():
        word_vec[word] = word_vec[word]
    pkl_data.close()

    word_id = 0
    dataset_word_vec = []
    dataset_word_index = {} 
    categories = cnt2category.values()
    for category in categories:
        words = category.split(" ")
        # print(words)
        for word in words:
            word = word.lower()
            if (word in word_vec) and (word not in dataset_word_index): 
                dataset_word_index[word] = word_id
                word_id += 1
                dataset_word_vec.append(word_vec[word])
    print("word_index: ", dataset_word_index)
    return dataset_word_vec, dataset_word_index, word_id


