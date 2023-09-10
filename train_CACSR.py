import argparse
import configparser
# from tensorboardX import SummaryWriter
import preprocess.load_data_for_CACSR as preprocess 
from model.CACSR import *
from copy import deepcopy
from utils import *
from torch.nn.utils import clip_grad_norm_
import torch
import torch.nn as nn
import time

# read hyper-param settings
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='config/CACSR_nyc.conf', type=str,
                    help="configuration file path")
parser.add_argument("--dataroot", default='data/', type=str,
                    help="data root directory")
args = parser.parse_args()
config_file = args.config
data_root = args.dataroot
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
print('>>>>>>>  configuration   <<<<<<<')
with open(config_file, 'r') as f:
    print(f.read())
print('\n')
config.read(args.config)
data_config = config['Data']
training_config = config['Training']
model_config = config['Model']

# Data config
dataset_name = data_config['dataset_name']
max_his_period_days = data_config['max_his_period_days']
max_merge_seconds_limit = data_config['max_merge_seconds_limit']
max_delta_mins = data_config['max_delta_mins']
min_session_mins = data_config['min_session_mins']
least_disuser_count = data_config['least_disuser_count']
least_checkins_count = data_config['least_checkins_count']
latN = data_config['latN']
lngN = data_config['lngN']
split_save = bool(int(data_config['split_save']))
dataset_name = dataset_name + '_' + max_his_period_days + 'H' + max_merge_seconds_limit + 'M' + max_delta_mins + 'd' + min_session_mins + 's' + least_disuser_count + 'P' + least_checkins_count + 'U'

# Training config
mode = training_config['mode'].strip()
ctx = training_config['ctx']
os.environ["CUDA_VISIBLE_DEVICES"] = ctx
USE_CUDA = torch.cuda.is_available()
print("CUDA:", USE_CUDA, ctx)
device = torch.device("cuda" if USE_CUDA else "cpu")
print('device:', device)
use_nni = bool(int(training_config['use_nni']))
regularization = float(training_config['regularization'])
learning_rate = float(training_config['learning_rate'])
max_epochs = int(training_config['max_epochs'])
display_step = int(training_config['display_step'])
patience = int(training_config['patience'])
train_batch = int(training_config['train_batch'])
val_batch = int(training_config['val_batch'])
test_batch = int(training_config['test_batch'])
batch_size = int(training_config['batch_size'])
save_results = bool(int(training_config['save_results']))

specific_config = 'CACSR'

# Model Setting
loc_emb_size = int(model_config['loc_emb_size'])
tim_emb_size = int(model_config['tim_emb_size'])
user_emb_size = int(model_config['user_emb_size'])
hidden_size = int(model_config['hidden_size'])
adv = int(model_config['adv'])
rnn_type = model_config['rnn_type']
num_layers = int(model_config['num_layers'])
downstream = model_config['downstream']

#### need nni ####
loc_noise_mean = float(model_config['loc_noise_mean'])
loc_noise_sigma = float(model_config['loc_noise_sigma'])
tim_noise_mean = float(model_config['tim_noise_mean'])
tim_noise_sigma = float(model_config['tim_noise_sigma'])
user_noise_mean = float(model_config['user_noise_mean'])
user_noise_sigma = float(model_config['user_noise_sigma'])
tau = float(model_config['tau'])
pos_eps = float(model_config['pos_eps'])
neg_eps = float(model_config['neg_eps'])
dropout_rate_1 = float(model_config['dropout_rate_1'])
dropout_rate_2 = dropout_rate_1
self_weight = float(model_config['self_weight'])

if use_nni:
    import nni
    param = nni.get_next_parameter()
    # multi-dataset
    dataset_name = param['dataset_name']
    batch_size = int(param['batch_size'])
    loc_emb_size = int(param['loc_emb_size'])
    tim_emb_size = int(param['tim_emb_size'])
    user_emb_size = int(param['user_emb_size'])
    hidden_size = int(param['hidden_size'])
    rnn_type = param['rnn_type']
    num_layers = int(param['num_layers'])

    loc_noise_mean = float(param['loc_noise_mean'])
    loc_noise_sigma = float(param['loc_noise_sigma'])
    tim_noise_mean = float(param['tim_noise_mean'])
    tim_noise_sigma = float(param['tim_noise_sigma'])
    user_noise_mean = float(param['user_noise_mean'])
    user_noise_sigma = float(param['user_noise_sigma'])
    tau = float(param['tau'])
    pos_eps = float(param['pos_eps'])
    neg_eps = float(param['neg_eps'])
    dropout_rate_1 = float(param['dropout_rate_1'])
    dropout_rate_2 = dropout_rate_1
    self_weight = float(param['self_weight'])

train_batch = batch_size
val_batch = batch_size
test_batch = batch_size

print('load dataset:', dataset_name)
print('split_save:', split_save)

# Data
if data_config['dataset_name'] == "www_NYC" or data_config['dataset_name'] == "TSMC_www_NYC":
    data = np.load(data_root + "nyc_cnt2category2cnt.npz", allow_pickle=True)
elif data_config['dataset_name'] == "www_JKT":
    data = np.load(data_root + "jkt_cnt2category2cnt.npz", allow_pickle=True)
elif data_config['dataset_name'] == "www_IST":
    data = np.load(data_root + "ist_cnt2category2cnt.npz", allow_pickle=True)
elif data_config['dataset_name'] == "www_TKY" or data_config['dataset_name'] == "TSMC_www_TKY":
    data = np.load(data_root + "tky_cnt2category2cnt.npz", allow_pickle=True)
else:
    data = np.load(data_root + "nyc_cnt2category2cnt.npz", allow_pickle=True)


# cnt2category = data['cnt2category']
# print("cnt2category: ", type(cnt2category), cnt2category) # numpy.ndarry 'dict'
# print("cnt2category: ", cnt2category.shape) # ()
# print("cnt2category: ", cnt2category.size) # size=1
# assert(1==0)

cnt2category = data['cnt2category'].item()  # numpy.ndarray.item() category'index->category
print("cnt2category: ", type(cnt2category), cnt2category)

word_vec, word_index, text_size = get_semantic_information(cnt2category, data_root)

print('Loading data...')
data_train, data_val, data_test, feature_category, feature_lat, feature_lng, latN, lngN, category_cnt = preprocess.load_dataset_for_CACSR(
    dataset_name, save_split=split_save, data_root=data_root, device=device)
print("feature_category: ", feature_category.shape,
      feature_category)  # feature_category[venue's index] -> venue's category's index

collate = preprocess.collate_session_based  # padding sequence with variable len

dl_train = torch.utils.data.DataLoader(data_train, batch_size=train_batch, shuffle=True,
                                       collate_fn=collate)  
dl_val = torch.utils.data.DataLoader(data_val, batch_size=val_batch, shuffle=False, collate_fn=collate)
dl_test = torch.utils.data.DataLoader(data_test, batch_size=test_batch, shuffle=False, collate_fn=collate)

# Model setup
print('Building model...', flush=True)
# General model config
tim_size = 48  
fill_value = 0  
use_semantic = True  
if use_semantic:
    print("Use semantic information from venue name!")
else:
    print("Don't use semantic information from venue name!")
general_config = CACSR_ModelConfig(loc_size=int(data_train.venue_cnt), tim_size=tim_size,
                                  uid_size=int(data_train.user_cnt), tim_emb_size=tim_emb_size,
                                  loc_emb_size=loc_emb_size, hidden_size=hidden_size, user_emb_size=user_emb_size,
                                  device=device,
                                  loc_noise_mean=loc_noise_mean, loc_noise_sigma=loc_noise_sigma,
                                  tim_noise_mean=tim_noise_mean, tim_noise_sigma=tim_noise_sigma,
                                  user_noise_mean=user_noise_mean, user_noise_sigma=user_noise_sigma, tau=tau,
                                  pos_eps=pos_eps, neg_eps=neg_eps, dropout_rate_1=dropout_rate_1,
                                  dropout_rate_2=dropout_rate_2, rnn_type=rnn_type, num_layers=num_layers, downstream=downstream)
# Define model
model = CACSR(general_config).to(device)
print(model, flush=True)

params_path = os.path.join('experiments', dataset_name.replace('(', '').replace(')', ''), specific_config)
print('params_path:', params_path)

if use_nni:
    exp_id = nni.get_experiment_id()
    trail_id = nni.get_trial_id()
    best_name = str(exp_id) + '.' + str(trail_id) + 'best.params'
    params_filename = os.path.join(params_path, best_name)
else:
    best_name = 'best.params'
    params_filename = os.path.join(params_path, best_name)

if mode == 'train':
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("total_params:", total_params, flush=True)
    print("total_trainable_params:", total_trainable_params, flush=True)

    if os.path.exists(params_path):
        # shutil.rmtree(params_path)
        # os.makedirs(params_path)
        # print('delete the old one and create params directory %s' % (params_path), flush=True)
        print('already exist %s' % (params_path), flush=True)
    else:
        os.makedirs(params_path)
        print('create params directory %s' % (params_path), flush=True)

    print('Starting training...', flush=True)

    impatient = 0
    best_hit20 = -np.inf
    best_model = deepcopy(model.state_dict())
    global_step = 0
    best_epoch = -1
    # sw = SummaryWriter(logdir=params_path, flush_secs=5)
    opt = torch.optim.Adam(model.parameters(), weight_decay=regularization, lr=learning_rate, amsgrad=True)
    # opt = Adafactor(model.parameters())
    start = time.time()

    for epoch in range(0, max_epochs):

        model.train()

        for input in dl_train:

            opt.zero_grad()
            # print("train:",global_step)

            if adv == 1:
                s_loss_score, cont_loss, top_k_pred, indice = model(input, mode='train', adv=adv, downstream=downstream)
                loss_total = (1 - self_weight) * s_loss_score + cont_loss * self_weight
            else:
                s_loss_score, top_k_pred, indice = model(input, mode='train', adv=adv, downstream=downstream)
                loss_total = s_loss_score
            loss_total.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            global_step += 1
            if downstream == 'POI_RECOMMENDATION':
                # ys = input.Y_location  # (batch,)
                ys = torch.index_select(torch.tensor(input.Y_location), dim=0, index=indice)
            elif downstream == 'TUL':
                ys = input.X_users  # (batch,)
            else:
                raise ValueError('downstream is not in [POI_RECOMMENDATION, TUL]')

            loss_total = loss_total.item()
            # sw.add_scalar('training_loss_s', loss_total, global_step)
            hit_ratio, mrr = evaluate_location(ys.cpu().numpy(), top_k_pred.cpu().numpy())  # [k]
            # sw.add_scalar('training_mrr', mrr, global_step)
            # sw.add_scalar('training_hit_1', hit_ratio[0], global_step)
            # sw.add_scalar('training_hit_20', hit_ratio[19], global_step)

        model.eval()
        with torch.no_grad():
            all_loss_s_val, hit_ratio_val, mrr_val = get_s_baselines_total_loss_s_for_CACSR_DOWN(dl_val, model, downstream=downstream)
            # sw.add_scalar('validation_loss_s', all_loss_s_val, epoch)
            # sw.add_scalar('validation_mrr', mrr_val, global_step)
            # sw.add_scalar('validation_hit_1', hit_ratio_val[0], global_step)
            # sw.add_scalar('validation_hit_5', hit_ratio_val[4], global_step)
            # sw.add_scalar('validation_hit_10', hit_ratio_val[9], global_step)
            # sw.add_scalar('validation_hit_15', hit_ratio_val[14], global_step)
            # sw.add_scalar('validation_hit_20', hit_ratio_val[19], global_step)

            if (hit_ratio_val[19] - best_hit20) < 1e-4: 
                impatient += 1
                if best_hit20 < hit_ratio_val[19]: 
                    best_hit20 = hit_ratio_val[19]
                    best_model = deepcopy(model.state_dict())
                    best_epoch = epoch
            else:
                best_hit20 = hit_ratio_val[19]
                best_model = deepcopy(model.state_dict())
                best_epoch = epoch
                impatient = 0

            if impatient >= patience:
                print('Breaking due to early stopping at epoch %d,best epoch at %d' % (epoch, best_epoch), flush=True)
                break

            if (epoch) % display_step == 0:
                print('Epoch %4d, train_loss=%.4f, val_loss=%.4f, val_mrr=%.4f, val_hit_1=%.4f, val_hit_20=%.4f' % (
                epoch, loss_total, all_loss_s_val, mrr_val, hit_ratio_val[0], hit_ratio_val[19]), flush=True)

            if use_nni:
                nni.report_intermediate_result(hit_ratio_val[19])

        torch.save(best_model, params_filename)

    print("best epoch at %d" % best_epoch, flush=True)
    print('save parameters to file: %s' % params_filename, flush=True)
    print("training time: ", time.time() - start)

### Evaluation
print('----- test ----')
model.load_state_dict(torch.load(params_filename))
model.eval()
with torch.no_grad():
    train_all_loss_s, train_hit_ratio, train_mrr = get_s_baselines_total_loss_s_for_CACSR_DOWN(dl_train, model, downstream=downstream)
    val_all_loss_s, val_hit_ratio, val_mrr = get_s_baselines_total_loss_s_for_CACSR_DOWN(dl_val, model, downstream=downstream)
    test_all_loss_s, test_hit_ratio, test_mrr = get_s_baselines_total_loss_s_for_CACSR_DOWN(dl_test, model, downstream=downstream)

    print('Dataset\t loss\t hit_1\t hit_3\t hit_5\t hit_7\t hit_10\t hit_15\t hit_20\t MRR\t\n' +
          'Train:\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t\n' % (
          train_all_loss_s, train_hit_ratio[0], train_hit_ratio[2], train_hit_ratio[4], train_hit_ratio[6],
          train_hit_ratio[9], train_hit_ratio[14], train_hit_ratio[19], train_mrr) +
          'Val:\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t\n' % (
          val_all_loss_s, val_hit_ratio[0], val_hit_ratio[2], val_hit_ratio[4], val_hit_ratio[6], val_hit_ratio[9],
          val_hit_ratio[14], val_hit_ratio[19], val_mrr) +
          'Test:\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t\n' % (
          test_all_loss_s, test_hit_ratio[0], test_hit_ratio[2], test_hit_ratio[4], test_hit_ratio[6],
          test_hit_ratio[9], test_hit_ratio[14], test_hit_ratio[19], test_mrr), flush=True)

    if use_nni:
        nni.report_final_result(val_hit_ratio[19])
