# CACSR
Contrastive learning framework using Adversarial perturbations for Check-in Sequence Representation (CACSR)
# Datasets
To demonstrate the superiority of our proposed model, our experiements are carried out on three real-world datasets derived from the raw Gowalla, and Foursquare of New York City(NYC) and Jakarta(JKT) check-in data.

In order to facilitate the training of our model, we screened out relatively high-quality trajectory sequences through some conditions. For all datasets, we take the historical time as long as no more than 120 days. We filter out users with at least 10 records, and places visited at least 10 times. 

The below table shows the statistics of three datasets.
|             |  Gowalla  |    NYC    |    JKT    |
| ----------  |:---------:|:---------:|:---------:|
| # users     | 5,016   | 1,596  | 3,943 |
| # locations | 4,873   | 5,529  | 9,563   |
| # check-ins | 154,253 | 39,975 | 119,317 |

# Requirements
- python >= 3.6
- PyTorch >= 1.8

# Prepare data
  - Download raw data from following sources:
    - <a href='https://drive.google.com/drive/folders/1nsGjpQxEe4h4AQT9i3V2f0tUndUIiACa?usp=sharing'>Google Drive</a>
      - https://drive.google.com/drive/folders/1nsGjpQxEe4h4AQT9i3V2f0tUndUIiACa?usp=sharing 
    - <a href='https://1drv.ms/u/s!Ail1jqmbAhYKkVFBI5B0nF8cHLaE?e=xdgn0W'>One Drive</a> 
      - https://1drv.ms/u/s!Ail1jqmbAhYKkVFBI5B0nF8cHLaE?e=xdgn0W
    - <a href='https://pan.baidu.com/s/1yKBDXZxnIX9r1km7gh3tyg'>BaiduNetDisk</a> code: `yif1`
      - https://pan.baidu.com/s/1yKBDXZxnIX9r1km7gh3tyg
  - Copy all files and directories to `CACSR-main/data`
  ## FourSquare (NYC):
  - Enter directory `CACSR-main/prepare_data` and run the command below:
    `python prepare_data.py --config prepare_config/prepare_4squ_nyc.conf --dataroot ../data/`
  ## FourSquare (JKT):
  - Enter directory `CACSR-main/prepare_data` and run the command below:
    `python prepare_data.py --config prepare_config/prepare_4squ_jkt.conf --dataroot ../data/`
  ## Gowalla
  - Enter directory `CACSR-main/prepare_data` and run the command below:
    `python prepare_data.py --config prepare_config/prepare_gow.conf --dataroot ../data/`

# Usage :
  Enter directory `CACSR-main`.
  Downstream tasks:
  Location Prediction (LP) or Trajectory User Link (TUL).
  You need to modify the `downstream` field in configuration file to choose the type of tasks.
  - Train model on NYC of LP task:
    `python train_CACSR.py --config config/CACSR_nyc_LP.conf --dataroot data/`
    <br>
  - Train model on JKT of LP task:
    `python train_CACSR.py --config config/CACSR_jkt_LP.conf --dataroot data/`
    <br>
  - Train model on GOW of LP task:
    `python train_CACSR.py --config config/CACSR_gow_LP.conf --dataroot data/`
    <br>
  - Train model on NYC of TUL task:
    `python train_CACSR.py --config config/CACSR_nyc_TUL.conf --dataroot data/`
    <br>
  - Train model on JKT of TUL task:
    `python train_CACSR.py --config config/CACSR_jkt_TUL.conf --dataroot data/`
    <br>
  - Train model on GOW of TUL task:
    `python train_CACSR.py --config config/CACSR_gow_TUL.conf --dataroot data/`

# Configuration
The configuration file `CACSR_*.conf` contains three parts: Data, Training and Model:

## Data
- dataset_name: The name of the datasets, represents NYC, JKT or Gowalla.
- max_his_period_days: The max history time.
- max_merge_seconds_limit: To judge whether two identical locations are the same event.
- max_delta_mins: To limit the prediction range.
- least_disuser_count: To filter locations, keep locations which have at least * users.
- least_checkins_count: To filter users, keep users who have at least * checkins.
- split_save: 1 or 0, representing whether datasets are split saved.

## Training
- mode: train for default, 
- ctx: cuda index, 0 for default
- regularization: float, regularization factor.
- learning_rate: float
- max_epochs: int
- display_step: int
- patience: int, for early stopping.
- train_batch: int
- val_batch: int
- test_batch: int
- batch_size: int
- save_results: bool

## Model
- adv: 0 or 1, enable adversarial or not.
- downstream: POI_RECOMMENDATION or TUL, representing LP and TUL respestively.

The remaining parameters are the best parameters of the model.