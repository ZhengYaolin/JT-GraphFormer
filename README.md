# ST-GraphFormer
ST-GraphFormer: Effective Human Action Recognition with Spatio-Temporal Subgraph
## Acknowledgements
This repository is based on [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN) and [STTFormer](https://github.com/heleiqiu/STTFormer). Thanks to the original authors for their **outstanding** work!!!

# Data Preparation
This section references the [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN).

### Download datasets.

#### There are 3 datasets to download:

- NTU RGB+D 60 Skeleton
- NTU RGB+D 120 Skeleton
- N-UCLA (NW-UCLA)
#### NTU RGB+D 60 and 120
1. Request dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:
   1. `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60)
   2. `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)
   3. Extract above files to `./gendata/nturgbd_raw`
#### N-UCLA (NW-UCLA)
1. Download dataset from [here](https://www.dropbox.com/s/10pcm4pksjy6mkq/all_sqe.zip?dl=0)
2. Move `all_sqe` to `./gendata/n_ucla`
### Data Processing

#### Directory Structure

Put downloaded data into the following directory structure:

```
- gendata/
  - n_ucla/
    - all_sqe
      ... # raw data of NW-UCLA
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
```
#### Generating Data

- Generate NTU RGB+D 60 or NTU RGB+D 120 dataset:

```
 cd ./gendata/ntu # or cd ./gendata/ntu120
 # Get skeleton of each performer
 python3 get_raw_skes_data.py
 # Remove the bad skeleton 
 python3 get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python3 seq_transformation.py
```

# Training & Testing
Since we work in an environment that allows easy modification of config files, parameter changes are made in config files. We recommend creating more config files for different parameter combinations and saving them regularly.

### Training

- Change the config file depending on what you want.

```
# Example:
python main.py --config config/gf2.yaml
```

- To train model on NTU RGB+D 60/120 with bone or motion modalities, setting `data_mode` arguments in the config file.
- To train model using Koopman operator, please load the weights of the saved fc-based model, and set the `freeze_weights: True` . After that, loading the weights of the new weights, setting the `freeze_weights: False` and the arguments in 'optim'. You can reference these two configuration files:
```
- config/
  - gf2_k.yaml
  - gf2_koopman.yaml
```

### Testing

- To test the trained models saved in <work_dir1> and get the `score.pkl` for ensemble, set the `work_dir: <the path for saving score.pkl>` in the config file and the the `weights: <work_dir1/<saved weight>.pt>`, and run following command:
```
python koopman_train.py --config config/test.yaml
```
- To ensemble the results of different modalities, we recommend setting the directories of the test results, scores.pkl, as:
```
- pt_saved/
  - pt_120xsub/
    - joint
      - l2  # different amount of frames for ST-Graph
        - score.pkl
      - l4
      ...
    - bone
      - l2
      ...
    ...
  ...
```
Then you can run following command:
```
# Example: ensemble four modalities on NTU RGB+D 120 cross subject
python ensemble.py --datasets ./gendata/ntu120/NTU120_XSub.npz --joint_dir ./pt_saved/pt_120xsub/joint --bone_dir ./pt_saved/pt_120xsub/bone --joint_motion_dir ./pt_saved/pt_120xsub/joint_motion --bone_motion_dir ./pt_saved/pt_120xsub/bone_motion
```

### Pretrained Models

- The pretrained models will be available soon.
