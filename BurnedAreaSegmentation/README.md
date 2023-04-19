# Burned Area Segmentation

## Container
A docker container has been provided that contains all necessary software to train the models:
```
jalexhurt/sc23-burned-area-segmentation
```

# Data processing pipeline
The pipeline consists of 6 steps for which the yml files are saved in **canada_2019_sentinel_pipeline** folder:
- 0.pull_repo
- 1.download
- 2.postprocessing
- 3.normalization
- 4.create_label
- 5.generate_chips
the jupyter notebook **generate_job_config_sentinel_pipeline.ipynb** can be used to auto-generate these yml files. The options can be chnaged within the notebook

# Deep Learning model training
The yml files are saved in **experiments** folder

## config file generation
The notebook **modify_train_config.ipynb** is used to auto-generate the config files for training and evaluation of deep learning models

## training and evaluation yml files generation
The notebook **generate_yml_for_training.ipynb** is used for auto-generation of yml files for training deep learning models
The notebook **generate_yml_for_evaluation.ipynb** is used for auto-generation of yml files for the evaluation of the trained models

## Running of yml files
We can run each yml file manually as follows
```
kubectl apply -f example.yaml
```
We can also use a bash file to 
- Automatically submit multiple jobs
```

```
- Automatically delete multiple jobs
```

```

