# Object Detection with Transformers

This directory provides the config files, k8s YAML files, and instructions to train analogous models.

## Container
A docker container has been provided that contains all necessary software to train the models:
```
jalexhurt/sc23-object-detection-transformers
```

All models were trained with this provided docker container.

## Training Jobs in Kubernetes

A template Kubernetes YAML Job Spec is provided: [train_job.yml](./train_job.yml).

A bash script ([train.sh](./train.sh)) is also provided that utilizes the _envsubst_ command to replace the environment variables in the YAML file with the approparite values which is then piped to `kubectl`

The appropriate train configuration files were placed in the correct directories on persistent storage prior to job kickoff

## Training Configuration Files
Training configs for the modified MMDetection package (included in the provided container) are provided in the [configs](./configs/) directory of this repository.

The paths used for the data in each config file reference the persistent storage on the cluster

The three datasets (DOTA, XView, RarePlanes) were saved to persistent storage on the Nautilus Hyperluster.
