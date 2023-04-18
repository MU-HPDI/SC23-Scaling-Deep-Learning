# Deforestation Detection

## Container
A docker container has been provided that contains all necessary software to train the models:
```
jalexhurt/sc23-deforestation-detection
```

This container was used to train all [ChangeFormer](https://github.com/wgcban/ChangeFormer) models.

## Processing:
All steps are connected, so, step 1 output will be step 2 input. 

- To be able to process Step 1:
This code will automatically download the sentinel 2 tiles for you, but to do that, you need to meet the following criterias: 

You need to set 2 environment variables to be able to run this code:
SENTINELSAT_USERNAME
SENTINELSAT_PASSWORD

If you don't have an account, you need to access this [link](https://scihub.copernicus.eu/) and create yours. Then add your credentials to the environment variables.

You also need to download the conservation units and the yearly deforestation files from [here](http://terrabrasilis.dpi.inpe.br/en/download-2/).

you can run the code like this:

```
python step1_download.py yearly_deforestation.shp conservation_units_amazon_biome.shp <sentinel_output_dataset_path>
```

## Training

To be able to train using ChangeFormer, you need to edit the "data_config.py" file, changing the DataConfig class, to add your dataset folder.

There is a Jupyter Notebook file who I have used for creating the yaml file for running the training process. Besides that, an example of a yaml file can be found in the same folder here.