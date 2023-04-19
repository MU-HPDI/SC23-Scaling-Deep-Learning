# Deforestation Detection

## Container
A docker container has been provided that contains all necessary software to train the models:
```
jalexhurt/sc23-deforestation-detection
```

This container was used to train all [ChangeFormer](https://github.com/wgcban/ChangeFormer) models.

## Data Download and Processing

The data processing pipeline consists of sequential steps, where the output of one step serves as the input for the subsequent step.

### Step 1: Downloading Sentinel-2 Tiles 

Ensure you meet the following requirements:

1. **Create a Sentinel-2 API Account:** If you do not have an account, visit the [Copernicus Open Access Hub](https://scihub.copernicus.eu/) to create one.

2. **Set Environment Variables:** After registration, configure the following environment variables with your Sentinel-2 API credentials:
   
   - `SENTINELSAT_USERNAME`
   - `SENTINELSAT_PASSWORD`

3. **Download Required Data:** Obtain the conservation units and yearly deforestation files from the [TerraBrasilis website](http://terrabrasilis.dpi.inpe.br/en/download-2/).

With the prerequisites in place, execute the `step1_download.py` script as follows:

```
python step1_download.py yearly_deforestation.shp conservation_units_amazon_biome.shp <sentinel_output_dataset_path>
```

This command will automatically download the Sentinel-2 tiles, using the provided shapefiles for yearly deforestation and conservation units, and save the output to the specified dataset path.

### Post Step 1: Tile Selection and Subsequent Steps

After completing Step 1, follow the instructions below:

1. **Tile Selection:** Carefully review and manually select the highest quality tiles for use in the subsequent steps of the pipeline.

2. **Execute Remaining Steps:** With the optimal tiles selected, proceed to execute Steps 2, 3, 4, and 5. Ensure that each step is completed successfully before proceeding to the next one.

## Training with ChangeFormer

In order to successfully run the training, you need to modify the `data_config.py` file by updating the `DataConfig` class. This includes specifying the path to your dataset folder.

The provided Jupyter Notebook file enables you to automatically generate the YAML configuration files required for executing the training process. Additionally, an example YAML file is included in the same directory for your reference.
