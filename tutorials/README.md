# Tutorials

## Data Preprocessing Tutorial

In the notebook `data_prep_tutorial.ipynb`, we provide the code for preparing all the necessary data for estimating the total object count in a region.

The notebook will prepare following data are needed to run IS-Count:
* A binary mask with the same resolution as the covariate raster for the region of interest
* An "all-pixel" file that contains `[all_lats, all_lons, all_s_pix, all_probs['nl'], all_probs['pop']]` of the region of interest

Before running the notebook, make sure you have all the required data downloaded and have a directory under the root path named `sample_data/`.

* The covariate raster data could be downloaded [here](https://drive.google.com/drive/folders/1iIj_70_lT5ZGWGIcjlkpo9FDjzavdWTt?usp=sharing).
* The Microsoft Building Footprints data is available to download [here](https://github.com/microsoft/USBuildingFootprints).
* The country-wise Google Open Buildings data is available to download [here](https://colab.research.google.com/github/google-research/google-research/blob/master/building_detection/open_buildings_download_region_polygons.ipynb?authuser=1#scrollTo=qP6ADuzRdZTF).

The `sample_data/` folder needs to be structured as follows before you run the code in the notebook:

```
sample_data
├── covariates
│   ├── NL_raster.tif
│   └── population_raster.tif
├── ms_building_footprint
│   ├── us
│   │   ├── NewYork.geojson
│   │   └── ...
│   └── ...
├── open_buildings
│   ├── us
│   │   └── ...
│   └── ...
└── shapefiles
    └── us_states
        ├── cb_2018_us_state_20m.cpg
        ├── cb_2018_us_state_20m.dbf
        ├── cb_2018_us_state_20m.prj
        ├── cb_2018_us_state_20m.shp
        └── ...
```


## Count Estimation Tutorial

In the notebook `count_estimation_tutorial.ipynb`, we provide the code for actually estimating object count in a given region, using the data we prepared in the previous step.
