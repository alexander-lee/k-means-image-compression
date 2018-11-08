# Image Compression via K-means Clustering (Color Quantization)

## Overview
This uses the Lloyd's K-means Clustering algorithm with RGB tuples in order to find the `k` colors that best represent the input image. The K-means Clustering algorithm was implemented by me for a better understanding of how the algorithm worked. In addition, this program also produces scatter plots of the colors to better visualize the clusters.

## Running the Program
**NOTE:**  Assuming you are using Python 3.x

```sh
python image_k_means.py <image_name> [-k <default: 10>] [--iterations <default: 20>] [--save-scatter]
```

* `-k` is the number of clusters you want to find
* `--iterations` is the number of iterations you want to run Lloyd's Algorithm
* `--save-scatter` is a flag to choose to save a Scatter Plot of the data **(WARNING: Very slow)**

## Results

#### Input Image:

#### 5-means Clustering (with 20 iterations)

#### 10-means Clustering (with 20 iterations)

#### 20-means Clustering (with 20 iterations)
