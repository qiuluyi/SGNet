## The Description of the SGNet
Road extraction from remote sensing images in very high resolution is important for autonomous driving and road planning. Compared with large-scale objects, roads are smaller, winding, and likely to be covered by buildings' shadows, causing deep convolutional neural networks (DCNNs) to be difficult to identify roads. The paper proposes a semantics-geometry framework (SGNet) with a two-branch backbone, i.e. semantics-dominant branch and geometry-dominant branch. The semantics-dominant branch inputs images to predict dense se-mantic features, and the geometry-dominant branch takes images to generate sparse boundary features. Then, dense semantic features and boundary details generated by two branches are adaptively fused. Further, by utilizing affinity between neighbor-hood pixels, a feature refinement module is proposed to refine textures and road details.

## The Overview of the SGNet
<img width="755" alt="image" src="https://user-images.githubusercontent.com/42291810/233682269-4db1233e-9f0f-487a-ba3f-87743a8907f2.png">

## Experimental Dataset
The Ottawa road dataset is used as the experiment dataset to evaluate the SGNet on road extraction and vector-ization. In experiments, 15 images were used as the training set (IDs 1-15),3 images as the validation set (IDs 16-18),and 2 images as the test set (IDs 19-20).


## Deployment Environment

|Name| Version|Build|Channel|
|-----| ----- |-----|-----|
|bzip2           |   1.0.8             |     h7b6447c_0|          |
|cudatoolkit     |   11.0.221          |     h6bb024c_0|          |
|freetype        |   2.12.1            |     h4a9f257_0|          |
|freexl          |   1.0.6             |     h27cfd23_0|          |
|gdal            |   2.2.4             | py36h04863e7_1|          |
|geos            |   3.6.2             |     heeff764_2|          |
|giflib          |   5.1.4             |     h14c3975_1|          |
|glib            |   2.63.1            |     h5a9c865_0|          |
|gst-plugins-base|   1.14.0            |     hbbd80ab_1|          |
|gstreamer       |   1.14.0            |     hb453b48_1|          |
|hdf4            |   4.2.13            |     h3ca952b_2|          |
|hdf5            |   1.8.18            |     h6792536_1|          |
|icu             |   58.2              |     he6710b0_3|          |
|imageio         |   2.15.0            |         pypi_0|     pypi |
|install         |   1.3.5             |         pypi_0|     pypi |
|intel-openmp    |   2022.1.0          |  h9e868ea_3769|          |
|joblib          |   1.1.0             |         pypi_0|     pypi |
|jpeg            |   9e                |     h7f8727e_0|          | 
|json-c          |   0.13.1            |     h1bed415_0|          |
|kealib          |   1.4.7             |     h5472223_5|          |
|kiwisolver      |   1.3.1             | py36h2531618_0|          |
|krb5            |   1.16.1            |     hc83ff2d_6|          |
|lcms2           |   2.12              |     h3be6417_0|          |
|lerc            |   3.0               |     h295c915_0|          |
|libboost        |   1.73.0            |    h28710b8_12|          |
|libcurl         |   7.61.0            |     h1ad7b7a_0|          |
|libdap4         |   3.19.1            |     h6ec2957_0|          |
|libdeflate      |   1.8               |     h7f8727e_5|          |
|libedit         |   3.1.20221030      |     h5eee18b_0|          |
|libffi          |   3.2.1             |  hf484d3e_1007|          |
|libgcc-ng       |   11.2.0            |     h1234567_1|          |
|libgdal         |   2.2.4             |     heea9cce_1|          |
|libgfortran-ng  |   7.5.0             |    ha8ba4b0_17|          |
|libgfortran4    |   7.5.0             |    ha8ba4b0_17|          |
|libgomp         |   11.2.0            |     h1234567_1|          |
|libkml          |   1.3.0             |     h096b73e_6|          |
|libnetcdf       |   4.6.1             |     h015f1c5_0|          |
|libpng          |   1.6.37            |     hbc83047_0|          |
|libpq           |   10.5              |     h1ad7b7a_0|          |
|libprotobuf     |   3.17.2            |     h4ff587b_1|          |
|libspatialite   |   4.3.0a            |    he475c7f_19|          |
|libssh2         |   1.8.0             |     h9cfc8f7_4|          |
|libstdcxx-ng    |   11.2.0            |     h1234567_1|          |
|libtiff         |   4.5.0             |     h6a678d5_1|          |
|libuuid         |   1.41.5            |     h5eee18b_0|          |
|libwebp-base    |   1.2.4             |     h5eee18b_1|          |
|libxcb          |   1.15              |     h7f8727e_0|          |
|libxml2         |   2.9.14            |     h74e7548_0|          |
|lz4-c           |   1.9.4             |     h6a678d5_0|          |
|matplotlib      |   3.3.4             | py36h06a4308_0|          |
|matplotlib-base |   3.3.4             | py36h62a2d02_0|          |
|mkl             |   2020.2            |            256|          |
|mkl-service     |   2.3.0             | py36he8ac12f_0|          |
|mkl_fft         |   1.3.0             | py36h54f3939_0|          |
|mkl_random      |   1.1.1             | py36h0573a6f_0|          |
|ml-collections  |   0.1.1             |         pypi_0|      pypi| 
|ncurses         |   6.4               |     h6a678d5_0|          |
|networkx        |   2.5.1             |         pypi_0|      pypi|
|ninja           |   1.10.2            |     h06a4308_5|          |
|ninja-base      |   1.10.2            |     hd09550d_5|          |
|numpy           |   1.19.2            | py36h54aff64_0|          |
|numpy-base      |   1.19.2            | py36hfa32c7d_0|          |
|olefile         |   0.46              |         py36_0|          |
|opencv-python   |   4.0.1.23          |         pypi_0|      pypi|
|openjpeg        |   2.4.0             |     h3ad879b_0|          |
|openssl         |   1.0.2u            |     h7b6447c_0|          |
|pandas          |   1.1.5             |         pypi_0|      pypi|
|pcre            |   8.45              |     h295c915_0|          |
|pillow          |   8.4.0             |         pypi_0|      pypi|
|pip             |   21.2.2            | py36h06a4308_0|          |
|pixman          |   0.40.0            |     h7f8727e_1|          |
|poppler         |   0.65.0            |     h581218d_1|          |
|poppler-data    |   0.4.11            |     h06a4308_1|          |
|proj4           |   5.0.1             |     h14c3975_0|          |
|protobuf        |   3.17.2            | py36h295c915_0|          |
|pycparser       |   2.21              |   pyhd3eb1b0_0|          |
|pyparsing       |   3.0.4             |   pyhd3eb1b0_0|          |
|pyqt            |   5.9.2             | py36h05f1152_2|          |
|python          |   3.6.6             |     h6e4f718_2|          |
|python-dateutil |   2.8.2             |   pyhd3eb1b0_0|          |
|pytz            |   2022.1            |         pypi_0|      pypi|
|pywavelets      |   1.1.1             |         pypi_0|      pypi|
|pyyaml          |   6.0               |         pypi_0|      pypi|
|qt              |   5.9.6             |     h8703b6f_2|          |
|readline        |   7.0               |     h7b6447c_5|          |
|scikit-image    |   0.17.2            |         pypi_0|      pypi|
|scikit-learn    |   0.24.2            |         pypi_0|      pypi|
|scipy           |   1.5.4             |         pypi_0|      pypi|
|setuptools      |   58.0.4            | py36h06a4308_0|          |
|sip             |   4.19.8            | py36hf484d3e_0|          |
|six             |   1.16.0            |   pyhd3eb1b0_1|          |
|sklearn         |   0.0               |         pypi_0|      pypi|
|sqlite          |   3.33.0            |     h62c20be_0|          |
|tensorboardx    |   2.2               |   pyhd3eb1b0_0|          |
|thop            |   0.0.31-2005241907 |           pypi|      pypi|
|threadpoolctl   |   3.1.0             |         pypi_0|      pypi|
|tifffile        |   2020.9.3          |         pypi_0|      pypi|
|timm            |   0.5.4             |         pypi_0|      pypi|
|tk              |   8.6.12            |     h1ccaba5_0|          |
|torch           |   1.10.1            |         pypi_0|      pypi|
|torchstat       |   0.0.7             |         pypi_0|      pypi|
|torchsummary    |   1.5.1             |         pypi_0|      pypi|
|torchvision     |   0.11.2            |         pypi_0|      pypi|
|tornado         |   6.1               | py36h27cfd23_0|          |
|typing-extension|s  4.1.1             |         pypi_0|      pypi|
|wheel           |   0.37.1            |   pyhd3eb1b0_0|          |
|xerces-c        |   3.2.4             |     h94c2ce2_0|          |
|xz              |   5.2.10            |     h5eee18b_1|          |
|yimage          |   1.1.0             |         pypi_0|      pypi|
|zlib            |   1.2.13            |     h5eee18b_0|          | 
|zstd            |   1.5.2             |     ha4553b 6_0|         |

#### If you make use of the method, please cite our following paper:
>
@ARTICLE{10105922,  
  author={Qiu, Luyi and Yu, Dayu and Zhang, Chenxiao and Zhang, Xiaofeng},    
  journal={IEEE Geoscience and Remote Sensing Letters},   
  title={A Semantics-Geometry Framework for Road Extraction from Remote Sensing Images},   
  year={2023},  
  doi={10.1109/LGRS.2023.3268647}}  
>
