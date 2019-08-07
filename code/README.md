# Timage – A Robust Time Series Classification Pipeline

timage is a generic time series classification pipeline using ResNet.
It was designed to wrok with the [UCR archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/), but can in general work on all time series.

## Requirements

* python > 3.0
* pip
* Click
* timage may be installed in a virtuelenv using pipenv **OR** system-wide. 

### Using pipenv
1. Install requirements
   ```
   pipenv install
   ```
2. Start shell
   ```
   pipenv shell
   ```

### Install system wide
1. Install requirements
   ```
   sudo pip install -r requirements.txt
   ```
2. Install timage
   ```
   sudo python setup.py install
   ```

## Usage

For general help call `--help`
```
timage --help
Usage: timage [OPTIONS] COMMAND [ARGS]...

Options:
  -l, --log-level [DEBUG|INFO|WARNING|ERROR|CRITICAL]
  --help                          Show this message and exit.

Commands:
  check-images
  create-images
  make-all-data-dir
  results
  single-results
  train
```

For command specific usages call `--help` on a specific command.



### Download Dataset
This software was written for the [UCR archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/), 
but will in general work with all data that can be converted to the input image format.

### Dowload ResNet Weights:

Constains the initial resnet50 and resnet152 weights used in timage.
* resnet50: [GitHub](https://github.com/fchollet/deep-learning-models/releases)
* resnet152: [GitHub](https://gist.github.com/flyyufelix/7e2eafb149f72f4d38dd661882c554a6)

### Create Images
The image creation expect the data in the format used by the UCR 2015 dataset. 
In order to use the 2018 data set in needs to be converted.

##### Convert UCR2018 to 2015 format

```bash
# Remove desktop files
find . -type f -name desktop.ini -exec rm {} \;
# replace tab with comma, !!! beware on Mac !!! \t does not work in Mac(BSD) sed. You have to type the tab stop directly in the statement
find . -type f -name '*.tsv' -exec sed -i -e 's/\t/,/g' {} \;
# remove file suffix
find . -type f -name '*.tsv' | while read f; do mv "$f" "${f%.*}"; done
```

#### Example: Convert UCR2015 to images

Create Reccurence-Plot images of resolution 224x224.

```
timage create-images \
    --data-dir=/data/UCRArchive_2018 \
    --image-dir=/data/images_2018 \
    --resolution=224x224 \
    --image-cls=recu
```

Create Spectral Images

```
timage create-images \
    --data-dir=/data/UCRArchive_2018 \
    --image-dir=/data/images_2018 \
    --resolution=224x224 \
    --image-cls=spec
```

### Train Network

Initial weights can be found in the weights folder

```
python -m timage train \
    --image-dir=/data/images_2018 \
    --log-dir=/logs \
    --resolution=224x224 \
    --image-cls=recu \
    --network-model=resnet50 \
    --nb-epoch=100 \
    --weights-path=/data/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
```

This will output a tensorboard log file to `/log-dir` as well as the model and weights

### Train All Network

Trains a network on all classes in all datasets from the UCR archive.
Before training, the all data dir must be created.

#### Create All data dir
Creates an folder "all" inside the output data dir. The folder must
The created folder must be moved into the timage folder structure and then be used as any other dataset in the UCR during training.

```bash
python -m timage make-all-data-dir --input-data-dir /data/images_2018 --output-data-dir /data/output
```

#### Train all network

```bash
python -m timage train \
    --image-dir=/data/images_2018 \
    --log-dir=/logs \
    --resolution=224x224 \
    --image-cls=recu \
    --network-model=resnet50 \
    --data-sets all \
    --nb-epoch=100 \
    --weights-path=/data/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
```


### Parse Results

Results can be parsed using the `results` command and providing the path to the tensorboard, e.g.:
```
python -m timage results --log-dir=/logs/19-01-01_01-01/resnet50_model/ReccurenceImage/224x224x1/
```

### Plotting confusion matrices:

conf-matrices can be called two ways. The most important switch for this is the presence of a label dictionary.
- For single models
- For models trained on all datasets

The main difference between both calls is that the "all-model" is used with a model trained on all classes and
enables you to classify all datatasets in the UCR archive with just one model.

#### Example call for single model:
conf-matrices-for-single-models is for one model / dataset in the UCR archive. 
Displays inner class misclassifications.
```bash
python -m timage conf-matrices \
        --experiment-dir \
        /logs/19-02-18_08-49 \
        --image-dir \
        /data/images_2018_znorm_new_labels \
        --network-model \
        resnet50 \
        --image-cls \
        recu \
        --resolution \
        224x224 \
        --out-dir \
        /data/confusion_matrix
```

Must not have a label dict. This is the primary trigger for the kind of confusion matrix to be plotted. 
The misclassifications to other datasets is shown by the size of unlabeled space in the confusion matrix plot. 

#### Important! Needs a labeldict! The image dir must not be the all_images dir! It must be the one with images of different classes separated in single folders
```bash
python -m timage conf-matrices \
        --experiment-dir \
        /logs/19-02-14_16-12 \
        --image-dir \
        /data/images_2018_znorm_new_labels \
        --network-model \
        resnet50 \
        --image-cls \
        recu \
        --resolution \
        224x224 \
        --out-dir \
        /data/confusion_matrix \
        --label-dict \
        /data/images_2018_znorm_new_labels_all/ReccurenceImage/224x224x1/all/train/labeldict.csv
```
