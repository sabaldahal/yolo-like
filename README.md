## RUNNING THE CODE


### INSTALL required libraries:


`pip install -r requirements.txt`


---
### Download dataset


Download any object detection dataset. 

Make sure the dataset is annotated in `YOLO` format.

Split the dataset into train, test and val set.

```text
project-root/
├── dataset/
│   ├── train/
│   |   ├── images/
│   |   ├── labels/
│   ├── test/
│   |   ├── images/
│   |   ├── labels/
│   ├── val/
│   |   ├── images/
│   |   ├── labels/

```
Dataset used for this project: 
`https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes`

This dataset can automatically be downloaded and split into train, test and val set by running the file `dataset_downloader.py`


---
### Running main.py
This is the entry point to the code.

`python main.py --help`

```text
positional arguments:
  {train,detect}  Run mode
    train         Train a model
    detect        Run Prediction
```

Either pass `train` or `detect` as an argument to train or run prediction.


---
### Train the Model


To train the model run main.py by passing 'train' as an argument

`python main.py train`

Additional arguments can be passed before training the model.

Run this in the terminal to learn about the accepted parameters.

`python main.py train --help`

```text
options:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset root directory
  --classes CLASSES [CLASSES ...]
                        Class names
  --epochs EPOCHS
  --batch-size BATCH_SIZE
  --lr LR
  --img-size IMG_SIZE
```

---
### Run Prediction
---

`python main.py detect`

This selects a random image from the test set and runs the prediction.

Run this to learn about other arguments that can be passed before running the detection

`python main.py detect --help`
``` text
options:
  -h, --help            show this help message and exit
  --dataset DATASET
  --weights WEIGHTS     Path to trained weights
  --batch-size BATCH_SIZE
  --img-size IMG_SIZE
  ```



---
### CONSTANTS
When nothing is specified as arguments in the terminal, the program uses default values to run the code.

Default parameters can be found under `./utils/constants.py`

These parameters can be edited here instead of passing as arguments in the terminal.

---
### Plots
After the training is completed, `loss_history.json` file will be created under the folder `plot`

This data can be used to visualize the loss plots. To create the plots, run `python ./plot/plot_loss.py`.