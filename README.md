# Path finder detector with OpenCV and Keras using Colab

This learning project pretends to detect path and roads in satellite images. 
In order to train the model, we use Colab that provides free processing power on a GPU. [How to use Colab](https://medium.com/@alyafey22/train-on-google-colab-and-run-on-the-browser-a-case-study-8a45f9b1474e)

## Instructions

You have two options: 

### Use your own model and data

Create a directory with the following structure: 

```bash
data/
	item1/
		item1_001.png/jpg...
		item1_002.png/jpg...
	item2/
		item2_001.png/jpg...
		item2_002.png/jpg...
	....
```

If you want path images, please check the following [project](https://github.com/mblanchartf/PathDataset)

Then run the mountain_heatmap.py script that has the following arguments: 

```bash
python mountain_heatmap.py -h
usage: mountain_heatmap.py [-h] -m MOUNTAIN_NAME -tpath TRACKS_PATH
                           [--z [ZOOM]] [--pxe [PX_ERROR]]

Generate heatmap from GPX tracks

required arguments:
  -m MOUNTAIN_NAME, --mountain_name MOUNTAIN_NAME
                        Mountain name for output file
  -tpath TRACKS_PATH, --tracks_path TRACKS_PATH
                        Path to the GPX tracks

optional arguments:
  -h, --help            show this help message and exit
  --z [ZOOM], --zoom [ZOOM]
                        Satellite image zoom
  --pxe [PX_ERROR], --px_error [PX_ERROR]
                        Pixel error value to differentiate between tracks

```

Remember than you can use Colab if your computer is not powerful enough. 

In a Colab project, run: 

```bash
!python3 drive/Code/Colab/keras_path_model.py -dpath drive/Code/Colab/data
Model saved at drive/Code/Colab/data named 20180705174027_model_keras_path.hdf5
```

### Use an existing model

Models can be found in /models

First example of the path finder system. As you can see, it's not working properly, so we stopped the detection. But this is our first approach. 

<img src=screenshots/path_finder_first_solution.png width=100% />

**This section is still in progress.....**



## Authors

* **Marc Blanchart** - *Learning project* - [MarcBlanchart](https://github.com/mblanchartf)

