
agroScan - v1 2023-05-28 4:15pm
==============================

This dataset was exported via roboflow.com on May 28, 2023 at 10:50 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 54 images.
Mango are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Randomly crop between 0 and 51 percent of the image
* Random shear of between -26° to +26° horizontally and -25° to +25° vertically
* Random brigthness adjustment of between -40 and +40 percent
* Random Gaussian blur of between 0 and 2.5 pixels
* Salt and pepper noise was applied to 11 percent of pixels

The following transformations were applied to the bounding boxes of each image:
* 50% probability of horizontal flip


