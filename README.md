# Hemp Segmentation and Classification, a Deep Learning approach

The  yield  of  crops  such  as  hemp  (Cannabis)  depends  on  various  factors,  such  as  environmental influences,  location  site  and  lighting  conditions. Therefore,  the  task  of  quantifying the volume  and the number  of plants per unit area as well as to classify different species in the same field is critical for plant breeders to get an exact overview of the current situation in the field. For large scale sites it is almostinfeasible to manually count and estimate the volume of such fields. To automate this process, a machine learning approach, based on deep convolutional neural networks  (CNN), is proposed to generate pixel level segmentation mapsfor area and volume calculationsand to estimate the number of plants on non-overlappingcases from UAV-images.It could have been shown  that a reasonable number of training samples  already  yield  good  results.  With  post-processing steps such as majority voting which is taking shots on multiple dates into account the method delivers good results on images where it is difficult even for human individualsto labelthe plants correctly. The overall segmentation rate is above 86% measured by the dice coefficient and an accuracy of about 84.5% and higher regarding the plant counting is reported. The developed solution is therefore a robust method to segment and detect Hemp plants under real-world conditions.

![alt text](docs/image/detection1.png)

![alt text](docs/image/segmentation1.png)

Hemp Segmentation and Detection (predictions on test-set): Green and Red corresponds to different Species

#### Required Packages:
see: [requirements.txt](requirements.txt)

# Documentation
* [Doc_HempSegmentation_DamianSchori.pdf](docs/Doc_HempSegmentation_DamianSchori.pdf)
 
 # [Notebooks:](notebooks/)
 * [Hemp Segmentation](notebooks/Hemp_Segmentation.ipynb) Summary of the steps described in the documentary
 * [Multispectral_Analysis](notebooks/Multispectral_Analysis.ipynb) Analysis of multispectral images for the species classification
 
# [Utils:](utils/)
* [hemp_segmentation](utils/hemp_segmentation.py) Tools for Hemp- Segmentation, including U-Net code
* [data interface](utils/data_interface.py) Data Interface class to load image / mask pairs from the georeferenced .tif files
* [evaluation](utils/utils.py) Tools for evaluation Purposes

# Data:
Password: "hemp-segmentation"
* [Pretrained Hemp Model](https://drive.switch.ch/index.php/s/ARkdCvvkFrVFbAN) The trained model on the hemp-plants as tensorflow.keras model
* [Image Data](https://drive.switch.ch/index.php/s/ARkdCvvkFrVFbAN) The image data used for training and testing


## Sample usage:

Example for usage:

```python

from utils.data_interface import Dataset, Data_Interface
from utils.evaluation import *
from utils.hemp_segmentation import *

class Config():
    dates = ['20190703', '20190719', '20190822']
    fields = ['Field_A', 'Field_C']

#Create Datasets:
d_0703_A = Dataset(name=Config.dates[0] + '_rgb_A', 
            date=Config.dates[0],
            rgb_path='../data/rasters/' + Config.dates[0] + '/rgb.tif',
            ms_path=None,
            mask_shapefile=Plant_Masks[Config.dates[0] + Config.fields[0]],
            outer_shapefile=Fields['field_A'],
            rgb_bands_to_read=[0, 1, 2],
            ms_bands_to_read=None,
            slice_shape=(384, 384))

d_0719_A = Dataset(name=Config.dates[1] + '_rgb_A',  
            date=Config.dates[1],
            rgb_path='../data/rasters/' + Config.dates[1] + '/rgb.tif',
            ms_path=None,
            mask_shapefile=Plant_Masks[Config.dates[1] + Config.fields[0]],
            outer_shapefile=Fields['field_A'],
            rgb_bands_to_read=[0, 1, 2],
            ms_bands_to_read=None,
            grid=d_0703_A.grid.copy(),
            slice_shape=(384, 384))

d_0822_A = Dataset(name=Config.dates[2] + '_rgb_A',  
            date=Config.dates[2],
            rgb_path='../data/rasters/' + Config.dates[2] + '/rgb.tif',
            ms_path=None,
            mask_shapefile=Plant_Masks[Config.dates[2] + Config.fields[0]],
            outer_shapefile=Fields['field_A'],
            rgb_bands_to_read=[0, 1, 2],
            ms_bands_to_read=None,
            grid=d_0703_A.grid.copy(), slice_shape=(384, 384))

#Create Data Interface:
di_test_A = Data_Interface([d_0703_A, d_0719_A, d_0822_A], {1001 : 1, 1005 : 2})

#Predict on all three dates:
imgs, msks = di_test_A.get_pair_on_same_date()
preds = model.predict(imgs)

#Apply Majority Vote:
sev = Segmentation_Evaluation(model)
pred = sev.majority_vote(preds, preds[1])
msk = sev.preprocess_mask(msks[1])

_, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(imgs[1])
ax[1].imshow(pred)
ax[2].imshow(msk)
ax[0].set_title("Input")
ax[1].set_title("Prediction with Majority Voting")
ax[2].set_title("Ground Truth Mask")

```

Results in:

![alt text](docs/image/output1.png)
