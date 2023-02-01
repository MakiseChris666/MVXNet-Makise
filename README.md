This is an unofficial implementation of <b>MVXNet</b>, 
which is base on `Faster R-CNN with ResNet50 and FPN` and 
`VoxelNet`.

The image branch(Faster R-CNN) is from pre-trained model in 
`torchvision`, which specifically is `torchvision.models.detection.faster_rcnn.fasterrcnn_resnet50_fpn()`.

The backbone(VoxelNet) is of the very original VoxelNet, which
is implemented by myself, you can check the model [here](https://github.com/MakiseChris666/VoxelNet-Makise).
Note that the VoxelNet used in this project is not the same as that.

To train the model you can use: 
> python train.py &lt;dataroot&gt; &lt;iterations&gt; &lt;lastiteration&gt;

and the dataset should be arranged as:

``````````
<dataroot>
|--ImageSets  
    |--train.txt # this .txt contains the samples' indices to be in training set 
    |--val.txt # to be in validation set 
|--training 
    |--calib 
    |--image_2 
    |--label_2 
    |--velodyne
````````````

Before you train the model, use the following command to preprocess the data:
> python cropdata.py &lt;dataroot&gt; &lt;mode&gt; \
> python create_gtdatabase.py &lt;dataroot&gt;

`cropdata.py` will create a folder `velodyne_croped` in 
`<dataroot>/training`, and the training process will use
the processed data.

`mode` can be `numpy`, `torch`, `torch-cuda`, default `numpy`.

`create_gtdatabase.py` will crop the ground truth samples out
of original data, and save them in `<dataroot>/training/gtdatabase`.
These data will be used in data augmentation.

Before you create gt database, please get `update_train_2020.json`
from [KINS Dataset](https://github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset),
and put the file into `MVXNet-Makise/seglabel`.