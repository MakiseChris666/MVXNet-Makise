This is an unofficial implementation of <b>MVXNet</b>, 
which is base on `Faster R-CNN with ResNet50 and FPN` and 
`VoxelNet`.

The image branch(Faster R-CNN) is from pre-trained model in 
`torchvision`, which specifically is `torchvision.models.detection.faster_rcnn.fasterrcnn_resnet50_fpn()`.

The backbone(VoxelNet) is of the very original VoxelNet, which
is implemented by myself, you can check the model [here](https://github.com/MakiseChris666/VoxelNet-Makise).
The VoxelNet used in this project is not the same as that.

To train the model you can use: 
> python train.py &lt;dataroot&gt; &lt;inforoot> &lt;iterations&gt; &lt;lastiteration&gt;

and the dataset should arranged as:

&lt;dataroot&gt; \
|--ImageSets # if &lt;inforoot&gt; is specified, this folder's content should be in inforoot \
&emsp;|--train.txt # this .txt contains the samples' number to be in training set \
&emsp;|--val.txt # to be in validation set \
|--training \
&emsp;|--calib \
&emsp;|--image_2 \
&emsp;|--label_2 \
&emsp;|--velodyne