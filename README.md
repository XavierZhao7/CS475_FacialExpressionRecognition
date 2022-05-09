# CS475_facial expression recognition

This project uses tensorflow. 

For Apple M1 and newest chip carried MacBook Pro, refer to env_apple_M1.sh

If you find an error message that a certain package is not found, you will need to install the package and contact either one of us at:
<xzhao77@jhu.edu>,<ylu106@jhu.edu>,<zsha2@jhu.edu>,<pxu11@jhu.edu>, if you still have troubles.

## Here are the instructions to run the facial expression recognition application in src.
> python src/facial_recog_camera.py -i <input video directory/ 0 for camera> -o <output file directory (preferred .avi format)>

**example:**
 
> python src/facial_recog_camera.py -i data/test_video.mp4 -o output/out.avi

> python src.facial_recog_camera.py -i 0 -o output/cameraout.avi

## Here are the instructions to train model in src.

> python src/train.py --dataset <#name of dataset> --model <#name of model> --epoches <#number of epochs> --batch-size <#batch size>

Note that the <#name of dataset> is either combined, raf, or ck+.

Note also that <#name of model> is either SimpleCNN3, GaborCNN3,or VGG16.


## Data set
Since we cannot copy, publish or distribute any portion of the [RAF database](http://www.whdeng.cn/raf/model1.html#:~:text=Real%2Dworld%20Affective%20Faces%20Database%20(RAF%2DDB)%20is,labeled%20by%20about%2040%20annotators.). We remove the RAF data from this github repo.


## Results
Please refer to graphs in output/curves.ipynb for our model results.

Please see sample below for our sample video results.
> video output
![](pic/video_output.gif)

> camera output
![](pic/camera_output.gif)
