# CS475_facial expression recognition

This project uses tensorflow. 

For Apple M1 and newest chip carried MacBook Pro, refer to env_apple_M1.sh

Note: If you find an error message that a certain package is not found, you will need to install the package and contact either one of us at:
<ylu106@jhu.edu>,<zsha2@jhu.edu>,<pxu11@jhu.edu>,<xzhao77@jhu.edu> if you still have troubles.

## Here are the instructions to run the Python files in src.
To run filter_bank.py use
> python src/filter_bank.py

To run models.py, use
> python src/models.py

To run data.py, use
> python src/data.py

To run facial_recog_camera.py, use
> python src/facial_recog_camera.py -i <#name of input video file> -o <#name of output file>

To run train.py, use
> python src/train.py --dataset <#name of dataset> --model <#name of model> --epoches <#number of epochs> --batch-size <#batch size>

Note that the <#name of dataset> is either combined, raf, or ck+.
Note also that <#name of model> is either simple_cnn,gabor_cnn,or vgg.