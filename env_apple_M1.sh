# Follow instructions to get prerequisite macbook softwares setting up: https://betterdatascience.com/install-tensorflow-2-7-on-macbook-pro-m1-pro/

conda create --name env_tf python=3.9
conda activate env_tf
conda install -c apple tensorflow-deps -y
python -m pip install tensorflow-macos
pip install tensorflow-metal