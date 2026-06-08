
## Code Architecture

    ├── checkpoint        # Saved models
    ├── data              # Dataset folder
    ├── models            # Model architectures
    │   ├── resnet.py     # ResNet models
    │   └── vgg.py        # VGG models
    ├── dataset.py        # Dataset processing function
    ├── main.py           # Main function
    ├── partition.py      # (Implicit) partioning function
    ├── dpso.py           # optimization algorithm
    ├── train.py          # Training function
    ├── trigger_**.py     # Trigger function for various ablation experiment
    ├── trigger_wshape_all2.py   # Trigger function 
    └── utils.py          # Utility functions
```


环境配置
依赖安装
bash pip install torch torchvision numpy opencv-python tqdm matplotlib

数据集下载
Download dataset and storage them in data folder: from torchvision import datasets

训练
After downloading the dataset run data_process.py to pre-process the train and test dataset 
python data_process.py 
run train_ours.py to train backdoor model 
python train_ours.py

防御方法

/fp/
python defense_FP.py --dataset --outfile

/nc/
python defense_NC.py --dataset

/strip/
python defense_STRIP.py --dataset

/gradcam/
python defense_gradcam.py --dataset --target_class

/ftd/
python defense_FTD.py --dataset
