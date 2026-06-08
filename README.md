
## Code Architecture

    ├── AVEA.py                 # avea
    ├── GradCAM.py              # GradCAM可视化工具
    ├── cancha.py               # 残差
    ├── data_process.py         # 数据处理工具
    ├── dct.py                  # DCT变换核心模块
    ├── dp.py                   # dp
    ├── train_ours.py           # 模型训练
    ├── networks/               # 网络结构定义
    ├── bd/result/              # 后门攻击结果保存
    ├── cancha/                 # 残差检测结果保存
    ├── clean/                  # 干净样本保存
    ├── dct_steps/              # DCT处理步骤可视化
    ├── defense/                # 防御方法代码
    ├── .gitignore              
    ├── README.md               # 项目说明
    └── Readme.txt              # 简要说明
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
