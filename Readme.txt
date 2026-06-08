环境配置
bash
pip install torch torchvision numpy opencv-python tqdm

数据集下载
Download dataset and storage them in data folder:
from torchvision import datasets

训练
After downloading the dataset
run data_process.py to pre-process the train and test dataset
python data_process.py
run train_ours.py to train backdoor model
python train_ours.py

防御方法
# 运行Fine-Pruning防御
python defense_FP.py --dataset <datasetName>  --outfile <outputFile>
# 运行Neural Cleanse防御
python defense_NC.py --dataset <datasetName> 
# 运行STRIP防御
python defense_STRIP.py --dataset <datasetName> 
# 运行GradCAM检测
python defense_gradcam.py --dataset <datasetName> --target_class <classId>
# 运行FTD防御
python defense_FTD.py --dataset <datasetName>

