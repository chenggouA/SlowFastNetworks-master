# SlowFastNetworks
pytorch 实现 ["SlowFast Networks for Video Recognition"](https://arxiv.org/abs/1812.03982).
## Train
1. 数据集格式应如下  
```
dataset
│    │ train/training  
│    │    │ ApplyEyeMakeup  
│    │    │ ApplyLipstick  
│    │    │ ...  
│    │ validation  
     │    │ ApplyEyeMakeup  
     │    │ ApplyLipstick  
     │    │ ...   
```


## Requirements
python 3  
PyTorch >= 0.4.1  
tensorboardX  
OpenCV  

## 一个打架检测的模型文件如下
链接: https://pan.baidu.com/s/11-6L9bjfApSUy-Ul4ejOFA 提取码: Ash1 
--来自百度网盘超级会员v4的分享

把下载的文件放到 `fightDetection\2023-12-29-11-39-14`
下

更换`predict.py` 中的视频路径,
运行`predict.py` 
