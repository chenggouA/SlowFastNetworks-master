import streamlit as st 
from config import params
import torch
from lib import slowfastnet
from lib.dataset import Video_handle
file = st.file_uploader("选择视频文件")

@st.cache_resource
def get_model(model_path):
    model = slowfastnet.resnet101(class_num=params['num_classes'])
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    return model

model = get_model("fightDetection/2023-12-29-11-39-14/clip_len_64frame_sample_rate_1_checkpoint_39.pth.tar")

def predict(video_path: str, model, device):
    import os
    assert os.path.exists(video_path), "文件不存在"
    model.to(device)
    model.eval()
    VH = Video_handle(video_path, clip_len=params['clip_len'], frame_sample_rate=params['frame_sample_rate'])

    inputs = VH.get_values()
    inputs = torch.tensor(inputs).unsqueeze(0) # 添加一维
    inputs = inputs.to(device)
    outputs = model(inputs)
    _, indices = torch.max(outputs, 1)

    result = "fight" if indices.item() == 0 else "nofight"
    # print(model)
    return result

import subprocess

save_path = "upload/1.avi"
if file:
    with open(save_path, 'wb') as f:
        f.write(file.getvalue())


    command = f"ffmpeg -i {save_path} upload/1.mp4"
    subprocess.call(command, shell=True)
    st.write("处理结果: ")

    res = predict("upload/1.mp4", model, "cpu")
    st.write(res)
    st.video("upload/1.mp4")



    

