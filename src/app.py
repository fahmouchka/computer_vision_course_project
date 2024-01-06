import streamlit as st

from inference import render_inference_section
from eda import render_eda_section
from utils import generate_csv, unzip

LABEL_MAP = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprise': 6
}

TRAIN_FOLDER = '../inputs/train/'
TEST_FOLDER = '../inputs/test/'
ZIP_FILE_PATHS = ['../inputs/input_data.zip','../inputs/resnet50_fold0_best.pth.zip']

def main():
    for zip_file_path in ZIP_FILE_PATHS:
        unzip(zip_file_path=zip_file_path)
    train = generate_csv(root_folder=TRAIN_FOLDER,label_map=LABEL_MAP)
    
    with st.sidebar:
        mode = st.selectbox("Please Choose Exploration Mode ", ["EDA", "Inference"])
    if mode == "EDA": 
        train = generate_csv(root_folder=TRAIN_FOLDER,label_map=LABEL_MAP)
        render_eda_section(images_df=train, label_map=LABEL_MAP)
    elif mode == "Inference":
        render_inference_section(label_map=LABEL_MAP)
            

if __name__ == "__main__":
    main()