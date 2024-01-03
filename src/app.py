import streamlit as st

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
ZIP_FILE_PATH = '../inputs/input_data.zip'

def main():
    unzip(zip_file_path=ZIP_FILE_PATH)
    train = generate_csv(root_folder=TRAIN_FOLDER,label_map=LABEL_MAP)
    render_eda_section(images_df=train, label_map=LABEL_MAP)
    
    with st.sidebar:
        mode = st.selectbox("Please Choose Exploration Mode ", ["EDA", "Inference"])

if __name__ == "__main__":
    main()