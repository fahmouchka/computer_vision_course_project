import pandas as pd

import streamlit as st
from torch.utils.data import Dataset

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

train_folder = '../inputs/train/'
test_folder = '../inputs/test'


    
    

    
def main():

    zip_file_path = '..\inputs\input_data.zip'
    unzip(zip_file_path=zip_file_path)
    train = generate_csv(root_folder=train_folder,label_map=LABEL_MAP)
    
    render_eda_section(images_df=train, label_map=LABEL_MAP)
    

         
    with st.sidebar:
        mode = st.selectbox("Please Choose Exploration Mode ", ["EDA", "Inference"])
    


if __name__ == "__main__":

    main()