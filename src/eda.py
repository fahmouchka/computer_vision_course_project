import pandas as pd
import random
import matplotlib.pyplot as plt

import streamlit as st
from torch.utils.data import Dataset

from utils import Dataset


def plot_bar_chart(data_dict: dict) -> None:
    """A function that generates a barplot that correspond to the count of each label in the dataset.

    Args:
        data_dict (dict): A dictionary that contains the labels and their corresponding count.
    """
    labels = list(data_dict.keys())
    values = list(data_dict.values())
    
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_xlabel('Emotions')
    ax.set_ylabel('Count')
    ax.set_title('Emotion Distribution')
    st.pyplot(fig)
    
def render_target_statistics(images_df: pd.DataFrame, label_map: dict) -> None:
    """A function that renders target statistics.

    Args:
        images_df (pd.DataFrame): A dataframe that contains 2 features: image_path and the integer label of the image.
        label_map (dict): A mapping between the original labels and the integer ones.
    """
    count_dict = dict(images_df["label"].value_counts())
    counts = {list(label_map.keys())[list(label_map.values()).index(label)]: count_dict[label] for label in count_dict.keys()}
    plot_bar_chart(counts)
    
def render_images(train: pd.DataFrame, label_map: dict) -> None:
    """A function that renders 4 random images in the dataset.

    Args:
        images_df (pd.DataFrame): A dataframe that contains 2 features: image_path and the integer label of the image.
        label_map (dict): A mapping between the original labels and the integer ones.
    """
    
    train_dataset = Dataset(train, transform=None)

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    for col in [row1_col1, row1_col2, row2_col1, row2_col2]:
        random_index = random.randint(0, len(train) - 1)
        image, label = train_dataset[random_index]
        decoded_label = list(label_map.keys())[list(label_map.values()).index(label)]
        col.image(image, caption=f"Label: {decoded_label}", use_column_width=True)

    
    
def render_eda_section(images_df: pd.DataFrame, label_map: dict) -> None:
    """A function that renders simple plots that are used to conduct Exploratory Data Analysis. 

    Args:
        images_df (pd.DataFrame): A dataframe that contains 2 features: image_path and the integer label of the image.
        label_map (dict): A mapping between the original labels and the integer ones.
    """
    st.title("Exploratory Data Analysis")
    st.header("Target Statistics")
    render_target_statistics(images_df=images_df, label_map=label_map)
    st.info("The above target statistics show that there is an imbalance between the labels of the dataset, we will need to take care of that.")
    st.header("Let's Explore The Dataset")
    st.info("Click the below button if you wish to plot 4 random images in the train dataset.")
    st.button("Render 4 Random Images")
    render_images(train=images_df, label_map=label_map)
    
    
    