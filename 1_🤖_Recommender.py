import streamlit as st
import tensorflow as tf
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


st.title("Number Recommender 🤖")

# set seed for reproducibility
seed = st.sidebar.number_input("Seed (New random setting)", min_value=0, max_value=100000, value=0, step=1)
np.random.seed(seed)
random.seed(seed)

colors = ["#5933d6", "#a80390", "#000000"]
cmaps = {}
for color in colors:
    cmaps[color] = mpl.colors.ListedColormap([color, "#ffffff"])
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
def get_price(number, color):
    noise = random.gauss(0, 1)
    if color == "#000000":
        return (number + 1) * 1
    elif color == "#a80390":
        return (number + 1) * 2 + noise
    else:
        return (number + 1) * 2.2 + noise

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

n_choices = 6
random_indecies = random.sample(range(x_train.shape[0]), n_choices)
picture_df = pd.DataFrame({
    "index": random_indecies,
    "label": y_train[random_indecies],
})
picture_df["color"] = [random.choice(colors) for _ in range(n_choices)]
picture_df["price"] = [
    np.round(get_price(label, color) , 2)
    for label, color in zip(picture_df["label"], picture_df["color"])
]

def display_image(index, color_map="viridis"):
    import matplotlib.pyplot as plt
    plt.imshow(x_train[index], cmap=color_map)
    plt.axis('off')
    st.pyplot(plt)

ref_cols = st.columns([3, 3, 3])
with ref_cols[1]:
    st.write("Currently selected item:")
    display_image(picture_df["index"].iloc[0], color_map=cmaps[picture_df["color"].iloc[0]])
    st.write(f"Number: **{picture_df['label'].iloc[0]}**\n\nPrice: {picture_df['price'].iloc[0]}€")

st.write("Now the recommender should choose a the next best item for you from the following options:")

expander_recommend = st.sidebar.expander("Perform recommendations", expanded=False)
with expander_recommend:
    should_recommend = st.checkbox("Recommend?")
    price_weight = st.number_input("Price weight", min_value=0, max_value=1_000, value=0, step=1)
    label_weight = st.number_input("Label weight", min_value=0, max_value=1_000, value=0, step=1)
    color_weight = st.number_input("Color weight", min_value=0, max_value=1_000, value=0, step=1)
if should_recommend:
    distances = np.sqrt(
        price_weight * (picture_df["price"] - picture_df["price"].iloc[0])**2 +
        label_weight * (picture_df["label"] - picture_df["label"].iloc[0])**2 +
        color_weight * (picture_df["color"] != picture_df["color"].iloc[0]).astype(int)
    )

    picture_df["distance"] = distances
    picture_df = picture_df.sort_values("distance")

picture_df = picture_df.iloc[1:].copy()

img_cols = st.columns(n_choices-1)
for i, col in enumerate(img_cols):
    with col:
        display_image(
            picture_df["index"].iloc[i], 
            color_map=cmaps[picture_df["color"].iloc[i]]
        )
        st.write(f"Number: **{picture_df['label'].iloc[i]}**")
        st.write(f"Price: {picture_df['price'].iloc[i]}€")
        if should_recommend:
            st.write(f"Distance: {picture_df['distance'].iloc[i]:.2f}")

