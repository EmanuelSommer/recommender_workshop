import streamlit as st
import tensorflow as tf
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
pd.options.mode.chained_assignment = None

N_IMGS = 6

# init session state
if 'feedback_df' not in st.session_state:
    st.session_state['feedback_df'] = pd.DataFrame({
        "price_delta": [1],
        "label_delta": [1],
        "color_delta": [1],
        "closest": [1],
    })

if "indices" not in st.session_state:
    st.session_state["indices"] = range(N_IMGS)


st.title("Fashion Recommender 🤖")

# set seed for reproducibility
new_setting_button = st.sidebar.button("New random setting", key="new_setting")
if new_setting_button:
    st.session_state["indices"] = [i + N_IMGS for i in st.session_state["indices"]]
np.random.seed(st.session_state["indices"][0])
random.seed(st.session_state["indices"][0])

colors = ["#5933d6", "#a80390", "#000000"]
cmaps = {}
for color in colors:
    cmaps[color] = mpl.colors.ListedColormap(["#ffffff", color])
itemtype = ["t-shirt", "trouser", "sneaker"]
itemtype_num_orig = [0, 1, 7]
itemtype_num = [1, 2, 2.5]
def get_price(itemtype_num, color):
    noise = random.gauss(0, 0.5)
    if color == "#000000":
        return (itemtype_num + 1) * 10
    elif color == "#a80390":
        return (itemtype_num + 1) * 20 + noise
    else:
        return (itemtype_num + 1) * 22 + noise
def number_to_itemtype(input_num):
    for i, num in enumerate(itemtype_num):
        if input_num == num:
            return itemtype[i]
def number_to_itemtype_num(input_num):
    for i, num in enumerate(itemtype_num_orig):
        if input_num == num:
            return itemtype_num[i]

# Load Fashion MNIST dataset
(x_train, y_train), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train[(y_train == 0) | (y_train == 1) | (y_train == 7)]
y_train = y_train[(y_train == 0) | (y_train == 1) | (y_train == 7)]
y_train = np.array([number_to_itemtype_num(y) for y in y_train])

picture_df = pd.DataFrame({
    "index": st.session_state["indices"],
    "label": y_train[st.session_state["indices"]],
})

picture_df["color"] = [random.choice(colors) for _ in range(N_IMGS)]
picture_df["price"] = [
    np.round(get_price(label, color) , 2)
    for label, color in zip(picture_df["label"], picture_df["color"])
]
picture_df["label_str"] = [number_to_itemtype(num) for num in picture_df["label"]]
picture_df["price_delta"] = (picture_df["price"] - picture_df["price"].iloc[0])**2
picture_df["label_delta"] = (picture_df["label"] - picture_df["label"].iloc[0])**2
picture_df["color_delta"] = (picture_df["color"] != picture_df["color"].iloc[0]).astype(int)

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
    price_weight = st.number_input("Price weight", min_value=-1_000.0, max_value=1_000.0, value=0.0, step=0.1)
    label_weight = st.number_input("Item weight", min_value=-1_000.0, max_value=1_000.0, value=0.0, step=0.1)
    color_weight = st.number_input("Color weight", min_value=-1_000.0, max_value=1_000.0, value=0.0, step=0.1)
if should_recommend:
    distances = (
        price_weight * picture_df["price_delta"] +
        label_weight * picture_df["label_delta"] +
        color_weight * picture_df["color_delta"]
    )

    picture_df["distance"] = distances
    picture_df = picture_df.sort_values("distance")

expander_feedback = st.sidebar.expander("Record feedback", expanded=False)
with expander_feedback:
    record_feedback = st.checkbox("Record feedback?")
    submit_feedback = st.button("Add feedback")

picture_df = picture_df.iloc[1:].copy()
picture_df["distance_feedback"] = 0

img_cols = st.columns(N_IMGS-1)
for i, col in enumerate(img_cols):
    with col:
        display_image(
            picture_df["index"].iloc[i], 
            color_map=cmaps[picture_df["color"].iloc[i]]
        )
        st.write(f"Item: **{picture_df['label_str'].iloc[i]}**")
        st.write(f"Price: {picture_df['price'].iloc[i]}€")
        if should_recommend:
            st.write(f"Distance: {picture_df['distance'].iloc[i]:.2f}")
        if record_feedback:
            picture_df["distance_feedback"].loc[i] = st.number_input("Distance", min_value=0, max_value=1_000, value=0, step=1, key=i)

if submit_feedback:
    update_df = picture_df.copy()
    # bad encoding: closest is 0 if it is the closest, 1 otherwise
    update_df["closest"] = 1
    update_df.loc[update_df["distance_feedback"].idxmin(), "closest"] = 0
    st.session_state['feedback_df'] = pd.concat(
        [st.session_state['feedback_df'], update_df[["price_delta", "label_delta", "color_delta", "closest"]]],
    )

with st.expander("Show feedback", expanded=False):
    st.write(st.session_state['feedback_df'])

with st.expander("ML recommendation for the weights", expanded=False):
    st.info("This is a simple logistic regression model that predicts which is the closest image to the user's preference. The input features are the differences in price, label, and color between the images. The more good examples you provide the better the model.")
    fit_df = st.session_state['feedback_df'].copy()
    # balance the dataset such that 50/50 are closest and not closest
    n_closest = fit_df["closest"].sum()
    n_not_closest = fit_df.shape[0] - n_closest
    if n_closest > n_not_closest:
        fit_df = pd.concat([
            fit_df[fit_df["closest"] == 1].sample(n=n_not_closest, random_state=st.session_state["indices"][0]),
            fit_df[fit_df["closest"] == 0]
        ])
    elif n_closest < n_not_closest:
        fit_df = pd.concat([
            fit_df[fit_df["closest"] == 1],
            fit_df[fit_df["closest"] == 0].sample(n=n_closest, random_state=st.session_state["indices"][0])
        ])
    if fit_df.shape[0] > 1:
        X = fit_df[["price_delta", "label_delta", "color_delta"]]
        y = fit_df["closest"]
        clf = LogisticRegression(random_state=st.session_state["indices"][0]).fit(X, y)
        # display the coefficients in a pretty way coefficient by coefficient
        for i, coef in enumerate(clf.coef_[0]):
            st.write(f'{X.columns[i].replace("_delta", "").title()} Weight: {coef:.2f}')
