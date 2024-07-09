# Mini-Workshop: Introduction to Recommender Systems

This repository contains the code for the **mini-workshop introducing the world of ML and in particular Recommender Systems**. The workshop is organized by the [Munich Center for Machine Learning (MCML)](https://mcml.ai/) and aims to provide an interactive introduction to the topic. The target audience are high school students with a strong interest in computer science and machine learning. However **no prior knowledge** is required to participate in the workshop. This repository contains the code for the interactive code based part of the workshop.

If you are interested in organizing a similar workshop, feel free to use the material provided here but please make sure to give proper credit. Also, feel free to reach out to me if you have any questions or are interested in the slides or additional material.

A link to the read2use streamlit app can be found [HERE](https://emanuelsommer-recommender-workshop-1--recommender-rc8jfb.streamlit.app/).

## Local Setup

1. Create a virtual environment and activate it.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install the required packages using `poetry` (install it if you haven't already):

```bash
poetry install
```

3. Run streamlit app locally from your top level directory:

```bash
streamlit run 1_🤖_Recommender.py
```

Have fun! 🚀

## Contributing

If you have any suggestions, ideas or improvements, feel free to open an issue or create a pull request. I'd be happy if the workshop material could be improved and extended. Especially easy deployment guides and utilities for the streamlit app would be a nice addition.