from dash import Dash, dcc, dash_table, html
import pandas as pd
from utils import *
import numpy as np
import plotly.express as px
import pickle
from PIL import Image
import base64
from io import BytesIO

# Initialize the Dash app
app = Dash(__name__)

# Define the relative path to the image
image_path = 'img_celeba/000211.jpg'  # Path to your image

# Function to convert image to base64 string
def encode_image(image_path):
    # Open image using Pillow
    img = Image.open(image_path)
    
    # Convert image to a byte stream
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    
    # Encode the byte stream to base64
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return f"data:image/jpeg;base64,{img_str}"

# Encode the image
encoded_image = encode_image(image_path)

# App layout
app.layout = html.Div([
    html.H1(
        "Presentation Project \"Better Embedding\" ",
        style={"textAlign": "center", "marginBottom": "20px"}
    ),
    html.P(
                "This project uses two types of labeling for facial images: ",
                style={'fontSize': 16}
            ),
    # Use a flexbox layout to place the image on the left and explanation on the right
    html.Div([
        # Image on the left
        html.Div([
            html.Img(src=encoded_image, style={'width': '50%'})
        ], style={'flex': 1, 'padding': '10px'}),
        
        # Explanation on the right
        html.Div([
            html.P(
                "1. Human labeling: Each image is labeled by a human annotator who provides "
                "information about the presence or absence of specific facial attributes, such as 'Smiling', 'Bald', etc.",
                style={'fontSize': 16}
            ),
            html.P(
                "2. Automated labeling: A machine learning model is used to predict these labels, based on the "
                "features extracted from the image.",
                style={'fontSize': 16}
            ),
        ], style={'flex': 1, 'padding': '10px'})
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'})
])

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
