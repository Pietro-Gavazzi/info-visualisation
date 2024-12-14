from dash import Dash, html, dcc, dash_table
from PIL import Image
import base64
from io import BytesIO
from utils import *
df_s, df_l = load_data()

human_columns = ["image_name", "Male", "Young", "Smiling", "Wearing_Hat", "Big_Nose"]
DNN_columns = ["image_name", "embedding_0", "embedding_1", "embedding_2", "embedding_510", "embedding_511"]

# Initialize the Dash app
app = Dash(__name__)

# Define the relative path to the image
image_path = 'img_celeba/011256.jpg'  # Path to your image

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

# Global margin style
global_style = {'marginLeft': '10%', 'marginRight': '10%'}

# App layout
app.layout = html.Div([
    # html.H1(
    #     "Presentation Project \"Better Embedding\"",
    #     style={"marginBottom": "20px"}
    # ),
    html.Div([
        html.H2(
            "Aim of the project:",
            style={"marginBottom": "20px"}
        ),
        dcc.Markdown(
            """
            Deep Neural Networks (DNN) have recently demonstrated an impressive capacity to extract usable information from images.  
            These models transform the input data (i.e., an image) into a latent representation (or embedding), which is an abstract representation of the image.  
            We have empirically observed that latent representations produced by large models seem to perform better in various tasks than latent representations produced by small models, 
            pointing to a difference in the amount of information present in the former.  

            The aim of this project is thus to study the differences in the amount of information present in a latent representation of a dataset produced by a small model 
            and the one produced by a large model.
            """,
            style={'fontSize': 16}
        ),
    ], style=global_style),  # Apply global margins here
    
    # Use a flexbox layout to place the image on the left and explanation on the right
    html.Div([
        html.H2(
            "Differences between human and DNN representation of a picture:",
            style={"marginBottom": "20px"}
        ),
        dcc.Markdown(
            """
            If we were to ask a human to describe a picture of someone, we would likely describe the person in the picture.  
            We might say that the person in the picture is a woman, that she has straight, black hair, ... we want to know if this information is present in the latent space.  
            """,
            style={'fontSize': 16}
        ),

        html.H3(
            "Example of labels for an image",
            style={"marginBottom": "20px"}
        ),     

        dcc.Markdown(
            """
            The dataset we will work with is a dataset of 30,012 pictures of 1,000 different stars:
            """,
            style={'fontSize': 16}
        ),

        # Image on the left with a label
        html.Div([
            # Image on the left with a label using html.Figcaption
            html.Figure([
                html.Img(src=encoded_image, style={'width': '50%'}),
                html.Figcaption("Image of a celebrity (011256.jpg)", style={'marginTop': '10px', 'fontSize': '14px', 'fontStyle': 'italic'})
            ], style={'textAlign': 'center', 'flex': 1, 'padding': '10px'}),

            # Explanation and table on the right
            html.Div([
                html.Div([
                    dcc.Markdown(
                        """
                        1. Each image is labeled by a human annotator who provides 
                        information about the presence or absence of specific attribute.  
                        """,
                        mathjax=True, style={'fontSize': 16}
                    ),
                ]),

                # Display table for labels_columns
                dash_table.DataTable(
                    id='table_human',
                    columns=[{"name": col, "id": col} for col in human_columns],
                    data=df_s[human_columns].head(1).to_dict('records'),
                    style_table={'overflowX': 'auto', 'marginTop': '20px'},
                    style_cell={'textAlign': 'center', 'fontSize': 14},
                ),
                
                html.Div([
                    dcc.Markdown(
                        """
                        2. Each image is labeled by the DNN's with 511 embeddings very difficult to interpret by humans with the naked eye.  
                        """,
                        mathjax=True, style={'fontSize': 16}
                    ),
                ]),

                # Display table for labels_columns
                dash_table.DataTable(
                    id='table_DNN',
                    columns=[{"name": col, "id": col} for col in DNN_columns],
                    data=df_s[DNN_columns].head(1).to_dict('records'),
                    style_table={'overflowX': 'auto', 'marginTop': '20px'},
                    style_cell={'textAlign': 'center', 'fontSize': 14},
                ),
            ], style={'flex': 1, 'padding': '10px'}),

        ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'})
    ], style=global_style)  # Apply global margins here
])

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
