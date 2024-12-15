from dash import Dash, dcc, html, dash_table
import base64
from io import BytesIO
from utils import *
df_s, df_l = load_data()

human_columns = ["image_name", "Male", "Young", "Smiling", "Wearing_Hat", "Big_Nose"]
DNN_columns = ["image_name", "embedding_0", "embedding_1", "embedding_2", "embedding_510", "embedding_511"]

app = Dash(__name__)


# Define the relative path to the image
image_path = 'img_celeba/011256.jpg'  # Path to your image



# Encode the image
encoded_image = encode_image(image_path)

# Global margin style
global_style = {'marginLeft': '10%', 'marginRight': '10%'}

# Define content blocks
header_block = html.Div([
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
        style={'fontSize': 14}
    ),
], style=global_style)





# Create example data table
data = {
    "Number of Stars": ["20 stars", "964 stars", "16 stars"],
    "Picture Distribution": ["have in the range of 27-29 pictures.", "have precisely 30 pictures.", "have in the range of 31-35 pictures."]
}
df_example = pd.DataFrame(data)


table_sub_block = html.Div([
    html.H3("30,012 pictures of 1,000 stars", style={"marginBottom": "20px"}),
    dcc.Markdown(
        """
        We will work with a dataset of 30,012 pictures of 1,000 different stars, not every stars has the same number of pictures:  
        """,
        style={'fontSize': 14}
    ),
    dash_table.DataTable(
        data=df_example.to_dict('records'),
        columns=[{"name": col, "id": col} for col in df_example.columns],
        style_table={'margin': '20px auto', 'width': '60%'},
        style_cell={
            'textAlign': 'center',
            'fontSize': 12,  # Smaller font size
            'padding': '5px',  # Reduced padding
        },
        style_header={'fontWeight': 'bold', 'backgroundColor': 'lightgrey'}
    )
])



image_sub_block =  html.Div([
    html.H3("Human and DNN's labels for a picture", style={"marginBottom": "20px"}),
    dcc.Markdown(
        """
        Human and DNNs do not label a picture the same, let's see picture "011256.jpg" as an example.  
        """,
        style={'fontSize': 14}
    ),
    html.Div([
        html.Figure([
            html.Img(src=encoded_image, style={'width': '50%'}),
            html.Figcaption("Image of a celebrity (011256.jpg)", style={'marginTop': '10px', 'fontSize': '14px', 'fontStyle': 'italic'})
        ], style={'textAlign': 'center', 'flex': 1, 'padding': '10px'}),

        html.Div([

            html.Div([
                dcc.Markdown(
                    """
                    1. The first set of labels is provided by a human annotator and gives information about the presence or absence of specific attributes.  
                    """,
                    mathjax=True, style={'fontSize': 14}
                ),
            ]),

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
                    2. The two other sets of labels are provided by the DNNs, and consist of 2 sets of 511 "embeddings", very difficult to interpret by humans with the naked eye.  
                    """,
                    mathjax=True, style={'fontSize': 14}
                ),
            ]),

            dash_table.DataTable(
                id='table_DNN',
                columns=[{"name": col, "id": col} for col in DNN_columns],
                data=df_s[DNN_columns].head(1).to_dict('records'),
                style_table={'overflowX': 'auto', 'marginTop': '20px'},
                style_cell={'textAlign': 'center', 'fontSize': 14},
            ),
        ], style={'flex': 1, 'padding': '10px'}),

    ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'})
])

image_and_table_block = html.Div([
    # html.H2(
    #     "Dataset for the project",
    #     style={"marginBottom": "20px"}
    # ),
    table_sub_block,
    image_sub_block
], style=global_style)



differences_block = html.Div([
    # html.H3(
    #     "Differences between human and DNN representation of a picture:",
    #     style={"marginBottom": "20px"}
    # ),
    # dcc.Markdown(
    #     """
    #     If we were to ask a human to describe a picture of someone, we would likely describe the person in the picture.  
    #     We might say that the person in the picture is a woman, that she has straight, black hair, ... we will study how this information is present in the latent space.  
    #     """,
    #     style={'fontSize': 14}
    # ),
], style=global_style)


# App layout
project_presentation_page= html.Div([
    back_to_menu_block,
    header_block,
    image_and_table_block,
    differences_block,
    back_to_menu_block,   

])

# Run the app
if __name__ == "__main__":
    # Initialize the Dash app
    app.layout =project_presentation_page
    app.run_server(debug=True)
