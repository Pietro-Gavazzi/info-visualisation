import pickle
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
from utils import *

import pickle
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
from PIL import Image
import base64
from io import BytesIO



# Load precomputed data
with open("datasets/preprocessed_data.pkl", "rb") as f:
    data = pickle.load(f)

# Unpack data
buffalo_s = data["buffalo_s"]
buffalo_l = data["buffalo_l"]
tsne_results_s = data["tsne_results_s"]
tsne_results_l = data["tsne_results_l"]
#pca_results_s = data["pca_results_s"]
#pca_results_l = data["pca_results_l"]
kmeans_fig_s = data["kmeans_fig_s"]
kmeans_fig_l = data["kmeans_fig_l"]

dbscan_fig_s = data["dbscan_fig_s"]
dbscan_fig_l = data["dbscan_fig_l"]

dendrogram_image_tsne_s = data["dendrogram_image_tsne_s"]
dendrogram_image_tsne_l = data["dendrogram_image_tsne_l"]
#dendrogram_image_pca_s = data["dendrogram_image_pca_s"]
#dendrogram_image_pca_l = data["dendrogram_image_pca_l"]

# Initialize Dash app


# Layout
APage = html.Div([
    
    html.Div([
    html.A("Back to Menu", href="/", style={"margin-top": "20px", 'alignItems': 'center'})
], style={"justify-content": "center",'alignItems': 'center'}),
    
    html.H1("Buffalo Dataset Analysis"),

    
    # Static text explanation
    html.Div([
        html.H2("T-SNE Visualization"),
        html.P(
                "In the following visualizations, we use T-SNE to project the buffalo dataset into a 2D space. "
                "TSNE is a dimensionality reduction technique that is particularly useful for visualizing high-dimensional data. "
                "The most important property of TSNE is the concevation of local structure, which means that similar data points in the high-dimensional space will be close to each other in the 2D projection. "
                "The most important parameter for TSNE is the perplexity value, which determines the number of nearest neighbors used in the algorithm. "
        ),
        html.P(
            "The left panel displays projections for the smaller subset of the data (buffalo_s), "
            "while the right panel shows projections for the larger subset (buffalo_l). "
            "These visualizations help identify patterns and clusters in the dataset but also allow for a comparison between the two subsets."
            "We could expect to see similar patterns in both subsets, but the larger subset may have a more 'clusterised' structure."
        ),
        html.P(
            "No selections are required to start; the default settings will display "
            "the visualizations using a perplexity value of 30."
            "You can select an ID from the dropdown menu to highlight a specific data point in the projections."
            "You can also change the perplexity value to see how the projections change."
        )
        ], style={'margin-bottom': '20px', 'padding': '10px', 'backgroundColor': '#f9f9f9', 'border': '1px solid #ddd'}),

    # Dropdown to select an ID for highlighting
    html.Div([
        html.Label("Select ID to Highlight:"),
        dcc.Dropdown(
            options=[{'label': f'ID {i}', 'value': i} for i in tsne_results_s['id'].unique()],
            id='id-selector',
            value=None,
            clearable=True
        )
    ]),
    # Dropdown to select a perplexity value for t-SNE
    html.Div([
        html.Label("Select Perplexity Value for T-SNE:"),
        dcc.Dropdown(
            options=[3,30,60,1000],
            id='perplexity-selector',
            value=30,
            clearable=False
        )
    ]),

    

    # Projections and visualizations
    html.Div([
        html.Div([
            html.H3("Projection of buffalo_s"),
            dcc.Graph(id='projection-plot-s')
        ], style={'display': 'inline-block', 'width': '49%'}),
        html.Div([
            html.H3("Projection of buffalo_l"),
            dcc.Graph(id='projection-plot-l')
        ], style={'display': 'inline-block', 'width': '49%'})
    ]),
    
# Static text explanation
    html.Div([
        html.H2("Primary Analysis"),
        html.P(
                "We can observe a good clustering of the data in the T-SNE projections. "
                "Those cluster reprensent the different persons, by their id."
                "but we can somethimes see appear some outliers like with ID 15 and 47 indicating that the model is not perfect or wrongly labelled images."
        ),
        ], style={'margin-bottom': '20px', 'padding': '10px', 'backgroundColor': '#f9f9f9', 'border': '1px solid #ddd'}),

    html.Div([
    html.A("Back to Menu", href="/", style={"margin-top": "20px", 'alignItems': 'center'})
], style={"justify-content": "center",'alignItems': 'center'}),
])

# Callbacks for projections
def register_callbacksAPage(app):
    @app.callback(
        [Output('projection-plot-s', 'figure'),
        Output('projection-plot-l', 'figure')],
        [Input('id-selector', 'value'),
        Input('perplexity-selector', 'value')]
    )
    def update_projection_plots(selected_id,selected_perplexity):
        
        
        # Highlight selected ID
        
        with open("datasets/preprocessed_data.pkl", "rb") as f:
            data = pickle.load(f)

        # Unpack data
        
        tsne_results_s = data["tsne_results_s"]
        tsne_results_l = data["tsne_results_l"]
        tsne_results_s60 = data["tsne_results_s60"]
        tsne_results_l60 = data["tsne_results_l60"]
        tsne_results_s3 = data["tsne_results_s3"]
        tsne_results_l3 = data["tsne_results_l3"]
        tsne_results_s1000 = data["tsne_results_s1000"]
        tsne_results_l1000 = data["tsne_results_l1000"]
        
        buffalo_l, buffalo_s = data["buffalo_l"], data["buffalo_s"]
        buffalo_s_embed, buffalo_s_label, buffalo_l_embed, buffalo_l_label = preprocess_data(buffalo_s, buffalo_l)
        tsne_results_s60['id'] = buffalo_s_label['id']
        tsne_results_l60['id'] = buffalo_l_label['id']
        tsne_results_s3['id'] = buffalo_s_label['id']
        tsne_results_l3['id'] = buffalo_l_label['id']
        tsne_results_s1000['id'] = buffalo_s_label['id']
        tsne_results_l1000['id'] = buffalo_l_label['id']
        
        if selected_perplexity == 3:
            tsne_s = tsne_results_s3.copy()
            tsne_l = tsne_results_l3.copy()
        elif selected_perplexity == 30:
            tsne_s = tsne_results_s.copy()
            tsne_l = tsne_results_l.copy()
        elif selected_perplexity == 60:
            tsne_s = tsne_results_s60.copy()
            tsne_l = tsne_results_l60.copy()
        elif selected_perplexity == 1000:
            tsne_s = tsne_results_s1000.copy()
            tsne_l = tsne_results_l1000.copy()
            
        
        tsne_s['color'] = tsne_s['id'].apply(lambda i: 'red' if str(i) == str(selected_id) else 'blue')
        tsne_l['color'] = tsne_l['id'].apply(lambda i: 'red' if str(i) == str(selected_id) else 'blue')

        # Generate projection figures
        proj_fig_s = px.scatter(
        tsne_s, x='x', y='y', color='color',
        title="Projection of buffalo_s",
        color_discrete_map={'red': 'red', 'blue': 'blue'},
        hover_data=['id']
        )
        proj_fig_l = px.scatter(
        tsne_l, x='x', y='y', color='color',
        title="Projection of buffalo_l",
        color_discrete_map={'red': 'red', 'blue': 'blue'},
        hover_data=['id']
        )

        return proj_fig_s, proj_fig_l




