import pickle
from dash import Dash, html, dcc, Input, Output, State
import plotly.express as px
from utils import *



# Load and preprocess data
with open("datasets/preprocessed_data.pkl", "rb") as f:
    data = pickle.load(f)


# data_to_save = {
#     
#     "kmeans_fig_s": kmeans_fig_s,
#     "kmeans_fig_l": kmeans_fig_l,
#     "dbscan_fig_s": dbscan_fig_s,
#     "dbscan_fig_l": dbscan_fig_l,
#     "dendrogram_image_tsne_s": dendrogram_image_tsne_s,
#     "dendrogram_image_tsne_l": dendrogram_image_tsne_l,
#     #"dendrogram_image_pca_s": dendrogram_image_pca_s,
#     #"dendrogram_image_pca_l": dendrogram_image_pca_l,
#     "tsne_39D_s": tsne_39D_s,
#     "tsne_39D_l": tsne_39D_l,
#     "tsne_2D_s": tsne_2D_s,
#     "tsne_2D_l": tsne_2D_l,
#     "kmeans_fig_s2STEP": kmeans_fig_s2STEP,
#     "kmeans_fig_l2STEP": kmeans_fig_l2STEP,
#     "dbscan_fig_s2STEP": dbscan_fig_s2STEP,
#     "dbscan_fig_l2STEP": dbscan_fig_l2STEP,

# Unpack preprocessed data
buffalo_s, buffalo_l = load_data()
buffalo_s_embed, buffalo_s_label, buffalo_l_embed, buffalo_l_label = preprocess_data(buffalo_s, buffalo_l)

# Initialize Dash app


# Layout
Bpage = html.Div([
    html.Div([
    html.A("Back to Menu", href="/", style={"margin-top": "20px", 'alignItems': 'center'})
], style={"justify-content": "center",'alignItems': 'center'}),
    html.H1("t-SNE Projection with Selected Labels"),
    html.P(
                "Here we will visualize how well the t-SNE algorithm separates the data points based on the selected labels. "
                
                "You can select one or more labels from the dropdown menu to highlight the data points that have those labels. "
        
                
        ),
    # Multi-select dropdown for labels
    html.Div([
        html.Label("Select Labels to Highlight:"),
        dcc.Dropdown(
            options=[{'label': label, 'value': label} for label in buffalo_s_label.columns],
            id='multi-label-selector',
            multi=True,  # Allow multiple selections
            value=[],    # Default empty selection
            placeholder="Select labels to highlight..."
        )
    ], style={'margin-bottom': '20px'}),

    # Row for t-SNE plots
    html.Div([
        # t-SNE scatter plot for buffalo_s
        html.Div([
            html.H2("t-SNE Projection for buffalo_s"),
            dcc.Graph(id='tsne-plot-s')
        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),

        # t-SNE scatter plot for buffalo_l
        html.Div([
            html.H2("t-SNE Projection for buffalo_l"),
            dcc.Graph(id='tsne-plot-l')
        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'})
    ]),

    # Static text explanation
    html.Div([
        html.H2("Primary Analysis"),
        html.P(
                "We can observe some clustering of the data in the t-SNE projections, highly depending of the label. "
                "We can oberve some correlation between the labels, like between Male, Mustache, Beard, 5_o_Clock_Shadow..." 
        ),
        ], style={'margin-bottom': '20px', 'padding': '10px', 'backgroundColor': '#f9f9f9', 'border': '1px solid #ddd'}),
    html.Div([
    html.A("Back to Menu", href="/", style={"margin-top": "20px", 'alignItems': 'center'})
], style={"justify-content": "center",'alignItems': 'center'}),
    
], style={'padding': '20px'})

def register_callbacksBPage(app):
    # Callback to update t-SNE plots based on selected labels
    @app.callback(
        [Output('tsne-plot-s', 'figure'),
        Output('tsne-plot-l', 'figure')],
        Input('multi-label-selector', 'value')
    )
    def update_tsne_plots(selected_labels):
        # Copy t-SNE data for plotting
        with open("datasets/preprocessed_data.pkl", "rb") as f:
            data = pickle.load(f)
        
        tsne_s = data["tsne_results_s"]  # t-SNE embedding for small dataset
        tsne_l = data["tsne_results_l"]
        
        buffalo_s, buffalo_l = load_data()
        buffalo_s_embed, buffalo_s_label, buffalo_l_embed, buffalo_l_label = preprocess_data(buffalo_s, buffalo_l)

        # Default color: -1 (not highlighted)
        tsne_s['highlighted'] = False
        tsne_l['highlighted'] = False    

        # If labels are selected, highlight matching points
        if selected_labels:
            for label in selected_labels:
                tsne_s['highlighted'] = tsne_s['highlighted'] | (buffalo_s_label[label] == 1)
                tsne_l['highlighted'] = tsne_l['highlighted'] | (buffalo_l_label[label] == 1)

        # Create scatter plots
        tsne_fig_s = px.scatter(
            tsne_s, x='x', y='y', color='highlighted',
            title="t-SNE Projection for buffalo_s",
            labels={'highlighted': 'Highlighted'},
            color_discrete_map={True: 'red', False: 'blue'}
        )
        tsne_fig_s.update_traces(marker=dict(size=6), selector=dict(name='False'))
        tsne_fig_s.update_traces(marker=dict(size=8), selector=dict(name='True'))

        tsne_fig_l = px.scatter(
            tsne_l, x='x', y='y', color='highlighted',
            title="t-SNE Projection for buffalo_l",
            labels={'highlighted': 'Highlighted'},
            color_discrete_map={True: 'red', False: 'blue'}
        )
        tsne_fig_l.update_traces(marker=dict(size=6), selector=dict(name='False'))
        tsne_fig_l.update_traces(marker=dict(size=8), selector=dict(name='True'))

        return tsne_fig_s, tsne_fig_l




