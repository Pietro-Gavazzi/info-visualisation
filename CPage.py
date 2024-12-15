import pickle
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
from utils import *

# Load preprocessed data
with open("datasets/preprocessed_data.pkl", "rb") as f:
    data = pickle.load(f)

# Unpack preprocessed data
buffalo_s, buffalo_l = load_data()
buffalo_s_embed, buffalo_s_label, buffalo_l_embed, buffalo_l_label = preprocess_data(buffalo_s, buffalo_l)

tsne_results_s = data["tsne_2D_s"]  # t-SNE embedding for small dataset
tsne_results_l = data["tsne_2D_l"]  # t-SNE embedding for large dataset

# Initialize Dash app
app = Dash(__name__)

# Layout with tabs for multiple pages
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label="t-SNE Visualization", children=[
            html.H1("t-SNE Projection with Selected Labels"),

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
            ]),

            # t-SNE scatter plot for buffalo_s and buffalo_l side by side
            html.Div([
                html.Div([
                    html.H2("t-SNE Projection for buffalo_s"),
                    dcc.Graph(id='tsne-plot-s')
                ], style={'width': '48%', 'display': 'inline-block'}),

                html.Div([
                    html.H2("t-SNE Projection for buffalo_l"),
                    dcc.Graph(id='tsne-plot-l')
                ], style={'width': '48%', 'display': 'inline-block'})
            ])
        ]),

        dcc.Tab(label="Clustering Multi-Step", children=[
            html.H1("Clustering Multi-Step Results"),

            # KMeans multi-step results
            html.Div([
                html.H2("KMeans Multi-Step Results"),
                html.Div([
                    html.Div([
                        html.H3("KMeans Step 1 (buffalo_s)"),
                        dcc.Graph(figure=data["kmeans_fig_s"])
                    ], style={'width': '48%', 'display': 'inline-block'}),

                    html.Div([
                        html.H3("KMeans Step 1 (buffalo_l)"),
                        dcc.Graph(figure=data["kmeans_fig_l"])
                    ], style={'width': '48%', 'display': 'inline-block'})
                ]),

                html.Div([
                    html.Div([
                        html.H3("KMeans Step 2 (buffalo_s)"),
                        dcc.Graph(figure=data["kmeans_fig_s2STEP"])
                    ], style={'width': '48%', 'display': 'inline-block'}),

                    html.Div([
                        html.H3("KMeans Step 2 (buffalo_l)"),
                        dcc.Graph(figure=data["kmeans_fig_l2STEP"])
                    ], style={'width': '48%', 'display': 'inline-block'})
                ])
            ]),

            # DBSCAN multi-step results
            html.Div([
                html.H2("DBSCAN Multi-Step Results"),
                html.Div([
                    html.Div([
                        html.H3("DBSCAN Step 1 (buffalo_s)"),
                        dcc.Graph(figure=data["dbscan_fig_s"])
                    ], style={'width': '48%', 'display': 'inline-block'}),

                    html.Div([
                        html.H3("DBSCAN Step 1 (buffalo_l)"),
                        dcc.Graph(figure=data["dbscan_fig_l"])
                    ], style={'width': '48%', 'display': 'inline-block'})
                ]),

                html.Div([
                    html.Div([
                        html.H3("DBSCAN Step 2 (buffalo_s)"),
                        dcc.Graph(figure=data["dbscan_fig_s2STEP"])
                    ], style={'width': '48%', 'display': 'inline-block'}),

                    html.Div([
                        html.H3("DBSCAN Step 2 (buffalo_l)"),
                        dcc.Graph(figure=data["dbscan_fig_l2STEP"])
                    ], style={'width': '48%', 'display': 'inline-block'})
                ])
            ])
        ]),

        dcc.Tab(label="Text Input", children=[
            html.H1("Add Notes or Observations"),
            html.Div([
                dcc.Textarea(
                    id='text-input-area',
                    placeholder="Write your notes here...",
                    style={'width': '100%', 'height': '300px'}
                ),
                html.Button('Submit', id='submit-button', n_clicks=0),
                html.Div(id='output-text')
            ])
        ])
    ])
])

# Callbacks for t-SNE plots
@app.callback(
    [Output('tsne-plot-s', 'figure'),
     Output('tsne-plot-l', 'figure')],
    Input('multi-label-selector', 'value')
)
def update_tsne_plots(selected_labels):
    # Copy t-SNE data for plotting
    tsne_s = tsne_results_s.copy()
    tsne_l = tsne_results_l.copy()

    # Default color: Not highlighted (blue)
    tsne_s['highlighted'] = False
    tsne_l['highlighted'] = False

    if selected_labels:
        # Highlight points matching any selected labels
        for label in selected_labels:
            tsne_s['highlighted'] = tsne_s['highlighted'] | (buffalo_s_label[label] == 1)
            tsne_l['highlighted'] = tsne_l['highlighted'] | (buffalo_l_label[label] == 1)

    # Create figures with updated highlights
    tsne_fig_s = px.scatter(
        tsne_s, x='x', y='y', color='highlighted',
        title="t-SNE Projection for buffalo_s",
        color_discrete_map={True: 'red', False: 'blue'}
    )

    tsne_fig_l = px.scatter(
        tsne_l, x='x', y='y', color='highlighted',
        title="t-SNE Projection for buffalo_l",
        color_discrete_map={True: 'red', False: 'blue'}
    )

    return tsne_fig_s, tsne_fig_l

# Callback for Text Input
@app.callback(
    Output('output-text', 'children'),
    Input('submit-button', 'n_clicks'),
    Input('text-input-area', 'value')
)
def update_text_output(n_clicks, value):
    if n_clicks > 0 and value:
        return f"Your notes: {value}"
    return ""

if __name__ == '__main__':
    app.run_server(debug=True)
