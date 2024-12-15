import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from  A0project_presentation   import *
from  A1Dataset_presentation   import *
from  A2embeddingPresentation  import *
from CPage import *
from BPage import *
from APage import *



# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Define the layout for the menu page
menu_layout = html.Div([
    html.H1("Welcome to the \"Better Embedding\" project"),
    html.P("Welcome to project \"Better embedding\", feel free to navigate into our data representations and have fun ! "),
    
    # Centered box for vertically stacked menu links
    html.Div([
        html.A("Project presentation", href="/page-1", style={"margin-bottom": "10px", "text-decoration": "none", "font-size": "20px"}),
        html.A("Human labbeling", href="/page-2", style={"text-decoration": "none", "font-size": "20px"}),
        html.A("DNN labbeling", href="/page-3", style={"text-decoration": "none", "font-size": "20px"}),
        html.A("First Visualization", href="/page-4", style={"text-decoration": "none", "font-size": "20px"}),
        html.A("Label Visualization", href="/page-5", style={"text-decoration": "none", "font-size": "20px"}),
        html.A("Clustering", href="/page-6", style={"text-decoration": "none", "font-size": "20px"}),
    ], style={
        "display": "flex",
        "flex-direction": "column",  # Stacks the links vertically
        "justify-content": "center",
        "align-items": "center",
        "border": "2px solid #ccc",
        "padding": "20px",
        "margin-top": "30px",
        "border-radius": "8px",
        "background-color": "#f9f9f9",
        "width": "200px"  # Optional: Control the width of the box
    })
], style={
    "display": "flex",
    "flex-direction": "column",  # Stack the elements vertically
    "align-items": "center",  # Center content horizontally
    "height": "100vh",  # Full height of the viewport
    "text-align": "center"  # Center text inside the elements
})



# Define the callback to switch between pages
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/page-1':
        return project_presentation_page
    elif pathname == '/page-2':
        return dataset_presentation_page
    elif pathname == '/page-3':
        return embedding_presentation
    elif pathname == '/page-4':
        return APage
    elif pathname == '/page-5':
        return Bpage
    elif pathname == '/page-6':
        return Cpage
    else:
        return menu_layout

# Layout for the entire app
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),  # Keeps track of the current URL
    html.Div(id='page-content')  # This will display the content of the current page
])
# Register callbacks for each page
register_callbacksAPage(app)
register_callbacksBPage(app)
register_callbacksCPage(app)
register_callbacksA2Page(app)
register_callbacksA11Page(app)
register_callbacksA12Page(app)


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
