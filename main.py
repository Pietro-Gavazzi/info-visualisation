import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from  A0project_presentation   import *
from  A1Dataset_presentation   import *

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout for the menu page
menu_layout = html.Div([
    html.H1("Welcome to the \"Better Embedding\" project"),
    html.P("..."),
    
    # Centered box for vertically stacked menu links
    html.Div([
        html.A("Project presentation", href="/page-1", style={"margin-bottom": "10px", "text-decoration": "none", "font-size": "20px"}),
        html.A("Human labbeling", href="/page-2", style={"text-decoration": "none", "font-size": "20px"}),
        html.A("Human labbeling", href="/page-2", style={"text-decoration": "none", "font-size": "20px"}),
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
    else:
        return menu_layout

# Layout for the entire app
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),  # Keeps track of the current URL
    html.Div(id='page-content')  # This will display the content of the current page
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
