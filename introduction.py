from dash import Dash, dash_table, html
import pandas as pd





df_s, df_l = load_data()








# Initialize Dash app
app = Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Comparison of Dataset Image Ranges"),
    dash_table.DataTable(
        data=comparison_table.to_dict('records'),
        columns=[{'name': col, 'id': col} for col in comparison_table.columns],
        style_table={'margin': '20px auto', 'width': '70%'},
        style_cell={'textAlign': 'center', 'fontSize': '16px'},
        style_header={'fontWeight': 'bold'}
    )
])

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)
