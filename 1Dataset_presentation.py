from dash import Dash, dcc, dash_table, html
import pandas as pd
from utils import load_data, preprocess_data, labels_columns
import numpy as np
import plotly.express as px
import pickle

# Load dictionary from the file
with open('inconsistent_ids_and_labels.pkl', 'rb') as file:
    inconsistent_ids_and_labels = pickle.load(file)

# Calculate the percentage of inconsistent IDs for each label
inconsistent_percentages = [(len(inconsistent_ids_and_labels[label]) / 1000) * 100 for label in labels_columns]

# Create a bar plot using Plotly Express
fig2 = px.bar(
    x=labels_columns,
    y=inconsistent_percentages,
    labels={"x": "Labels", "y": "Percentage of Inconsistent IDs (%)"},
    title="Percentage of Inconsistent IDs and Labels for Each Label",
)
fig2.update_layout(
    xaxis=dict(title="Labels", tickangle=45),
    yaxis=dict(title="Percentage of Inconsistent IDs (%)"),
    margin=dict(l=40, r=40, t=40, b=120),
    height=600,
)
fig2.update_traces(marker_color="lightcoral", marker_line_color="black", marker_line_width=1)



df_s, df_l= load_data()

# Calculate the percentage of people with value = 1 for each label
percentages = (np.sum(df_s[labels_columns] == 1, axis=0) / len(df_s)) * 100

# Create a bar plot using Plotly Express
fig = px.bar(
    x=percentages.index,
    y=percentages.values,
    labels={"x": "Labels", "y": "Percentage (%)"},
    # title="Percentage of pictures with label",
)
fig.update_layout(
    xaxis=dict(title="Labels", tickangle=45),
    yaxis=dict(title="Percentage (%)"),
    margin=dict(l=40, r=40, t=40, b=120),
    height=600,
    title=dict(
        text="Percentage of pictures with label:",
        # x=0.5,  # Center the title
        # y=0,    # Position the title beneath the graph
        # yanchor="top"
    )
)
fig.update_traces(marker_color="skyblue", marker_line_color="black", marker_line_width=1)





# Initialize the Dash app
app = Dash(__name__)

# Data for the table
data = {
    "Number of stars:": [20, 964, 16],
    "Having that many pictures:": ["in the range of 27-29", "precisely 30", "in the range of 31-35"],
}
df = pd.DataFrame(data)






# App layout
app.layout = html.Div([
    html.H1(
        "Our labeled pictures dataset",
        style={"textAlign": "center", "marginBottom": "20px"}  # Center the title
    ),
    html.H2(
        "to answer if a large embedding's model implies better label comprehension than a small embedding's model:",
        style={"textAlign": "center", "marginBottom": "20px", "color": "gray"} 
    ),

    html.H3(
        "> 30.012 pictures of 1000 stars",
        style={"textAlign": "center", "marginBottom": "20px", "color": "gray"}  # Center the subtitle with styling
    ),
    dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{"name": col, "id": col} for col in df.columns],
        style_table={'margin': '20px auto', 'width': '50%'},
        style_cell={'textAlign': 'center', 'fontSize': 16, 'padding': '10px'},
        style_header={'fontWeight': 'bold', 'backgroundColor': 'lightgrey'}
    ),
    html.H3(
        "> 39 binary labels to describe our pictures:",
        style={"textAlign": "center", "marginBottom": "20px", "color": "gray"}  # Center the subtitle with styling
    ),
    dcc.Graph(figure=fig),  # Add the bar plot to the layout
          html.H3(
        "> some labels change whith different images of the same poeple",
        style={"textAlign": "center", "marginBottom": "20px", "color": "gray"}  # Center the subtitle with styling
    ),
    dcc.Graph(figure=fig2),  # Add the bar plot to the layout

])

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
