import dash
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
from dash.dependencies import Input, Output

import plotly.express as px

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, make_scorer

# Preparing the data for modelling

data = pd.read_csv("MassMobilizationProtests_cleaned.csv")
data = data.loc[data['protestduration'] <= 50, :]
data.index = list(range(data.shape[0]))
data = data.drop(['Unnamed: 0'], axis=1)

country_dict = {country: ccode for country, ccode in zip(data.loc[:, ['country', 'ccode']].drop_duplicates().country,
                                                         data.loc[:, ['country', 'ccode']].drop_duplicates().ccode)}
category_mapping = {'50-99': 1,
                    '100-999': 2,
                    '1000-1999': 3,
                    '2000-4999': 4,
                    '5000-10000': 5,
                    '>10000': 6}

inverse_category_mapping = {1: '50-99',
                            2: '100-999',
                            3: '1000-1999',
                            4: '2000-4999',
                            5: '5000-10000',
                            6: '>10000'}

stateresponse_category_mapping = {'accomodation': 1,
                                  'arrests': 2,
                                  'beatings': 3,
                                  'crowd dispersal': 4,
                                  'ignore': 5,
                                  'killings': 6,
                                  'shootings': 7}

stateresponse_inverse_category_mapping = {1: 'accomodation',
                                          2: 'arrests',
                                          3: 'beatings',
                                          4: 'crowd dispersal',
                                          5: 'ignore',
                                          6: 'killings',
                                          7: 'shootings'}
data["stateresponse1_encoded"] = data["stateresponse1"].replace(stateresponse_category_mapping)

data["participants_category_encoded"] = data.participants_category.replace(category_mapping)

protesterdemand_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
protesterdemand_encoder.fit(np.array(data.protesterdemand1).reshape(-1, 1))
encoded_protester_demand_df = pd.DataFrame(
    protesterdemand_encoder.transform(np.array(data.protesterdemand1).reshape(-1, 1)),
    columns=protesterdemand_encoder.categories_)

features = ['protestnumber', 'year', 'ccode', 'participants_category_encoded', 'protestduration', 'protesterviolence']

rfe_features = ['ccode', 'protestnumber', 'participants_category_encoded', 'protestduration', 'year',
                'protesterviolence']

self_features = ['ccode', 'participants_category_encoded', 'protesterviolence']

X = pd.concat([data[features], encoded_protester_demand_df], axis=1)
y = pd.DataFrame(data.stateresponse1_encoded)

X.columns = [i for i in list(X.columns) if type(i) == str] + [
    str(i).strip('(').strip(')').strip("'").strip(',').strip("'") for i in list(X.columns) if type(i) == tuple]
y.columns = ['stateresponse1']

X = X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
y = y.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

X = X[self_features]


def f1_cross_validation_score(model):
    pipe = make_pipeline(model)
    skf = StratifiedKFold(n_splits=5, random_state=31, shuffle=True)
    scorer = make_scorer(f1_score, average='micro')
    cv_results = cross_val_score(pipe, X, y, cv=skf, scoring=scorer, n_jobs=-1)
    return np.mean(cv_results)


# Training the LightGBM Classifier on the training dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
lgbmc = LGBMClassifier(learning_rate=0.05, max_depth=20, min_child_samples=15, num_leaves=20, reg_alpha=0.03)
pipeline = make_pipeline(lgbmc)
pipeline.fit(X_train, y_train)

fig_top_locations = px.bar(data, x=data.location.value_counts()[1:11].index, y=data.location.value_counts()[1:11])

sunburst_data = data.groupby(["region", "country"]).size().reset_index()
sunburst_data.columns = ["region", "country", "#Protests"]

# the style arguments for the sidebar.
SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '20%',
    'padding': '10px 5px',
    'background-color': '#e0bf65',
    'border-style': 'solid;',
    'border-width': 'large;',
    'border-radius': '5px;'
}

# the style arguments for the main content page.
CONTENT_STYLE = {
    'margin-left': '20%',
    'margin-right': '0%',

}

TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#191970'
}

CARD_TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#0074D9'
}

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

controls = html.Div(children=
[
    dcc.Checklist(
        id='overall-check',
        options=[
            {'label': 'Overall: 1990-2020', 'value': 'Overall'}

        ],
        style={
            'textAlign': 'center',
            'margin-left': '10%'
        },
        value=['Overall']
    ),

    html.Br(),
    html.Br(),
    html.Div(id='element-to-hide', children=[html.P('Year', style={
        'textAlign': 'center'
    }),
                                             dcc.Dropdown(
                                                 id='year-dropdown',
                                                 options=[{"label": year_str, "value": year} for year_str, year in
                                                          zip(pd.Series(data['year'].unique()).sort_values(
                                                              ascending=True),
                                                              pd.Series(data['year'].unique()).sort_values(
                                                                  ascending=True))],
                                                 value=1900,  # default value
                                                 multi=False,
                                                 searchable=False
                                             ),
                                             html.Br(),

                                             ], style={'display': 'block'}),

    html.P('Map Variable', style={
        'textAlign': 'center'
    }),
    dbc.Card(dcc.RadioItems(
        id='variable-select',
        options=[
            {'label': 'Protesters Demands', 'value': 'protesterdemand1'},
            {'label': 'Response from countries', 'value': 'stateresponse1'},
            {'label': 'Protest Duration', 'value': 'protestduration'}
        ],
        value='protestduration'
    )),
    html.Br(),
]
)

pred_controls = dbc.Row([
    dbc.Col([html.P('Country', style={
        'textAlign': 'center'
    }),
             dcc.Dropdown(
                 id='pred_country',
                 options=[{"label": country, "value": ccode} for country, ccode in
                          zip(data.loc[:, ['country', 'ccode']].drop_duplicates().country,
                              data.loc[:, ['country', 'ccode']].drop_duplicates().ccode)],
             )],
            md=4),
    dbc.Col([html.P('Number of Protesters', style={
        'textAlign': 'center'
    }),

             dcc.Dropdown(
                 id='pred_protesters',
                 options=[
                     {'label': '50-99', 'value': 1},
                     {'label': '100-999', 'value': 2},
                     {'label': '1000-1999', 'value': 3},
                     {'label': '2000-4999', 'value': 4},
                     {'label': '5000-10000', 'value': 5},
                     {'label': '>10000', 'value': 6}

                 ],

             )], md=4),
    dbc.Col([html.P('Is there violence?', style={
        'textAlign': 'center'
    }),
             dcc.Dropdown(
                 id='pred_violence',
                 options=[
                     {'label': 'Yes', 'value': '1'},
                     {'label': 'No', 'value': '0'}
                 ],

             )]
            , md=4)

], className='prediction-controls')

sidebar = html.Div(
    [
        html.H3('Parameters', style={'text-align': 'center'}),
        html.Hr(),
        controls,
        html.Pre("Click on the sectors for more detail")
    ],
    style=SIDEBAR_STYLE,
)

content_first_row = dbc.Row(
    [

        dbc.Col(
            dcc.Graph(
                id='sunburst',
                figure=px.sunburst(sunburst_data, path=['region', 'country'], values='#Protests',
                                   title="Total Protests area-wise breakdown",
                                   # width=1470, height=600,
                                   template='gridon',
                                   color_discrete_sequence=px.colors.qualitative.T10,

                                   ),
                # figure.update_layout(clickmode='event+select')

            ), md=6
        ),
        dbc.Col(
            dcc.Graph(
                id="top-protests-bar",
                config={"displayModeBar": False}
            ), md=6
        )
    ]
)

content_second_row = dbc.Row(
    [
        dbc.Col(
            dcc.Graph(
                id='map',
            ), md=12
        )
    ]
)

content_third_row = dbc.Row(
    [
        dbc.Col(
            dcc.Graph(
                id='demand-response',
                figure=px.density_heatmap(data, y='stateresponse1', x='protesterdemand1',

                                          # width=1470, height=600,
                                          template='gridon',
                                          labels={'stateresponse1': 'Response from country',
                                                  'protesterdemand1': 'Demand from protesters'}
                                          )

            ), md=12,
        )
    ]
)

content_fourth_row = dbc.Row(
    [
        dbc.Col(
            dcc.Graph(
                id='line-plot-pviolence',
                figure=px.scatter(data, y=data.groupby(pd.to_datetime(data.startdate).dt.year).protesterviolence.mean(),
                                  x=data.groupby(pd.to_datetime(data.startdate).dt.year).protesterviolence.mean().index,
                                  trendline='ols',
                                  trendline_color_override="#DC143C",
                                  labels={'x': 'Date', 'y': 'Average score of Protester Violence'},
                                  title="Average protester violence rates over the years",
                                  # width=1470, height=600,
                                  template='gridon'
                                  ).update_traces(mode='lines')

            ), md=6
        ),
        dbc.Col(
            dcc.Graph(
                id='line-plot-pduration',
                figure=px.scatter(data, y=data.groupby(pd.to_datetime(data.startdate).dt.year).protestduration.mean(),
                                  x=data.groupby(pd.to_datetime(data.startdate).dt.year).protestduration.mean().index,
                                  trendline='ols',
                                  trendline_color_override="#DC143C",
                                  labels={'x': 'Date', 'y': 'Average protest duration'},
                                  title="Average protest duration over the years",
                                  # width=1470, height=600,
                                  template='gridon'
                                  ).update_traces(mode='lines')

            ), md=6
        )
    ]
)

content_prediction = dbc.Row(dbc.Col(dcc.Textarea(
    title="Predicted repsonse",
    id='prediction',
    placeholder="Predicted response",
    draggable=False,
    className="prediction-text"

), md=12))

content = html.Div(
    [
        html.Div(children=[
            html.Br(),
            html.Br(),
            html.Img(src=app.get_asset_url("icon.jpg"), className="header-emoji", width="80", height="80"),
            html.Br(),
            html.H1(children="Mass Mobilization Protests", className="header-title"),
            html.P(
                children="Analyzing the mass mobilization protests against states across the world "
                         "between 1990 and 2020 ", className="header-description",
            ),
            html.Br(),
            html.Br()

        ], className="header"),
        html.Hr(),
        html.H2("Top Protests area-wise"),
        content_first_row,
        html.H2("Parameter-wise spatial distribution of protests"),
        content_second_row,
        html.H2("Demand versus Response"),
        content_third_row,
        html.H2("Time Series Analysis"),
        content_fourth_row,
        html.H2("Predicting the most likely response from country"),
        pred_controls,
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        content_prediction
    ],
    style=CONTENT_STYLE
)

app.layout = html.Div([sidebar, content])


@app.callback(
    Output('element-to-hide', 'style'),
    [Input('overall-check', 'value')])
def show_or_hide_detail(overall):
    print(overall)
    if 'Overall' in overall:
        return {'display': 'none'}
    else:
        return {'display': 'block'}


@app.callback(
    Output('sunburst', 'clickData'),
    [Input('overall-check', 'value')])
def reset_click_data(overall):
    print(overall)
    if 'Overall' in overall:
        return None


@app.callback(Output('map', 'figure'),
              [Input('variable-select', 'value'), Input('year-dropdown', 'value'), Input('overall-check', 'value')]
              )
def update_map(selected_variable, selected_year, overall):
    name_encoder = {'protestduration': "Protest Duration", 'stateresponse1': "response from states(countries)",
                    'protesterdemand1': "demand from the protesters"}
    if 'Overall' in overall:
        if selected_variable == 'protestduration':
            figure = px.choropleth(
                data.groupby("country")['protestduration'].mean().reset_index(),
                locationmode='country names', locations="country", color="protestduration",
                color_continuous_scale="Mint",
                scope="world",
                title="Average protest duration in days overall",
                # width=1470, height=600,
                template='gridon',
                projection='natural earth'
            )
            figure.update_layout()
            return figure

        else:
            figure = px.choropleth(
                data.groupby("country")[
                    selected_variable].describe().top.reset_index(),
                locationmode='country names', locations="country", color="top",
                color_discrete_sequence=px.colors.qualitative.T10,
                scope="world",
                title="Most frequent " + name_encoder[selected_variable] + " overall",
                # width=1470, height=600,
                template='gridon',
                projection='natural earth')
            figure.update_layout()
            return figure
    if selected_variable == 'protestduration':
        figure = px.choropleth(
            data.loc[data.year == selected_year, :].groupby("country")['protestduration'].mean().reset_index(),
            locationmode='country names', locations="country", color="protestduration",
            color_continuous_scale="Mint",
            scope="world",
            title="Average protest duration in days in year " + str(selected_year),
            # width=1470, height=600,
            template='gridon',
            projection='natural earth'
        )
        figure.update_layout()
        return figure

    figure = px.choropleth(
        data.loc[data.year == selected_year, :].groupby("country")[selected_variable].describe().top.reset_index(),
        locationmode='country names', locations="country", color="top",
        color_discrete_sequence=px.colors.qualitative.T10,
        scope="world",
        title="Most frequent " + name_encoder[selected_variable] + " in year " + str(selected_year),
        # width=1470, height=600,
        template='gridon',
        projection='natural earth'
    )
    figure.update_layout()
    return figure


@app.callback(
    Output('sunburst', 'figure'),
    [Input('year-dropdown', 'value'), Input('overall-check', 'value')]
)
def update_sunburst(selected_year, overall):
    if 'Overall' in overall:
        figure = px.sunburst(sunburst_data, path=['region', 'country'], values='#Protests',
                             title="Total Protests area-wise breakdown overall",
                             # width=1470, height=600,
                             template='gridon',
                             color_discrete_sequence=px.colors.qualitative.T10,

                             )
        return figure

    sunburst_data_year = data.loc[data.year == selected_year, :].groupby(["region", "country"]).size().reset_index()
    sunburst_data_year.columns = ["region", "country", "#Protests"]
    figure = px.sunburst(sunburst_data_year, path=['region', 'country'], values='#Protests',
                         title="Total Protests area-wise breakdown",
                         # width=1470, height=600,
                         template='gridon',
                         color_discrete_sequence=px.colors.qualitative.T10
                         )
    return figure


@app.callback(
    Output('top-protests-bar', 'figure'),
    [Input('year-dropdown', 'value'), Input('sunburst', 'clickData'), Input('overall-check', 'value')])
def update_top_protests(selected_year, clickData, overall):
    if 'Overall' in overall:
        if clickData is None:
            filtered_df = pd.DataFrame(
                data["location"].value_counts()[0:10].reset_index())
            filtered_df["Number of Protests"] = filtered_df["location"]
            filtered_df["location"] = filtered_df["index"]
            fig = px.bar(filtered_df, y="location", x="Number of Protests",
                         title="Top 10 cities/towns with most protests",
                         template="gridon")

            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            return fig

        # print(clickData['points'][0]['id'].split('/'))
        elif len(clickData['points'][0]['id'].split('/')) == 1:
            print("Region click")
            filtered_df = pd.DataFrame(
                data.loc[(data.region == clickData['points'][0]['id'].split('/')[0]), :][
                    "location"].value_counts()[0:10].reset_index())
            filtered_df["Number of Protests"] = filtered_df["location"]
            filtered_df["location"] = filtered_df["index"]
            fig = px.bar(filtered_df, y="location", x="Number of Protests",
                         title="Top 10 cities/towns with most protests in " + clickData['points'][0]['id'].split('/')[
                             0],
                         template="gridon")

            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            print(filtered_df)
            return fig
        else:
            print("Country click")
            filtered_df = pd.DataFrame(
                data.loc[(data.country == clickData['points'][0]['id'].split('/')[1]),
                :][
                    "location"].value_counts()[0:10].reset_index())
            filtered_df["Number of Protests"] = filtered_df["location"]
            filtered_df["location"] = filtered_df["index"]
            fig = px.bar(filtered_df, y="location", x="Number of Protests",
                         title="Top 10 cities/towns with most protests in " + clickData['points'][0]['id'].split('/')[
                             1],
                         template="gridon")

            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            print(filtered_df)
            return fig
    if clickData is None:
        filtered_df = pd.DataFrame(
            data.loc[data.year == selected_year, :]["location"].value_counts()[0:10].reset_index())
        filtered_df["Number of Protests"] = filtered_df["location"]
        filtered_df["location"] = filtered_df["index"]
        fig = px.bar(filtered_df, y="location", x="Number of Protests",
                     title="Top 10 cities/towns with most protests in year " + str(selected_year),
                     template="gridon")

        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        return fig

    # print(clickData['points'][0]['id'].split('/'))
    elif len(clickData['points'][0]['id'].split('/')) == 1:
        print("Region click")
        filtered_df = pd.DataFrame(
            data.loc[(data.year == selected_year) & (data.region == clickData['points'][0]['id'].split('/')[0]), :][
                "location"].value_counts()[0:10].reset_index())
        filtered_df["Number of Protests"] = filtered_df["location"]
        filtered_df["location"] = filtered_df["index"]
        fig = px.bar(filtered_df, y="location", x="Number of Protests",
                     title="Top 10 cities/towns with most protests in " + clickData['points'][0]['id'].split('/')[
                         0] + " in" + " year " + str(selected_year),
                     template="gridon")

        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        print(filtered_df)
        return fig
    else:
        print("Country click")
        filtered_df = pd.DataFrame(
            data.loc[(data.year == selected_year) & (data.country == clickData['points'][0]['id'].split('/')[1]), :][
                "location"].value_counts()[0:10].reset_index())
        filtered_df["Number of Protests"] = filtered_df["location"]
        filtered_df["location"] = filtered_df["index"]
        fig = px.bar(filtered_df, y="location", x="Number of Protests",
                     title="Top 10 cities/towns with most protests in " + clickData['points'][0]['id'].split('/')[
                         1] + " in" + " year " + str(selected_year),
                     template="gridon")

        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        print(filtered_df)
        return fig


@app.callback(Output('prediction', 'value'),
              [Input('pred_country', 'value'), Input('pred_protesters', 'value'), Input('pred_violence', 'value')]
              )
def predict_state_response(selected_country, selected_num_protesters, selected_violence):
    if selected_country is not None and selected_num_protesters is not None and selected_violence is not None:
        print(stateresponse_inverse_category_mapping[int(pipeline.predict(
            pd.DataFrame([int(selected_country), int(selected_num_protesters), int(selected_violence)]).T))])
        return stateresponse_inverse_category_mapping[int(pipeline.predict(
            pd.DataFrame([int(selected_country), int(selected_num_protesters), int(selected_violence)]).T))]


if __name__ == '__main__':
    app.run_server(debug=True)
