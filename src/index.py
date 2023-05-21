import base64
import io
import os
import webbrowser

import dash_bootstrap_components as dbc
import matplotlib
import pandas as pd

from app import app
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from explainerdashboard import ClassifierExplainer, ExplainerDashboard, ExplainerHub
from sklearn.model_selection import train_test_split
from supervised.automl import AutoML
from tabs.AutoMLReportTab import AutoMLReportTab
from tabs.FeaturesImportancesTab import FeaturesImportanceTab
from tabs.ClassificationStatsTab import ClassificationStatsTab
from tabs.WhatIfTab import WhatIfTab
from tabs.CounterfactualsTab import CounterfactualsTab
from waitress import serve

# Select agg as matplotlib backend.
matplotlib.use('agg')

# Alert for when Start button is pressed without a file uploaded.
no_file_alert = dbc.Alert(
        children="Please upload the dataset before pressing Start.", 
        color="danger",
        dismissable=True,
        duration=10000
),

# Alert for when the introduced model doesn't exist.
nonexistent_model_alert = dbc.Alert(
    children="The model you introduced doesn't exist. Please check if the name is correct or create a new one.", 
    color="danger",
    dismissable=True,
    duration=10000
),

# Alert for when explainer doesn't exist.
nonexistent_explainer_alert = dbc.Alert(
    children='Explainer not found. PLease check in saved_models/"NameYouIntroduced" if there is a explainer.dill file. If not, create a new model.', 
    color="danger",
    dismissable=True,
    duration=10000
),

# Alert for when an invalid file is uploaded.
invalid_type_alert = dbc.Alert(
    children="Invalid dataset type. Please be sure it is a .csv or .xlsx file.", 
    color="danger",
    dismissable=True,
    duration=10000
),

# Alert for when the explainer was already createds.
explainer_already_created_alert = dbc.Alert(
    children='Explainer already created. Please load a different train dataset or select "Load model" to load the saved model.', 
    color="danger",
    dismissable=True,
    duration=10000
),

# Layout of the home page.
app.layout = dbc.Container([
    html.Div(
        children=[          
        #NAVBAR
        dbc.Navbar(
            children=[
                html.A(
                    dbc.Row([
                        dbc.Col(
                            # Creating a brand for the navbar.
                            dbc.NavbarBrand(
                                "Academic Failure Prediction Tool", className="ms-2"
                            )
                        ),
                    ],                                                                     
                    className="g-0 ml-auto flex-nowrap mt-3 mt-md-0",
                    align="center",
                    ),
                ),
            ],     
        ),
        #BODY
        # A component that allows to switch between 2 tabs: New model or Load model
        dcc.Tabs([
            dcc.Tab(
                id="new-model",
                label="New model",
                children=[
                    html.Br(),
                    dbc.CardBody([
                        dcc.Markdown(
                            """
                            Welcome!

                            Before you upload a file, please make sure that the dataset meets the following requirements:
                             - Check that you are uploading a .csv or .xlsx file.
                             - Make sure that the first column of the dataset is reserved for the index.
                             - At least one of the columns should contain continuous values (represented as floating-point numbers)
                             - The last column should have only two unique values: "Dropout" and "No dropout".
                             - Make sure that both datasets have the same columns.

                             
                            Here you can upload 2 dataset:
                             - Train dataset: It will be used to train the model and get graphs of the model performance.
                             - Test dataset: This is the one that, based of the model obtained from the train dataset, will be predicted. Please leave last column empty.
                    
                            The code of this app is available at: https://github.com/0Kan0/Academic-Failure-Prediction-Tool""",
                            style={"margin": "0 10px"},
                        )
                    
                    ]),
                    html.Br(),
                    html.H3("Load train dataset"),
                    # A component that allows to upload the train dataset.
                    html.Div(
                        id='new-model-upload-train',
                        children=[
                            dcc.Upload(
                                id="new-model-upload-data-train",
                                children=[html.Div(["Drag and Drop or ", html.A("Select Files")])],
                                style={
                                    "width": "100%",
                                    "height": "60px",
                                    "lineHeight": "60px",
                                    "borderWidth": "1px",
                                    "borderStyle": "dashed",
                                    "borderRadius": "5px",
                                    "textAlign": "center",
                                    "margin": "10px",
                                },
                                multiple=False,
                            ),       
                        ],
                    ),

                    html.Br(),
                    dbc.Spinner(dbc.Row(
                        dbc.Col(
                            [
                                # Creating a table with the train data.
                                html.Div(id="new-model-output-data-train"),
                            ]
                        ),
                    )),

                    html.Br(),
                    html.H3("Load test dataset"),
                    # A component that allows to upload the test dataset.
                    html.Div(
                        id='new-model-upload-test',
                        children=[
                            dcc.Upload(
                                id="new-model-upload-data-test",
                                children=[html.Div(["Drag and Drop or ", html.A("Select Files")])],
                                style={
                                    "width": "100%",
                                    "height": "60px",
                                    "lineHeight": "60px",
                                    "borderWidth": "1px",
                                    "borderStyle": "dashed",
                                    "borderRadius": "5px",
                                    "textAlign": "center",
                                    "margin": "10px",
                                },
                                multiple=False,
                            ),       
                        ],
                    ),

                    html.Br(),
                    dbc.Spinner(dbc.Row(
                        dbc.Col(
                            [
                                # Creating a table with the test data.
                                html.Div(id="new-model-output-data-test"),
                            ]
                        ),
                    )),
                
                    html.Br(),
                    html.H3("Both model and explainer will be located in the saved_models folder."),

                    html.Br(),
                    html.Br(),
                    # Button to start the whole process of training the machine learning model and building the dashboard.
                    dbc.Button(f"Start", color="primary", id="new-model-start-button", style={'textAlign': 'center'}),

                    html.Br(),
                    dbc.Spinner(
                        html.Div(
                            id='new-model-dashboard-button',
                            # Invisible button that, when the whole proccess finish, appears and redirect to the dashboard.
                            children=[dbc.Button(
                                f"Go to dashboard",
                                id='new-model-dashboard-button-link',
                                href="http://127.0.0.1:8050", 
                                target="_blank",
                            )],
                            hidden=True
                        ),
                    ),
                    
                    html.Br(),
                    # Component to display the button that redirects to the dashboard.
                    html.Div(id='new-model-placeholder'),

                    html.Br(),
                    # Component that shows alerts.
                    html.Div(id="new-model-alert", children=[]),
                ]
            ),
#-----------------------------------------------------------------------------------------------------
        # Load model interface
            dcc.Tab(
                id="load-model",
                label="Load model",
                children=[
                    html.Br(),
                    dbc.CardBody([
                        dcc.Markdown(
                            """
                            Welcome!

                            Before you upload a file, please make sure that the dataset meets the following requirements:
                             - Check that you are uploading a .csv or .xlsx file.
                             - Make sure that the first column of the dataset is reserved for the index.
                             - At least one of the columns should contain continuous values (represented as floating-point numbers)
                             - The last column should have only two unique values: "Dropout" and "No dropout".

                             
                             Here you can upload 1 dataset:
                             - Test dataset: This is the one that, based of the model loaded, will be predicted. Please leave last column empty.

                            The code of this app is available at: https://github.com/0Kan0/Academic-Failure-Prediction-Tool""",
                            style={"margin": "0 10px"},
                        )
                    
                    ]),
                    html.Br(),
                    html.H3("Load test dataset"),
                    # A component that allows to upload the test dataset.
                    html.Div(
                        id='load-model-upload-div',
                        children=[
                            dcc.Upload(
                                id="load-model-upload-data-test",
                                children=[html.Div(["Drag and Drop or ", html.A("Select Files")])],
                                style={
                                    "width": "100%",
                                    "height": "60px",
                                    "lineHeight": "60px",
                                    "borderWidth": "1px",
                                    "borderStyle": "dashed",
                                    "borderRadius": "5px",
                                    "textAlign": "center",
                                    "margin": "10px",
                                },
                                multiple=False,
                            ),       
                        ]
                    ),

                    html.Br(),
                    dbc.Spinner(dbc.Row(
                        dbc.Col(
                            [
                                # Creating a table with the test data.
                                html.Div(id="load-model-output-data-test"),
                            ]
                        ),
                    )),

                    html.Br(),
                    html.H3("Name of the saved model (name of the folder inside the saved_model folder)"),
                    # A component to indicate the name of the folder that includes the saved explainer and AutoML model.
                    dcc.Input(
                        id="explainer-name",
                        type="text",
                        placeholder="Explainer name",
                        debounce=True,
                    ),
                    
                    html.Br(),
                    html.Br(),
                    # Button to start the whole process of training the machine learning model and building the dashboard.
                    dbc.Button(f"Start", color="primary", id="load-model-start-button", style={'textAlign': 'center'}),

                    html.Br(),
                    dbc.Spinner(
                        html.Div(
                            id='load-model-dashboard-button',
                            # Invisible button that, when the whole proccess finish, appears and redirect to the dashboard.
                            children=[dbc.Button(
                                f"Go to dashboard",
                                id='load-model-dashboard-button-link',
                                href="http://127.0.0.1:8050", 
                                target="_blank",
                            )],
                            hidden=True
                        ),
                    ),
                    
                    html.Br(),
                    # Component to display the button that redirects to the dashboard.
                    html.Div(id='load-model-placeholder'),

                    html.Br(),
                    # Component that shows alerts.
                    html.Div(id="load-model-alert", children=[]),
                ]
            )
        ])
    ])
])

def parse_data(contents, filename):
    """
    Parse data function to convert uploaded file content to pandas dataframe.

    Args:
        - contents (str): The content of the uploaded file.
        - filename (str): The name of the uploaded file.

    Returns:
        - A pandas.DataFrame containing the data from the uploaded file.
    """
    if contents:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)

        if "csv" in filename:
            # Assume that the user uploaded a CSV
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), delimiter=',', decimal='.')
        elif "xlsx" in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))

        # Return the pandas.DataFrame of the uploaded file
        return df

def create_AutoML_model(contents, filename):
    """
    Create an AutoML model using the provided dataset.

    Args:
        - contents (str): The contents of the dataset in CSV format.
        - filename (str): The filename of the dataset.

    Returns:
        - tuple: A tuple containing the following items:
            - The trained `model` object
            - The `X_test` and `y_test` datasets for evaluating the performance of the trained model
            - A `model_report` which contains information about the performance of each algorithm tried by the AutoML model
            - The `trained_model` object which contains the trained model
            - The original dataset `df`.
    """

    # Parse the CSV data and set the first column as the index
    df = parse_data(contents, filename)
    df.set_index(df.columns[0], inplace=True)

    # Replace class labels with binary values (0 or 1)
    df.iloc[:, -1] = df.iloc[:, -1].replace({"Dropout": 0, "No dropout": 1})

    # Split the data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(
        df[df.columns[:-1]], df.iloc[:, -1], test_size=0.2
    )

    # Set path and create saved_models folder
    saved_models_path = os.path.join("saved_models",filename.split(".")[0],"AutoML")
    os.makedirs(saved_models_path)

    # Define the AutoML model configuration
    model = AutoML(
            results_path=saved_models_path,
            algorithms=["Baseline", "Linear", "Decision Tree", "Random Forest", "Extra Trees", "Xgboost", "LightGBM", "CatBoost", "Neural Network", "Nearest Neighbors"],
            start_random_models=1,
            stack_models=True,
            train_ensemble=True,
            explain_level=2,
            validation_strategy={
                "validation_type": "split",
                "train_ratio": 0.80,
                "shuffle" : True,
                "stratify" : True,
            })

    # Train the AutoML model
    trained_model = model.fit(X_train, y_train)

    # Get the performance report for the AutoML model
    model_report = model.report()

    # Return the trained model object along with other useful objects for evaluating the model's performance
    return X_test, y_test, model, model_report, trained_model

def update_table(contents, filename):
    """
    Update the table with the parsed data and create a Dash DataTable object that allows the user to view the data.

    Args:
        - contents (str): The contents of the file that is uploaded by the user.
        - filename (str): The name of the file that is uploaded by the user.

    Returns:
        - A Dash Div object containing a DataTable that displays the parsed data, the name of the uploaded file, 
        and the raw content of the file. Or an alert if the file uploaded was not a .csv or .xlsx file.
    """

    if contents:
        # Send alert if file is not a .csv or .xlsx file
        try:
            df=parse_data(contents,filename)
        except Exception:
            return invalid_type_alert
        
        table = html.Div(
            [
                html.H4(filename),
                dash_table.DataTable(
                    columns=[{"name": i, "id": i} for i in df.columns],
                    data=df.to_dict("records"),
                    # Making the table scrollable.
                    style_table={'overflowX': 'scroll'},
                    # Allows the user to sort the data in the table.
                    sort_mode='multi',
                    # Preventing the user from deleting rows.
                    row_deletable=False,
                    # It allows the user to select a single column.
                    column_selectable='single',
                    selected_columns=[],
                    selected_rows=[],
                    page_action='native',
                    page_current= 0,
                    # Setting the table to have 20 rows per page
                    page_size= 20,
                    # To have a different background color for odd rows.
                    style_data_conditional=[        
                        {'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'}
                    ],
                ),
                html.Hr(),
                html.Div("Raw Content"),
                html.Pre(
                    contents[0:200] + "...",
                    style={"whiteSpace": "pre-wrap", "wordBreak": "break-all"},
                ),
            ]
        )
        return table

@app.callback(
        Output('new-model-output-data-train','children'),
        [Input('new-model-upload-data-train', 'contents'),
        Input('new-model-upload-data-train', 'filename')],
        prevent_initial_call=True
)
def new_model_update_train_table(contents, filename):
    """
    Call update_table for the train dataset

    Args:
        - contents (str): The contents of the file that is uploaded by the user.
        - filename (str): The name of the file that is uploaded by the user.

    Returns:
        - A table div of the train dataset
    """
    return update_table(contents, filename)

@app.callback(
        Output('new-model-output-data-test','children'),
        [Input('new-model-upload-data-test', 'contents'),
        Input('new-model-upload-data-test', 'filename')],
        prevent_initial_call=True
)
def new_model_update_test_table(contents, filename):
    """
    Call update_table for the test dataset

    Args:
        - contents (str): The contents of the file that is uploaded by the user.
        - filename (str): The name of the file that is uploaded by the user.

    Returns:
        - A table div of the test dataset
    """
    return update_table(contents, filename)

@app.callback(
        Output('new-model-dashboard-button', 'hidden'),
        Output('new-model-alert', 'children'),
        [State('new-model-upload-data-train', 'contents'),
         State('new-model-upload-data-train', 'filename'),
         State('new-model-upload-data-test', 'contents'),
         State('new-model-upload-data-test', 'filename')],
        Input('new-model-start-button', 'n_clicks'),
        prevent_initial_call=True
)
def new_model_create_dashboard(train_contents, train_filename, test_contents, test_filename, n_clicks):
    """
    Create a dashboard using the contents of a file and return a boolean and an alert message.

    Args:
        - contents (str): The contents of the file.
        - filename (str): The name of the file.
        - n_clicks (int or None): The number of clicks.

    Returns:
        - tuple: A tuple containing the following items:
            - A boolean changing the property "hidden" of the 'dashboard-button' html.Div.
            - Alert message if no file was uploaded and Start button was clicked.
    """

    # Return an alert if any of the parameters are None and keep the html.Div hidden
    if (train_contents is None or train_filename is None or test_contents is None or test_filename is None or n_clicks is None):
        return True, no_file_alert
    
    # Set global variable hub and saved_explainer path
    global hub

    # Send alert if explainer was already created
    try:
        # Create an AutoML model using the contents of the file
        train_X_test, train_y_test, model, model_report, trained = create_AutoML_model(train_contents, train_filename)
    except:
        return True, explainer_already_created_alert

    # Parse the test dataset, set first column as index and set test_X_test 
    df_test = parse_data(test_contents, test_filename)
    df_test.set_index(df_test.columns[0], inplace=True)
    test_X_test = df_test[df_test.columns[:-1]]

    # Create explainer for the train dataset and the test dataset
    train_explainer = ClassifierExplainer(model, train_X_test, train_y_test, labels=["Dropout", "No dropout"], target=df_test.columns[-1])
    test_explainer = ClassifierExplainer(model, test_X_test, labels=["Dropout", "No dropout"], target=df_test.columns[-1])

    # Create two explainer dashboards with different tabs
    db1 = ExplainerDashboard(test_explainer, header_hide_selector=True, hide_poweredby=True, title="Predictions Dashboard", 
                            tabs=[FeaturesImportanceTab, WhatIfTab, CounterfactualsTab(explainer=test_explainer, dataframe=df_test, trained_model=trained)],
                            description="In this dashboard, you can access the following tabs: Features Impact, What If... and Counterfactuals. You will be able to make predictions of the students from the test dataset.")
    db2 = ExplainerDashboard(train_explainer, hide_poweredby=True, title="AutoML Model Performance Dashboard",
                            tabs=[AutoMLReportTab(explainer=train_explainer, ML_report=model_report), ClassificationStatsTab],
                            description="In this dashboard, you can access the following tabs: AutoML Report and Classificaction Stats. You will be able to see all models tried, which one was the best of all, and different metrics to check it's performance.")
    
    # Create an explainer hub with the two dashboards
    hub = ExplainerHub([db1, db2], title="Academic Failure Prediction Tool", n_dashboard_cols=2,
                        description="")

    # Dump the explainer in the saved_explainer path
    saved_explainer_path = os.path.join("saved_models",train_filename.split(".")[0],"explainer.dill")
    train_explainer.dump(saved_explainer_path)

    # Return no alert message and reveals the html.Div
    return False, None

@app.callback(
        Output('new-model-placeholder', 'children'),
        Input('new-model-dashboard-button-link', 'n_clicks'),
        prevent_initial_call=True
)
def new_model_start_dashboard(n_clicks):
    """
    Starts the dashboard by running the ExplainerHub.

    Args:
        - n_clicks (int or None): The number of times the start button has been clicked.

    Returns:
        - None if n_clicks is None.
        - The output of the hub.run() method (an HTML object) if n_clicks is not None.
    """
    if(n_clicks is None):
        return None
    
    return hub.run(use_waitress=True)

@app.callback(
        Output('load-model-output-data-test','children'),
        [Input('load-model-upload-data-test', 'contents'),
        Input('load-model-upload-data-test', 'filename')],
        prevent_initial_call=True
)
def load_model_update_test_table(contents, filename):
    """
    Call update_table for the test dataset

    Args:
        - contents (str): The contents of the file that is uploaded by the user.
        - filename (str): The name of the file that is uploaded by the user.

    Returns:
        - A table div of the test dataset
    """
    return update_table(contents, filename)

@app.callback(
        Output('load-model-dashboard-button', 'hidden'),
        Output('load-model-alert', 'children'),
        [State('load-model-upload-data-test', 'contents'),
         State('load-model-upload-data-test', 'filename'),
         State('explainer-name', 'value')],
        Input('load-model-start-button', 'n_clicks'),
        prevent_initial_call=True
)
def load_model_create_dashboard(test_contents, test_filename, saved_explainer, n_clicks):
    """
    Create a dashboard using the contents of a file and return a boolean and an alert message.

    Args:
        - contents (str): The contents of the file.
        - filename (str): The name of the file.
        - n_clicks (int or None): The number of clicks.

    Returns:
        - tuple: A tuple containing the following items:
            - A boolean changing the property "hidden" of the 'dashboard-button' html.Div.
            - Alert message if no file was uploaded and Start button was clicked.
    """

    # Return an alert if any of the parameters are None and keep the html.Div hidden
    if (test_contents is None or test_filename is None or saved_explainer is None or n_clicks is None):
        return True, no_file_alert
    
    # Set global variable hub and paths for the saved model and explainer
    global hub
    saved_explainer_path = os.path.join("saved_models",saved_explainer,"explainer.dill")
    saved_models_path = os.path.join("saved_models",saved_explainer,"AutoML")

    # Return alert if model does not exist
    if not os.path.exists(saved_models_path):
        return True, nonexistent_model_alert
    
    # Return alert if explainer does not exist
    if not os.path.exists(saved_explainer_path):
        return True, nonexistent_explainer_alert
    
    # Load saved model and it's report
    load_model = AutoML(saved_models_path)
    model_report = load_model.report()

    # Parse the test dataset, set first column as index and set test_X_test 
    df_test = parse_data(test_contents, test_filename)
    df_test.set_index(df_test.columns[0], inplace=True)
    test_X_test = df_test[df_test.columns[:-1]]

    # Create explainer for the train dataset and the test dataset
    train_explainer = ClassifierExplainer.from_file(saved_explainer_path)
    test_explainer = ClassifierExplainer(load_model, test_X_test, labels=["Dropout", "No dropout"], target=df_test.columns[-1])

    # Create two explainer dashboards with different tabs
    db1 = ExplainerDashboard(test_explainer, header_hide_selector=True, hide_poweredby=True, title="Predictions Dashboard", 
                            tabs=[FeaturesImportanceTab, WhatIfTab, CounterfactualsTab(explainer=test_explainer, dataframe=df_test, trained_model=load_model)],
                            description="In this dashboard, you can access the following tabs: Features Impact, What If... and Counterfactuals. You will be able to make predictions of the students from the test dataset.")
    db2 = ExplainerDashboard(train_explainer, hide_poweredby=True, title="AutoML Model Performance Dashboard",
                            tabs=[AutoMLReportTab(explainer=train_explainer, ML_report=model_report), ClassificationStatsTab],
                            description="In this dashboard, you can access the following tabs: AutoML Report and Classificaction Stats. You will be able to see all models tried, which one was the best of all, and different metrics to check it's performance.")
    # Create an explainer hub with the two dashboards
    hub = ExplainerHub([db1, db2], title="Academic Failure Prediction Tool", n_dashboard_cols=2,
                        description="")
    
    # Return no alert message and reveals the html.Div
    return False, None

@app.callback(
        Output('load-model-placeholder', 'children'),
        Input('load-model-dashboard-button-link', 'n_clicks'),
        prevent_initial_call=True
)
def load_model_start_dashboard(n_clicks):
    """
    Starts the dashboard by running the ExplainerHub.

    Args:
        - n_clicks (int or None): The number of times the start button has been clicked.

    Returns:
        - None if n_clicks is None.
        - The output of the hub.run() method (an HTML object) if n_clicks is not None.
    """
    if(n_clicks is None):
        return None

    return hub.run(use_waitress=True, port="8050")

# A way to run the app in a local server.
if __name__ == '__main__':
    app.title = "Academic Failure Prediction Tool"
    
    webbrowser.open_new_tab('http://localhost:8080/')
    serve(app.server, host='0.0.0.0', port=8080)