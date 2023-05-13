import dash_bootstrap_components as dbc

from explainerdashboard.custom import *
from tabs.components.components import SelectStudentComponent
    

class WhatIfTab(ExplainerComponent):
    """
    A class for creating a 'What If' tab in an explainer dashboard.
    """

    def __init__(self, explainer, title="What if...", name=None,
                hide_selector=True, index_check=True,
                n_input_cols=4, **kwargs):
        """
        Initialize a WhatIfTab instance.

        Args:
            - explainer (ClassifierExplainer): Explainer instance containing dataset and trained model.
            - title (str): Title of the component.
            - name (str): The name of the component.
            - hide_selector (bool): Whether to display a selector or hide it.
            - n_input_cols (int): Number of columns to split features inputs in.
            - **kwargs: Optional keyword arguments.
        """

        # Call the parent constructor
        super().__init__(explainer, title, name)

        # Setting attributes
        self.input = FeatureInputComponent(explainer, name=self.name+"0",
                        hide_selector=hide_selector, n_input_cols=n_input_cols,
                        **update_params(kwargs, hide_index=False))
        self.index = SelectStudentComponent(explainer, name=self.name+"1",
                        hide_selector=hide_selector, **kwargs)
        self.prediction = ClassifierPredictionSummaryComponent(explainer, name=self.name+"2",
                        feature_input_component=self.input,
                        hide_star_explanation=True,
                        hide_selector=hide_selector, **kwargs)
        self.contribution = ShapContributionsGraphComponent(explainer, name=self.name+"3",
                        hide_selector=hide_selector, **kwargs)
        self.index_connector = IndexConnector(self.index, [self.input, self.contribution], 
                        explainer=explainer if index_check else None)

    def layout(self):
        """
        Layout of the component.

        Returns:
            - The layout of the component wrapped in a Bootstrap card.
        """

        # Create a Bootstrap container
        return dbc.Container([
            dbc.Row([
                # Display SelectStudentComponent
                dbc.Col(
                        self.index.layout()),

                # Display ClassifierPredictionSummaryComponent       
                dbc.Col(
                        self.prediction.layout()), 

                ], class_name="mt-4 gx-4"),
            dbc.Row([
                # Display ShapContributionsGraphComponent
                dbc.Col(
                        self.contribution.layout()),

                # Display FeatureInputComponent 
                dbc.Col(
                        self.input.layout()),   
                             
                ], class_name="mt-4 gx-4")
        ], fluid=True)