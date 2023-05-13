import dash_bootstrap_components as dbc

from explainerdashboard.custom import *
    
class FeaturesImportanceTab(ExplainerComponent):
    """
    A class for creating a 'Feature Impact' tab in an explainer dashboard.
    """

    def __init__(self, explainer, title="Feature Impact", name=None,
                hide_descriptions=True, hide_selector=True, **kwargs):
        """
        Initialize a FeaturesImportanceTab instance.

        Args:
            - explainer (ClassifierExplainer): Explainer instance containing dataset and trained model.
            - title (str): Title of the component.
            - name (str): The name of the component.
            - hide_descriptions (bool): Whether to display descriptions of the variables.
            - hide_selector (bool): Whether to display a selector or hide it.
            - **kwargs: Optional keyword arguments.
        """
            
        # Call the parent constructor
        super().__init__(explainer, title, name)

        # Setting attributes
        self.importances = ImportancesComponent(explainer, name=self.name+"0", title="Feature Impact", subtitle="Average impact on predicted dropout",
                hide_selector=hide_selector, hide_descriptions=hide_descriptions)
        if not self.explainer.descriptions:
            self.hide_descriptions=True

    def layout(self):
        """
        Layout of the component.

        Returns:
            - The layout of the component wrapped in a Bootstrap card.
        """

        # Create a Bootstrap container
        return dbc.Container([
            dbc.Row([
                # Display ImportancesComponent
                dbc.Col([
                    self.importances.layout(),
                ]),
            ], class_name="mt-4"),
        ], fluid=True)