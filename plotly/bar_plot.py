from plotly.graph_objs._figure import Figure
import plotly.express as px
import pandas as pd
from sklearn.datasets import load_iris
from icecream import ic
import plotly.io as pio

# pio.renderers.default = "vscode"

def main():
    
    df = load_iris()
    # ic(df)
    
    X = df.data
    ic(X.shape)
    
    Y = df.target
    ic(Y.shape)
    
    feature_name = df.feature_names
    ic(feature_name)
    
    feature_mean = X.mean(axis=0)
    feature_std = X.std(axis=0)
    
    ic(type(feature_name))
    ic(feature_mean.tolist())
    ic(feature_std.tolist())
    
    fig: Figure = px.bar(
        x=feature_name,
        y=feature_mean,
        error_y=feature_std,
        title="Average of each feature of the iris dataset",
        labels={"x": "Feature", "y": "Mean"},
    )
    fig.show()
    
    
if __name__ == "__main__":
    main()