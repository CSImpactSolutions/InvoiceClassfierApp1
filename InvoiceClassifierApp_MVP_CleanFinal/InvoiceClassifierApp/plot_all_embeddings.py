import os
import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import webbrowser

# ---- CONFIG ----
PREDICTIONS_PATH = "predictions.csv"
EMBEDDINGS_DIR = "embeddings_extracted"
PDF_DIR = "Invoices"
HTML_OUT = "embedding_plot.html"
# ----------------

# Load predictions
predictions_df = pd.read_csv(PREDICTIONS_PATH)
predictions_df["key"] = predictions_df["Filename"].str.replace(".pdf", "", case=False)

# Load all embeddings
records = []
for file in os.listdir(EMBEDDINGS_DIR):
    if file.endswith(".json"):
        base_name = file.replace(".json", "")
        match = predictions_df[predictions_df["key"].str.lower() == base_name.lower()]
        if not match.empty:
            with open(os.path.join(EMBEDDINGS_DIR, file), "r") as f:
                try:
                    vec = json.load(f)
                    if isinstance(vec, list) and all(isinstance(x, (float, int)) for x in vec):
                        records.append({
                            "filename": match.iloc[0]["Filename"],
                            "label": match.iloc[0]["PredictedLabel"],
                            "vector": vec
                        })
                except Exception as e:
                    print(f"Failed to load {file}: {e}")

# Convert to DataFrame
df = pd.DataFrame(records)
X = np.array(df["vector"].tolist())

# Reduce with PCA
pca = PCA(n_components=2)
points = pca.fit_transform(X)
df["x"] = points[:, 0]
df["y"] = points[:, 1]

# Create plot
fig = px.scatter(
    df,
    x="x", y="y",
    color="label",
    hover_data=["filename", "label"],
    title="Document Embeddings (by Predicted Label)"
)

fig.update_traces(
    customdata=df["filename"],
    hovertemplate="<b>%{customdata}</b><br>Label: %{marker.color}<extra></extra>"
)

fig.write_html(HTML_OUT, include_plotlyjs='cdn')
print(f"✅ Saved to {HTML_OUT}")
webbrowser.open(HTML_OUT)