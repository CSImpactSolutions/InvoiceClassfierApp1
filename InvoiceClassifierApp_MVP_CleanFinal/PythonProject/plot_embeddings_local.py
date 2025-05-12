import os
import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px

# CONFIG
EMBEDDINGS_FOLDER = "./embeddings"  # Folder with .json vectors
PREDICTIONS_CSV = "./predictions.csv"  # CSV file with filename, label
PDF_FOLDER = "./Invoices"  # Folder containing the original PDFs
OUTPUT_HTML = "embedding_plot_with_links.html"

# Load predictions
predictions_df = pd.read_csv(PREDICTIONS_CSV)
predictions_df["key"] = predictions_df["Filename"].str.replace(".pdf", "", case=False)

# Load embeddings and assign labels/types
records = []
for file in os.listdir(EMBEDDINGS_FOLDER):
    if file.endswith(".json"):
        base_name = file.replace(".json", "")
        is_reference = base_name.lower().startswith(("craftsman", "healthcare", "upwork"))
        label_match = predictions_df[predictions_df["key"].str.lower() == base_name.lower()]
        label = label_match["PredictedLabel"].values[0] if not label_match.empty else "unlabeled"

        with open(os.path.join(EMBEDDINGS_FOLDER, file), "r") as f:
            vec = json.load(f)
        if isinstance(vec, list) and all(isinstance(x, (float, int)) for x in vec):
            records.append({
                "filename": base_name + ".pdf",
                "label": label,
                "vector": vec,
                "type": "reference" if is_reference else "inferred"
            })

# Perform PCA reduction
df = pd.DataFrame(records)
X = np.array(df["vector"].tolist())
pca = PCA(n_components=2)
points = pca.fit_transform(X)
df["x"] = points[:, 0]
df["y"] = points[:, 1]
df["pdf_link"] = df["filename"].apply(lambda name: os.path.join(PDF_FOLDER, name).replace('\\', '/'))

# Create interactive Plotly figure
fig = px.scatter(
    df, x="x", y="y",
    color="label",
    symbol="type",
    hover_data=["filename", "label", "type"],
    title="📄 Embedding Visualization with Download Links"
)

fig.update_traces(
    customdata=df[["filename", "pdf_link"]],
    hovertemplate="""<b>%{customdata[0]}</b><br>
    Label: %{marker.color}<br>
    Type: %{marker.symbol}<br>
    <a href='%{customdata[1]}' target='_blank'>📥 Download PDF</a><extra></extra>"""
)

fig.write_html(OUTPUT_HTML, include_plotlyjs='cdn')
print(f"✅ Saved to {OUTPUT_HTML}")