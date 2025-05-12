import os
import json
import pandas as pd
import numpy as np
import umap
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors

# CONFIG
EMBEDDINGS_FOLDER = "./embeddings"
PREDICTIONS_CSV = "./predictions.csv"
PDF_FOLDER = "./Invoices"
OUTPUT_HTML = "embedding_plot_umap_click.html"

# Load predictions
predictions_df = pd.read_csv(PREDICTIONS_CSV)
predictions_df["key"] = predictions_df["Filename"].str.replace(".pdf", "", case=False).str.strip()

# Load embeddings and assign labels/types
records = []
for file in os.listdir(EMBEDDINGS_FOLDER):
    if file.endswith(".json"):
        raw_name = file.replace(".json", "")
        base_name_cleaned = raw_name.replace("_", " ").strip()
        base_name_with_pdf = base_name_cleaned if base_name_cleaned.lower().endswith(".pdf") else base_name_cleaned + ".pdf"
        base_key = base_name_with_pdf.replace(".pdf", "")

        label_match = predictions_df[predictions_df["key"].str.lower() == base_key.lower()]
        label = label_match["PredictedLabel"].values[0] if not label_match.empty else "unlabeled"
        is_reference = label.lower() in {"craftsman", "healthcare", "upwork"}

        with open(os.path.join(EMBEDDINGS_FOLDER, file), "r") as f:
            vec = json.load(f)
        if isinstance(vec, list) and all(isinstance(x, (float, int)) for x in vec):
            records.append({
                "filename": base_name_with_pdf,
                "label": label,
                "vector": vec,
                "type": "reference" if is_reference else "inferred",
                "pdf_link": os.path.join(PDF_FOLDER, base_name_with_pdf).replace("\\", "/")
            })

# UMAP dimensionality reduction
df = pd.DataFrame(records)
X = np.array(df["vector"].tolist())
embedding = umap.UMAP(n_neighbors=10, min_dist=0.1, random_state=42).fit_transform(X)
df["x"] = embedding[:, 0]
df["y"] = embedding[:, 1]

# Identify possible misclassifications using nearest neighbors
print("\n🔍 Possible misclassified documents:")
labels = df["label"].values
nbrs = NearestNeighbors(n_neighbors=4).fit(embedding)
distances, indices = nbrs.kneighbors(embedding)
for idx, neighbors in enumerate(indices):
    own_label = labels[idx]
    neighbor_labels = labels[neighbors[1:]]  # skip self
    if not all(nl == own_label for nl in neighbor_labels):
        print(f"- {df.iloc[idx]['filename']} (label: {own_label}) is near: {', '.join(neighbor_labels)}")

# Create Plotly scatter plot
fig = go.Figure()
symbols = {"reference": "diamond", "inferred": "circle"}

for label in df["label"].unique():
    for type_ in df["type"].unique():
        subset = df[(df["label"] == label) & (df["type"] == type_)]
        fig.add_trace(go.Scatter(
            x=subset["x"],
            y=subset["y"],
            mode="markers",
            name=f"{label} ({type_})",
            customdata=subset[["filename", "pdf_link", "type"]],
            marker=dict(symbol=symbols.get(type_, "circle"), size=10),
            hovertemplate="<b>%{customdata[0]}</b><br>Type: %{customdata[2]}<extra></extra>"
        ))

# Set layout and add JS click handler
div_id = "plotly-div"
fig.update_layout(title="📄 UMAP Visualization by Label and Type (Click to Open PDF)")

fig.write_html(
    OUTPUT_HTML,
    include_plotlyjs='cdn',
    full_html=True,
    config={"responsive": True},
    div_id=div_id,
    post_script=f"""
    document.getElementById('{div_id}').on('plotly_click', function(data) {{
        const pdf = data.points[0].customdata[1];
        if (pdf) {{
            window.open(pdf, '_blank');
        }}
    }});
    """
)

print(f"\n✅ Saved interactive clickable UMAP plot to {OUTPUT_HTML}")
