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
predictions_df["key"] = (
    predictions_df["Filename"]
    .str.replace(".pdf", "", case=False)
    .str.strip()
)

# DEBUG: Show loaded keys
print("🔍 Sample keys from predictions_df:")
print(predictions_df["key"].head(10).tolist())

# Load embeddings and assign labels/types
records = []
for file in os.listdir(EMBEDDINGS_FOLDER):
    if file.endswith(".json"):
        raw_name = file.replace(".json", "")
        base_name_with_pdf = raw_name.replace("_", " ").strip() + ".pdf"
        base_key = base_name_with_pdf.replace(".pdf", "")

        label_match = predictions_df[predictions_df["key"].str.lower() == base_key.lower()]
        label = label_match["PredictedLabel"].values[0] if not label_match.empty else "unlabeled"
        is_reference = label.lower() in {"craftsman", "healthcare", "upwork"}

        # Debug: match status
        if label_match.empty:
            print(f"❌ No label found for: {base_key}")
        else:
            print(f"✅ Matched: {base_key} -> {label}")

        with open(os.path.join(EMBEDDINGS_FOLDER, file), "r") as f:
            vec = json.load(f)
        if isinstance(vec, list) and all(isinstance(x, (float, int)) for x in vec):
            records.append({
                "filename": base_name_with_pdf,
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

# Ensure label column is treated as a category for consistent coloring
df["label"] = df["label"].astype("category")

# Create interactive Plotly figure with colors by label and shapes by type
fig = px.scatter(
    df,
    x="x",
    y="y",
    color="label",
    symbol="type",
    hover_data=["filename", "label", "type"],
    title="📄 Embedding Visualization by Label and Type"
)

# Add customdata for filename and PDF link
fig.update_traces(
    customdata=df[["filename", "pdf_link"]],
    hovertemplate="""
    <b>%{customdata[0]}</b><br>
    Label: %{marker.color}<br>
    Type: %{marker.symbol}<br>
    <a href='%{customdata[1]}' target='_blank'>📥 Download PDF</a>
    <extra></extra>
    """
)

# Save the interactive plot to HTML
fig.write_html(OUTPUT_HTML, include_plotlyjs='cdn')
print(f"✅ Saved interactive plot to {OUTPUT_HTML}")
