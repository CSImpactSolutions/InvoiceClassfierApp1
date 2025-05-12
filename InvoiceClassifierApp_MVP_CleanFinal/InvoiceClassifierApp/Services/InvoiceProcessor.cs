
using System.Collections;
using System.Xml.Linq;
using InvoiceClassifierApp.Models;

namespace InvoiceClassifierApp.Services;

public class InvoiceProcessor
{
    private readonly OpenAIEmbeddingService _embedding;
    private readonly KnnClassifier _knn;

    public InvoiceProcessor(OpenAIEmbeddingService embedding, KnnClassifier knn)
    {
        _embedding = embedding;
        _knn = knn;
    }

    public async Task TrainAsync(List<InvoiceVector> data)
    {
        // Step 1: Prepare an empty list for training data
        var trainingData = new List<(string Label, string Filename, float[] Vector)>();

        // Step 2: Loop through and embed each document properly
        foreach (var doc in data)
        {
            if (doc == null || string.IsNullOrWhiteSpace(doc.Text))
            {
                Console.WriteLine($"Skipping invalid document: {doc?.Filename ?? "null"}");
                continue;
            }

            try
            {
                // Generate or load embedding
                doc.Vector = await _embedding.GetOrLoadEmbeddingAsync(doc.Filename, doc.Text);

                // Add to training data
                trainingData.Add((doc.Label, doc.Filename, doc.Vector));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error embedding {doc.Filename}: {ex.Message}");
            }
        }

        // Step 3: Train KNN with the clean and complete training data
        _knn.Fit(trainingData);
    }

     public async Task<string> ClassifyAsync(string filename, string text)
    {
        // Generate or load embedding (with caching)
        var vector = await _embedding.GetOrLoadEmbeddingAsync(filename, text);

        // Use the correct method 'PredictLabel' instead of the non-existent 'Predict'
        return _knn.PredictLabel(vector);
    }
}
