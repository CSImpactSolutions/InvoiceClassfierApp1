public class KnnClassifier
{
    private readonly int _k;
    private readonly bool _enableLogging;
    private List<(string Label, string Filename, float[] Vector)> _trainingData;

    public KnnClassifier(int k, bool enableLogging = true)
    {
        _k = k;
        _enableLogging = enableLogging;
        _trainingData = new List<(string Label, string Filename, float[] Vector)>();
    }

    public void Fit(List<(string Label, string Filename, float[] Vector)> trainingData)
    {
        _trainingData = trainingData;
    }

    public string PredictLabel(float[] inputVector)
    {
        var neighbors = _trainingData
            .Where(td => td.Vector != null && td.Vector.Length == inputVector.Length)
            .Select(td => new
            {
                td.Label,
                td.Filename,
                Score = CosineSimilarity(td.Vector, inputVector)
            })
            .OrderByDescending(x => x.Score)
            .Take(Math.Min(_k, _trainingData.Count))
            .ToList();

        if (!neighbors.Any())
        {
            if (_enableLogging)
                Console.WriteLine("⚠️ No neighbors found for prediction.");
            return "unknown";
        }

        if (_enableLogging)
        {
            Console.WriteLine("🔍 Top neighbors:");
            foreach (var neighbor in neighbors)
            {
                Console.WriteLine($" - {neighbor.Filename ?? "unknown"} | Label: {neighbor.Label} | Similarity: {neighbor.Score:F4}");
            }
        }

        // Group by label with both count and total similarity score
        var labelGroups = neighbors
            .GroupBy(x => x.Label)
            .Select(g => new
            {
                Label = g.Key,
                Count = g.Count(),
                TotalScore = g.Sum(x => x.Score)
            })
            .OrderByDescending(g => g.Count)
            .ThenByDescending(g => g.TotalScore)
            .ToList();

        string predictedLabel = labelGroups.First().Label;

        if (_enableLogging)
            Console.WriteLine($"✅ Predicted label: {predictedLabel}");

        return predictedLabel;
    }

    public Dictionary<string, float> PredictLabelProbabilities(float[] inputVector)
    {
        var neighbors = _trainingData
            .Where(td => td.Vector != null && td.Vector.Length == inputVector.Length)
            .Select(td => new
            {
                td.Label,
                Score = CosineSimilarity(td.Vector, inputVector)
            })
            .OrderByDescending(x => x.Score)
            .Take(Math.Min(_k, _trainingData.Count))
            .ToList();

        var labelProbabilities = neighbors
            .GroupBy(x => x.Label)
            .ToDictionary(
                g => g.Key,
                g => g.Count() / (float)_k
            );

        return labelProbabilities;
    }

    private double CosineSimilarity(float[] a, float[] b)
    {
        double dot = 0.0, normA = 0.0, normB = 0.0;

        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        if (normA == 0 || normB == 0)
            return 0;

        return dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
    }
}
