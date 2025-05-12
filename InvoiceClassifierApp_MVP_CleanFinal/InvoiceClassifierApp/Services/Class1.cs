using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace InvoiceClassifierApp.Services
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Text.Json;

    public class EmbeddingSimilarityAnalyzer
    {
        private readonly string embeddingsFolder;

        public EmbeddingSimilarityAnalyzer(string embeddingsFolder)
        {
            this.embeddingsFolder = embeddingsFolder;
        }

        public void Analyze(string outputCsvPath)
        {
            var embeddings = new Dictionary<string, float[]>();

            // Read all .json files (each should contain a float[] vector)
            foreach (var file in Directory.GetFiles(embeddingsFolder, "*.json"))
            {
                var json = File.ReadAllText(file);
                var vector = JsonSerializer.Deserialize<float[]>(json);
                embeddings[Path.GetFileNameWithoutExtension(file)] = vector;
            }

            var results = new List<(string, string, double)>();
            var keys = embeddings.Keys.ToList();

            // Compare each pair of embeddings
            for (int i = 0; i < keys.Count; i++)
            {
                for (int j = i + 1; j < keys.Count; j++)
                {
                    var a = keys[i];
                    var b = keys[j];
                    var sim = CosineSimilarity(embeddings[a], embeddings[b]);
                    results.Add((a, b, sim));
                }
            }

            // Write results to CSV
            using var writer = new StreamWriter(outputCsvPath);
            writer.WriteLine("FileA,FileB,SimilarityScore");

            foreach (var (a, b, score) in results)
            {
                writer.WriteLine($"\"{a}\",\"{b}\",{score:F4}");
            }
        }

        private double CosineSimilarity(float[] vecA, float[] vecB)
        {
            double dot = 0.0, magA = 0.0, magB = 0.0;

            for (int i = 0; i < vecA.Length; i++)
            {
                dot += vecA[i] * vecB[i];
                magA += Math.Pow(vecA[i], 2);
                magB += Math.Pow(vecB[i], 2);
            }

            return dot / (Math.Sqrt(magA) * Math.Sqrt(magB));
        }
    }
}

