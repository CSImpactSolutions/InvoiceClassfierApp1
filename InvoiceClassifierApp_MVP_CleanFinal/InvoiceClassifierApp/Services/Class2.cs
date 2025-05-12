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

    public class EmbeddingSimilarityMatrixExporter
    {
        private readonly string embeddingsFolder;

        public EmbeddingSimilarityMatrixExporter(string embeddingsFolder)
        {
            this.embeddingsFolder = embeddingsFolder;
        }

        public void ExportMatrix(string outputCsvPath)
        {
            var embeddings = new Dictionary<string, float[]>();

            // Load all embedding files
            foreach (var file in Directory.GetFiles(embeddingsFolder, "*.json"))
            {
                var json = File.ReadAllText(file);
                var vector = JsonSerializer.Deserialize<float[]>(json);
                var fileName = Path.GetFileNameWithoutExtension(file);
                embeddings[fileName] = vector;
            }

            var keys = embeddings.Keys.OrderBy(k => k).ToList(); // Sorted for consistent output

            using var writer = new StreamWriter(outputCsvPath);

            // Write header row
            writer.Write("FileName");
            foreach (var col in keys)
            {
                writer.Write($",{col}");
            }
            writer.WriteLine();

            // Write each row with similarities
            foreach (var rowKey in keys)
            {
                writer.Write(rowKey);
                foreach (var colKey in keys)
                {
                    var sim = CosineSimilarity(embeddings[rowKey], embeddings[colKey]);
                    writer.Write($",{sim.ToString("F4", System.Globalization.CultureInfo.InvariantCulture)}");
                }
                writer.WriteLine();
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
