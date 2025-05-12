
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using Microsoft.VisualBasic;
using OpenAI.Embeddings;

namespace InvoiceClassifierApp.Services;

public class OpenAIEmbeddingService
{
    private readonly string _apiKey;
    private readonly HttpClient _http;

    public OpenAIEmbeddingService(string apiKey)
    {
        _apiKey = apiKey;
        _http = new HttpClient();
        _http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);
    }

    public async Task<float[]> GetEmbeddingAsync(string text)
    {
        var payload = new
        {
            input = text,
            model = "text-embedding-3-large"
        };
        const int chunkSize = 6000; // Approx. ~500 tokens
        var chunks = ChunkText(text, chunkSize);
        // Initialize the embedding client with the API key
        OpenAI.Embeddings.EmbeddingClient client = new(payload.model, Environment.GetEnvironmentVariable("OPENAI_API_KEY"));
        var embeddings = new List<float[]>();
        foreach (var chunk in chunks)
        {
            List<string> inputs = new() { chunk };
            // Generate embeddings for the input texts
            OpenAIEmbeddingCollection collection = await client.GenerateEmbeddingsAsync(inputs);
            // Convert the embeddings to float arrays
            float[] embedding1 = collection[0].ToFloats().ToArray();
            embeddings.Add(collection[0].ToFloats().ToArray());
        }

        // Average the embeddings to return a single float array
        return AverageVectors(embeddings);


    }

    private List<string> ChunkText(string text, int chunkSize)
    {
        var chunks = new List<string>();
        for (int i = 0; i < text.Length; i += chunkSize)
        {
            chunks.Add(text.Substring(i, Math.Min(chunkSize, text.Length - i)));
        }
        return chunks;
    }

    private float[] AverageVectors(List<float[]> vectors)
    {
        int length = vectors[0].Length;
        float[] average = new float[length];

        foreach (var vec in vectors)
        {
            for (int i = 0; i < length; i++)
            {
                average[i] += vec[i];
            }
        }

        for (int i = 0; i < length; i++)
        {
            average[i] /= vectors.Count;
        }

        return average;
    }

    public async Task<float[]> GetOrLoadEmbeddingAsync(string identifier, string text)
    {
        string safeName = identifier.Replace(" ", "_").Replace("/", "_");
        string path = Path.Combine("embeddings", safeName + ".json");

        if (File.Exists(path))
        {
            var Environmentjson = await File.ReadAllTextAsync(path);
            return JsonSerializer.Deserialize<float[]>(json);
        }

        float[] embedding = await GetEmbeddingAsync(text);

        Directory.CreateDirectory("embeddings");
        await File.WriteAllTextAsync(path, JsonSerializer.Serialize(embedding));

        return embedding;
     }
}
