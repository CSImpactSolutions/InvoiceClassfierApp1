
using InvoiceClassifierApp.Services;
using InvoiceClassifierApp.Models;
using Microsoft.Extensions.Configuration; // Ensure this namespace is included at the top of the fileInstall - Package Microsoft.Extensions.Configuration
using System.Globalization;
using System.Text;
using System;

// Load labeled training data


// Ensure the API key is not null or empty before passing it to the OpenAIEmbeddingService constructor
var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
if (string.IsNullOrEmpty(apiKey))
{
    throw new InvalidOperationException("The OpenAI API key is not set. Please ensure the 'OPENAI_API_KEY' environment variable is configured.");
}

var openAiService = new OpenAIEmbeddingService(apiKey);
var knn = new KnnClassifier(k: 4);
var processor = new InvoiceProcessor(openAiService, knn);
var loader = new InvoiceLoader();
var trainingData = loader.LoadTrainingDataFromPdfFolders("C:\\Users\\Senthil Arumugam\\Downloads\\InvoiceClassifierApp_MVP_CleanFinal\\InvoiceClassifierApp\\TrainData");

// Pre-create output folders for all known labels
var knownLabels = trainingData.Select(t => t.Label).Distinct();
foreach (var label in knownLabels)
{
    var path = Path.Combine("output", label);
    Directory.CreateDirectory(path);
    Console.WriteLine($"Created folder for: {label}");
}

Console.WriteLine("Loading invoices...");

Console.WriteLine("\nClassifying and exporting predictions...");

// Load PDFs/images to classify
var testInvoices = loader.LoadInvoicesToClassify("C:\\Users\\Senthil Arumugam\\Downloads\\InvoiceClassifierApp_MVP\\InvoiceClassifierApp\\Invoices");

await processor.TrainAsync(trainingData);

// Classify each invoice
var output = new StringBuilder();

var csv = new StringBuilder();
csv.AppendLine("Filename,PredictedLabel");
foreach (var invoice in testInvoices)
{
    var predicted = await processor.ClassifyAsync(invoice.Filename, invoice.Text);
    Console.WriteLine($"[{invoice.Filename}] → {predicted}");
    csv.AppendLine($"\"{invoice.Filename}\",\"{predicted}\"");
    var sourceInvoicePath = "C:\\Users\\Senthil Arumugam\\Downloads\\InvoiceClassifierApp_MVP_CleanFinal\\InvoiceClassifierApp\\Invoices\\"+ invoice.Filename;
    var targetFolder = Path.Combine("output", predicted);
    var targetInvoicePath = Path.Combine(targetFolder, invoice.Filename);

    Directory.CreateDirectory(targetFolder);

    if (File.Exists(sourceInvoicePath))
    {
        File.Copy(sourceInvoicePath, targetInvoicePath, overwrite: true);
        Console.WriteLine($"Copied {invoice.Filename} to {targetFolder}");
    }
    else
    {
        Console.WriteLine($"File not found at: {sourceInvoicePath}");

        // Try fallback match by filename
        var invoiceFolder = Path.GetDirectoryName(sourceInvoicePath);
        var fallbackPath = Directory.GetFiles(invoiceFolder)
            .FirstOrDefault(f => Path.GetFileName(f).Equals(invoice.Filename, StringComparison.OrdinalIgnoreCase));

        if (fallbackPath != null)
        {
            File.Copy(fallbackPath, targetInvoicePath, overwrite: true);
            Console.WriteLine($"Fallback copy from: {fallbackPath}");
        }
        else
        {
            Console.WriteLine("Could not find matching file even by fallback.");
            Console.WriteLine("Available files in source folder:");
            foreach (var f in Directory.GetFiles(invoiceFolder))
            {
                Console.WriteLine(" - " + Path.GetFileName(f));
            }
        }
    }

    Directory.CreateDirectory(targetFolder);

    if (File.Exists(sourceInvoicePath))
    {
        File.Copy(sourceInvoicePath, targetInvoicePath, overwrite: true);
        Console.WriteLine($"Copied {invoice.Filename} to {targetFolder}");
    }
    else
    {
        Console.WriteLine($"File not found: {sourceInvoicePath}");
    }

}


Directory.CreateDirectory("output");
var csvPath = "output/predictions.csv";
var analyzer = new EmbeddingSimilarityAnalyzer(@"C:\Users\Senthil Arumugam\Downloads\InvoiceClassifierApp_MVP_CleanFinal\InvoiceClassifierApp\bin\Debug\net9.0\embeddings");
analyzer.Analyze(@"C:\Users\Senthil Arumugam\Downloads\InvoiceClassifierApp_MVP_CleanFinal\InvoiceClassifierApp\bin\Debug\net9.0\embeddings\SimilarityResults.csv");
var exporter = new EmbeddingSimilarityMatrixExporter(@"C:\Users\Senthil Arumugam\Downloads\InvoiceClassifierApp_MVP_CleanFinal\InvoiceClassifierApp\bin\Debug\net9.0\embeddings");
exporter.ExportMatrix(@"C:\Users\Senthil Arumugam\Downloads\InvoiceClassifierApp_MVP_CleanFinal\InvoiceClassifierApp\bin\Debug\net9.0\embeddings\SimilarityMatrix.csv");
await File.WriteAllTextAsync(csvPath, csv.ToString());
Console.WriteLine($"\nPredictions saved to: {csvPath}");
Console.WriteLine("Zipping classified folders...");
foreach (var label in knownLabels)
{
    var folderPath = Path.Combine("output", label);
    var zipPath = Path.Combine("output", label + ".zip");
    if (Directory.Exists(folderPath))
    {
        if (File.Exists(zipPath))
        {
            File.Delete(zipPath);
        }
        System.IO.Compression.ZipFile.CreateFromDirectory(folderPath, zipPath);
        Console.WriteLine($"Created: {zipPath}");
    }
}

