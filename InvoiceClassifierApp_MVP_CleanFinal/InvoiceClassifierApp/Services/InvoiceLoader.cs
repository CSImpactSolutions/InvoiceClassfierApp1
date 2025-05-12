
using InvoiceClassifierApp.Models;
using iText.Kernel.Pdf;
using iText.Kernel.Pdf.Canvas.Parser;
using iText.Kernel.Pdf.Canvas.Parser.Listener;
using Tesseract;

namespace InvoiceClassifierApp.Services;

public class InvoiceLoader
{
   

        string trainingPath = Path.Combine(AppContext.BaseDirectory, "TrainData");
    public List<InvoiceVector> LoadTrainingDataFromPdfFolders(string trainDataRoot)
    {
        var data = new List<InvoiceVector>();

        foreach (var dir in Directory.GetDirectories(trainDataRoot))
        {
            var label = Path.GetFileName(dir);

            foreach (var file in Directory.GetFiles(dir))
            {
                var ext = Path.GetExtension(file).ToLower();
                string text = "";

                if (ext == ".pdf")
                    text = ExtractTextFromPdf(file);  // Your existing method
                else if (ext == ".txt")
                    text = File.ReadAllText(file);
                else
                    continue;

                data.Add(new InvoiceVector
                {
                    Filename = Path.GetFileName(file),
                    Label = label.ToLower(),
                    Text = text
                });
            }
        }

        return data;
    }
  


    public List<InvoiceVector> LoadInvoicesToClassify(string invoiceDir)
    {
        var invoices = new List<InvoiceVector>();
        if (!Directory.Exists(invoiceDir)) return invoices;

        foreach (var file in Directory.GetFiles(invoiceDir))
        {
            string text = "";
            var ext = Path.GetExtension(file).ToLower();

            if (ext == ".pdf")
                text = ExtractTextFromPdf(file);

            invoices.Add(new InvoiceVector
            {
                Filename = Path.GetFileName(file),
                Label = "unlabeled",
                Text = text
            });
        }

        return invoices;
    }

    private string ExtractTextFromPdf(string path)
    {
        using var reader = new PdfReader(path);
        using var doc = new PdfDocument(reader);
        string result = "";
        for (int i = 1; i <= doc.GetNumberOfPages(); i++)
        {
            var page = doc.GetPage(i);
            var strategy = new SimpleTextExtractionStrategy();
            result += PdfTextExtractor.GetTextFromPage(page, strategy) + "\n";
        }
        return result;
    }

    private string ExtractTextFromImage(string path)
    {
        using var engine = new TesseractEngine("./tessdata", "eng", EngineMode.Default);
        using var img = Pix.LoadFromFile(path);
        using var page = engine.Process(img);
        return page.GetText();
    }
}
