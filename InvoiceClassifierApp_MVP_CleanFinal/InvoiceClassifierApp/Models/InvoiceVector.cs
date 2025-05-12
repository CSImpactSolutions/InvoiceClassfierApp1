namespace InvoiceClassifierApp.Models;

public class InvoiceVector
{
    public string Filename { get; set; }
    public string Label { get; set; }
    public string Text { get; set; }
    public float[] Vector { get; set; }
    public string SourcePath { get; set; }
}