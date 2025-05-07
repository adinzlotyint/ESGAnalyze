using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Wordprocessing;
using System.Text;
using UglyToad.PdfPig;
using UglyToad.PdfPig.Content;

namespace ESGanalyzer.Backend.Services {
    public class ParseService : IParseService {
        public async Task<string> ExtractTextFromDOCXAsync(IFormFile file) {
            using var stream = file.OpenReadStream();
            using var ms = new MemoryStream();
            await stream.CopyToAsync(ms);

            ms.Seek(0, SeekOrigin.Begin);

            using var doc = WordprocessingDocument.Open(ms, false);
            var body = doc.MainDocumentPart?.Document?.Body;
            if (body == null) return string.Empty;

            var paragraphs = body.Descendants<Paragraph>();
            var text = string.Join(Environment.NewLine, paragraphs.Select(p => p.InnerText));

            return text;
        }

        public async Task<string> ExtractTextFromPDFAsync(IFormFile file) {
            using var stream = file.OpenReadStream();
            using var pdf = PdfDocument.Open(stream);

            var builder = new StringBuilder();
            foreach (Page page in pdf.GetPages()) {
                builder.AppendLine(page.Text);
            }

            return builder.ToString();
        }
        }
}
