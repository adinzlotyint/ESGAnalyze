using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Wordprocessing;
using System.Text;
using UglyToad.PdfPig;
using UglyToad.PdfPig.Content;

namespace ESGanalyzer.Backend.Services {
    public class ParseService : IParseService {
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
