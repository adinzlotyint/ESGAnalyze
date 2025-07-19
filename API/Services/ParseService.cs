using System.Text;
using System.Text.RegularExpressions;
using UglyToad.PdfPig;
using UglyToad.PdfPig.Content;

namespace ESGanalyzer.Backend.Services {
    public class ParseService : IParseService {
        public async Task<string> ExtractTextFromPDFAsync(IFormFile file) {
            return await Task.Run(() => {
                using var stream = file.OpenReadStream();
                using var pdf = PdfDocument.Open(stream);

                var builder = new StringBuilder();
                foreach (Page page in pdf.GetPages()) {
                    builder.AppendLine(page.Text);
                }

                string rawText = builder.ToString();
                string cleanedText = Regex.Replace(rawText, @"\s+", " ").Trim();

                return builder.ToString();
            });
        }
    }
}
