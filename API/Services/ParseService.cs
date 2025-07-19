using ESGanalyzer.API.Exceptions;
using Microsoft.IdentityModel.Tokens;
using System.Text;
using System.Text.RegularExpressions;
using UglyToad.PdfPig;
using UglyToad.PdfPig.Content;

namespace ESGanalyzer.API.Services {
    public class ParseService : IParseService {
        public async Task<string> ExtractTextFromPDFAsync(Stream fileContent) {
            return await Task.Run(() => {
                if (fileContent == null || fileContent.Length == 0) {
                    throw new ParsingFailedException();
                }
                using var pdf = PdfDocument.Open(fileContent);

                var builder = new StringBuilder();
                foreach (Page page in pdf.GetPages()) {
                    builder.AppendLine(page.Text);
                }

                string rawText = builder.ToString();
                string cleanedText = Regex.Replace(rawText, @"\s+", " ").Trim();

                return cleanedText.ToString();
            });
        }
    }
}
