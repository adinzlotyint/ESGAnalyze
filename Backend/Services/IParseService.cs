namespace ESGanalyzer.Backend.Services {
    public interface IParseService {
        Task<string> ExtractTextFromDOCXAsync(IFormFile file);
        Task<string> ExtractTextFromPDFAsync(IFormFile file);
    }
}
