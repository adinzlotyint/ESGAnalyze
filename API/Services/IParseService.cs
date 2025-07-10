namespace ESGanalyzer.Backend.Services {
    public interface IParseService {
        Task<string> ExtractTextFromPDFAsync(IFormFile file);
    }
}
