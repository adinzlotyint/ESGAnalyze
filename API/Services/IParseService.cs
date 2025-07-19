namespace ESGanalyzer.API.Services {
    public interface IParseService {
        Task<string> ExtractTextFromPDFAsync(Stream fileContent);
    }
}
