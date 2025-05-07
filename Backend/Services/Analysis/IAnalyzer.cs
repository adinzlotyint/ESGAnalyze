using ESGanalyzer.Backend.Models;

namespace ESGanalyzer.Backend.Services.Analysis {
    public interface IAnalyzer {
        public ESGAnalysisResult Analyze(string text);
    }
}
