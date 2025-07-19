using ESGanalyzer.Shared.DTOs;

namespace ESGanalyzer.API.Services.Analysis {
    public interface IAnalyzer {
        public AnalysisResponse Analyze(string reportText, AnalysisResponse result);
    }
}
