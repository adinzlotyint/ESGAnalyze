using ESGanalyzer.Backend.DTOs;

namespace ESGanalyzer.Backend.Services.Analysis {
    public interface IAnalyzer {
        public AnalysisResponse Analyze(string reportText, AnalysisResponse result);
    }
}
