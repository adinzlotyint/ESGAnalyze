using ESGanalyzer.Shared.DTOs;

namespace ESGanalyzer.API.Services.Analysis {
    public class Analyzer : IAnalyzer {
        private readonly ICriterias _criterias;

        public Analyzer(ICriterias criterias) {
            _criterias = criterias;
        }
        public AnalysisResponse Analyze(string reportText, AnalysisResponse result) {
            _criterias.Evaluate(reportText, result);
            return result;
        }
    }
}
