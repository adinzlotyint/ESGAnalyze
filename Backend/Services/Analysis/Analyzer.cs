using ESGanalyzer.Backend.Models;

namespace ESGanalyzer.Backend.Services.Analysis {
    public class Analyzer : IAnalyzer {
        private readonly IEnumerable<ICriterions> _analyzers;

        public Analyzer(IEnumerable<ICriterions> analyzers) {
            _analyzers = analyzers.ToList();
        }

        public ESGAnalysisResult Analyze(string reportText) {
            var result = new ESGAnalysisResult();

            foreach (var analyzer in _analyzers) {
                analyzer.Evaluate(reportText, result);
            }

            return result;
        }
    }
}
