using ESGanalyzer.Backend.Models;

namespace ESGanalyzer.Backend.Services.Analysis {

    public interface ICriterions {
        void Evaluate(string reportText, ESGAnalysisResult result);
    }
}
