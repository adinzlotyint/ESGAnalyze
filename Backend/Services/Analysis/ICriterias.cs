using ESGanalyzer.Backend.DTOs;

namespace ESGanalyzer.Backend.Services.Analysis {

    public interface ICriterias {
        void Evaluate(string reportText, AnalysisResponse result);
    }
}
