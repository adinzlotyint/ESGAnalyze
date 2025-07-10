using ESGanalyzer.Shared.DTOs;
using ESGanalyzer.Backend.Models.Configuration;

namespace ESGanalyzer.Backend.Services.Analysis {
    public interface ICriterias {
        void Evaluate(string reportText, AnalysisResponse result);
    }
}
