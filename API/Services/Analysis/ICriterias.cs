using ESGanalyzer.Shared.DTOs;
using ESGanalyzer.API.Models.Configuration;

namespace ESGanalyzer.API.Services.Analysis {
    public interface ICriterias {
        void Evaluate(string reportText, AnalysisResponse result);
    }
}
