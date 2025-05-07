using System.Text.RegularExpressions;
using System.Linq;
using System.Collections.Generic;
using ESGanalyzer.Backend.Models;

namespace ESGanalyzer.Backend.Services.Analysis.RuleBased {
    public abstract class BaseRuleAnalyzer : ICriterions {

        public abstract void Evaluate(string reportText, ESGAnalysisResult result);

        protected double MatchFirstScore(string text, IEnumerable<(string Pattern, double Score)> patterns) {
            foreach (var (pattern, score) in patterns) {
                if (Regex.IsMatch(text, pattern, RegexOptions.IgnoreCase | RegexOptions.CultureInvariant))
                    return score;
            }
            return 0.0;
        }
    }

    public class C1PolicyStrategyAnalyzer : BaseRuleAnalyzer {
        public override void Evaluate(string reportText, ESGAnalysisResult result) {
            var patterns = new List<(string, double)>
            {
                ("publicznie dostępna polityka|opisane wszystkie istotne elementy|publicly available policy|all essential elements", 1.0),
                ("polityka.*klimat|strategi.*klimat|policy.*climate|strategy.*climate", 0.5)
            };
            result.C1PolicyStrategyScore = MatchFirstScore(reportText, patterns);
        }
    }

    public class C4EmissionsScopeAnalyzer : BaseRuleAnalyzer {
        public override void Evaluate(string reportText, ESGAnalysisResult result) {
            bool s1 = Regex.IsMatch(reportText, "(scope|zakres)\\s*1", RegexOptions.IgnoreCase);
            bool s2 = Regex.IsMatch(reportText, "(scope|zakres)\\s*2", RegexOptions.IgnoreCase);
            bool s3 = Regex.IsMatch(reportText, "(scope|zakres)\\s*3", RegexOptions.IgnoreCase);
            int m = 0;
            if (Regex.IsMatch(reportText, "location[- ]?based|wg lokalizacji|według lokalizacji", RegexOptions.IgnoreCase)) m++;
            if (Regex.IsMatch(reportText, "market[- ]?based|wg rynku|według rynku", RegexOptions.IgnoreCase)) m++;

            double score = 0.0;
            if (s1 && s2 && s3 && m == 2) score = 1.0;
            else if (s1 && s2 && s3 && m == 1) score = 0.75;
            else if (s1 && s2 && m == 2) score = 0.5;
            else if (s1 && s2 && m == 1) score = 0.25;

            result.C4EmissionsScopeScore = score;
        }
    }

    public class C5EmissionsBoundaryAnalyzer : BaseRuleAnalyzer {
        public override void Evaluate(string reportText, ESGAnalysisResult result) {
            var patterns = new List<(string, double)>
            {
                ("całą gk|cała grupa|entire group|all subsidiaries|all entities", 1.0),
                ("wybranych jednostek|selected entities|selected subsidiaries", 0.67),
                ("niektórych jednostek|tylko jednostki dominującej|some subsidiaries|only parent company", 0.33)
            };
            result.C5EmissionsBoundaryScore = MatchFirstScore(reportText, patterns);
        }
    }

    public class C6CalculationStandardAnalyzer : BaseRuleAnalyzer {
        public override void Evaluate(string reportText, ESGAnalysisResult result) {
            var pattern = "ISO\\s*14064(-1)?|GHG Protocol|norma|standard";
            result.C6CalculationStandardScore = Regex.IsMatch(reportText, pattern, RegexOptions.IgnoreCase)
                ? 1.0 : 0.0;
        }
    }

    public class C7GwpSourcesAnalyzer : BaseRuleAnalyzer {
        public override void Evaluate(string reportText, ESGAnalysisResult result) {
            var pattern = "GWP|global warming potential|źródła|source.*emission factor|emission factors|wskaźnik emisji";
            result.C7GwpSourcesScore = Regex.IsMatch(reportText, pattern, RegexOptions.IgnoreCase)
                ? 1.0 : 0.0;
        }
    }

    public class C8EmissionsTrendAnalyzer : BaseRuleAnalyzer {
        public override void Evaluate(string reportText, ESGAnalysisResult result) {
            var years = Regex.Matches(reportText, "20\\d{2}");
            int distinct = years.Cast<Match>().Select(m => m.Value).Distinct().Count();
            result.C8EmissionsTrendScore = distinct >= 3 ? 1.0 : distinct == 2 ? 0.5 : 0.0;
        }
    }

    public class C9IntensityIndicatorAnalyzer : BaseRuleAnalyzer {
        public override void Evaluate(string reportText, ESGAnalysisResult result) {
            var pattern = "CO2e na|CO2e per|tCO2e|emission intensity|wskaźnik intensywności";
            result.C9IntensityIndicatorScore = Regex.IsMatch(reportText, pattern, RegexOptions.IgnoreCase)
                ? 1.0 : 0.0;
        }
    }

    public class C11UnitCorrectnessAnalyzer : BaseRuleAnalyzer {
        public override void Evaluate(string reportText, ESGAnalysisResult result) {
            result.C11UnitCorrectnessScore = Regex.IsMatch(reportText, "CO2e|CO₂e", RegexOptions.IgnoreCase)
                ? 1.0 : 0.0;
        }
    }

    public class C12KeywordPresenceAnalyzer : BaseRuleAnalyzer {
        public override void Evaluate(string reportText, ESGAnalysisResult result) {
            var keywords = new[]
            {
                "dwutlenek węgla", "gaz cieplarniany", "CO2", "zmiany klimatu",
                "carbon dioxide", "greenhouse gas", "climate change"
            };
            result.C12KeywordPresenceScore = keywords.Any(k => Regex.IsMatch(reportText, Regex.Escape(k), RegexOptions.IgnoreCase))
                ? 1.0 : 0.0;
        }
    }
}
