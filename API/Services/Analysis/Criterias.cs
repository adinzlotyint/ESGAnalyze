using ESGanalyzer.Shared.DTOs;
using ESGanalyzer.Backend.Models.Configuration;
using System.Globalization;
using System.Text.RegularExpressions;

namespace ESGanalyzer.Backend.Services.Analysis {
    public class Criterias : ICriterias {
        private readonly AnalysisConfiguration _config;
        private readonly ILogger<Criterias> _logger;

        public Criterias(AnalysisConfiguration config, ILogger<Criterias> logger) {
            _config = config;
            _logger = logger;
        }

        public void Evaluate(string reportText, AnalysisResponse result) {
            result.Criteria3 = EvaluateCriterion(reportText, _config.Criteria3);
            result.Criteria5 = EvaluateCriterion(reportText, _config.Criteria5);
            result.Criteria9 = EvaluateCriterion(reportText, _config.Criteria9);
        }
        private double EvaluateCriterion(string text, Dictionary<string, ScoreRule> criterionRules) {
            foreach (var score in criterionRules) {
                var rule = score.Value;
                bool conditionA = DoesPatternSetMatch(text, rule.A);
                bool conditionB = rule.B != null && DoesPatternSetMatch(text, rule.B);
                _logger.LogInformation("Score key: {ScoreKey}, conditionA: {conditionA}, conditionB: {conditionB}", score.Key, conditionA, conditionB);

                if (conditionA || conditionB) {
                    return Double.Parse(score.Key, CultureInfo.InvariantCulture);
                }
            }

            return 0.0;
        }
        private bool DoesPatternSetMatch(string text, PatternSet patternSet) {
            return CheckAllPatternsMatch(text, patternSet.Include) &&
                   CheckNoPatternsMatch(text, patternSet.Exclude);
        }

        private bool CheckAllPatternsMatch(string text, List<string> patterns) {
            if (patterns == null || !patterns.Any()) return true;
            return patterns.All(pattern => Regex.IsMatch(text, pattern, RegexOptions.IgnoreCase | RegexOptions.Singleline));
        }

        private bool CheckNoPatternsMatch(string text, List<string>? patterns) {
            if (patterns == null || !patterns.Any()) return true;
            return !patterns.Any(pattern => Regex.IsMatch(text, pattern, RegexOptions.IgnoreCase | RegexOptions.Singleline));
        }
    }
}