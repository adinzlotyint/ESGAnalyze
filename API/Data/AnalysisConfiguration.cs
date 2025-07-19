using System.Text.Json.Serialization;

namespace ESGanalyzer.API.Models.Configuration {
    public class AnalysisConfiguration {
        [JsonPropertyName("Criteria3")]
        public required Dictionary<string, ScoreRule> Criteria3 { get; set; }

        [JsonPropertyName("Criteria5")]
        public required Dictionary<string, ScoreRule> Criteria5 { get; set; }

        [JsonPropertyName("Criteria9")]
        public required Dictionary<string, ScoreRule> Criteria9 { get; set; }
    }

    public class ScoreRule {
        public required PatternSet A { get; set; }

        public PatternSet? B { get; set; }
    }

    public class PatternSet {
        public required List<string> Include { get; set; }
        public List<string>? Exclude { get; set; }
    }
}