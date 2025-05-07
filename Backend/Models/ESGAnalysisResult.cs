namespace ESGanalyzer.Backend.Models {
    public class ESGAnalysisResult {
        public double C1PolicyStrategyScore { get; set; }          // [0, .5, 1]
        public double C2RiskOpportunityScore { get; set; }         // [0, .33, .67, 1]
        public double C3GovernanceStructureScore { get; set; }     // [0, .5, 1]
        public double C4EmissionsScopeScore { get; set; }          // [0, .25, .5, .75, 1]
        public double C5EmissionsBoundaryScore { get; set; }       // [0, .33, .67, 1]
        public double C6CalculationStandardScore { get; set; }     // [0, 1]
        public double C7GwpSourcesScore { get; set; }              // [0, 1]
        public double C8EmissionsTrendScore { get; set; }          // [0, .5, 1]
        public double C9IntensityIndicatorScore { get; set; }      // [0, 1]
        public double C10ReductionTargetsScore { get; set; }       // [0, .33, .5, .67, 1]
        public double C11UnitCorrectnessScore { get; set; }        // [0, 1]
        public double C12KeywordPresenceScore { get; set; }        // [0, 1]

        public double GetTotalScore() {
            return
                C1PolicyStrategyScore +
                C2RiskOpportunityScore +
                C3GovernanceStructureScore +
                C4EmissionsScopeScore +
                C5EmissionsBoundaryScore +
                C6CalculationStandardScore +
                C7GwpSourcesScore +
                C8EmissionsTrendScore +
                C9IntensityIndicatorScore +
                C10ReductionTargetsScore;
        }
    }
}
