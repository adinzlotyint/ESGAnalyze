namespace ESGanalyzer.Backend.Models {
    public class AnalysisReport {
        public int Id { get; set; }
        public required string ReportName { get; set; }
        public required string UserId { get; set; }
        public double Criteria1 { get; set; }
        public double Criteria2 { get; set; }
        public double Criteria3 { get; set; }
        public double Criteria4 { get; set; }
        public double Criteria5 { get; set; }
        public double Criteria6 { get; set; }
        public double Criteria7 { get; set; }
        public double Criteria8 { get; set; }
        public double Criteria9 { get; set; }
    }
}
