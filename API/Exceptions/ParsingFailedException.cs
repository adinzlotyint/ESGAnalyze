namespace ESGanalyzer.API.Exceptions {
    public class ParsingFailedException : Exception {
        public ParsingFailedException() : base("Login failed. Please check your input.") { }

        public ParsingFailedException(string message) : base(message) { }

        public ParsingFailedException(string message, Exception inner) : base(message, inner) { }
    }
}
