using Microsoft.AspNetCore.Http;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ESGanalyzer.API.Services;
using FluentAssertions;
using ESGanalyzer.API.Exceptions;
using Xunit.Abstractions;

namespace ESGAnalyzer.Tests.Unit {
    public class ParseServiceTests {
        private readonly ITestOutputHelper _output;
        public ParseServiceTests(ITestOutputHelper output) {
            _output = output;
        }

        [Theory]
        [InlineData(null)]
        public async Task ExtractTextFromPDFAsync_WithInvalidInput_ReturnsException(Stream fileContent) {
            //Arrange
            var parseService = new ParseService();
            //Act
            var act = () => parseService.ExtractTextFromPDFAsync(fileContent);
            //Assert
            await act.Should().ThrowAsync<ParsingFailedException>();
        }

        [Fact]
        public async Task ExtractTextFromPDFAsync_WithValidInput_ReturnsString() {
            //Arrange
            var parseService = new ParseService();
            var filePath = Path.Combine(AppContext.BaseDirectory, "TestAssets", "Valid-document.pdf");
            await using var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
            //Act
            var result = await parseService.ExtractTextFromPDFAsync(fileStream);
            //Assert
            result.Should().BeOfType<String>();
            result.Should().Be("Valid pdf");
        }

        [Fact]
        public async Task ExtractTextFromPDFAsync_WithValidEmptyPdf_ReturnsEmptyString() {
            //Arrange
            var parseService = new ParseService();
            var filePath = Path.Combine(AppContext.BaseDirectory, "TestAssets", "Empty-document.pdf");
            await using var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
            //Act
            var result = await parseService.ExtractTextFromPDFAsync(fileStream);

            // Assert
            result.Should().BeEmpty();
        }
    }


}
