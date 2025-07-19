using Microsoft.AspNetCore.Http;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ESGanalyzer.API.Services;
using FluentAssertions;
using ESGanalyzer.API.Exceptions;

namespace ESGAnalyzer.Tests.Unit {
    public class ParseServiceTests {
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
    }
}
