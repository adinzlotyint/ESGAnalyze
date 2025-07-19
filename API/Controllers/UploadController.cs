using ESGanalyzer.Shared.DTOs;
using ESGanalyzer.API.Services;
using ESGanalyzer.API.Services.Analysis;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Http.Timeouts;
using ESGanalyzer.API.Exceptions;

namespace ESGanalyzer.API.Controllers {

    [ApiController]
    //[Authorize]
    [Route("/[controller]")]
    public class UploadController : ControllerBase {
        private readonly IParseService _parseService;
        private readonly IAnalyzer _analyzer;

        public UploadController(IParseService parseService, IAnalyzer analyzer) {
            _parseService = parseService;
            _analyzer = analyzer;
        }

        [HttpPost("analyze/txt")]
        [RequestTimeout(30_000)]
        public async Task<IActionResult> AnalyzeTxt(IFormFile file) {
            if (file == null || Path.GetExtension(file.FileName)?.ToLower() != ".txt") {
                return BadRequest("Only .txt files are supported.");
            }

            string text;
            using (var reader = new StreamReader(file.OpenReadStream())) {
                text = await reader.ReadToEndAsync();
            }
            AnalysisResponse result = new();
            _analyzer.Analyze(text, result);

            return Ok(result);
        }
        [DisableRequestSizeLimit]
        [HttpPost("analyze/pdf")]
        [RequestTimeout(30_000)]
        public async Task<IActionResult> AnalyzePdf(IFormFile file) {
            if (file == null || Path.GetExtension(file.FileName)?.ToLower() != ".pdf")
                return BadRequest("Only .pdf files are supported.");

            using var stream = file.OpenReadStream();
            string text;
            try {
                text = await _parseService.ExtractTextFromPDFAsync(stream);
            } catch (ParsingFailedException e) {
                return BadRequest(e);
            }

            AnalysisResponse result = new();
            _analyzer.Analyze(text, result);

            return Ok(result);
        }
    }
    }
