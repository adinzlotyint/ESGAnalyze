using ESGanalyzer.Backend.Models;
using ESGanalyzer.Backend.DTOs;
using ESGanalyzer.Backend.Services;
using ESGanalyzer.Backend.Services.Analysis;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Http.Timeouts;

namespace ESGanalyzer.Backend.Controllers {

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

            string text = await _parseService.ExtractTextFromPDFAsync(file);
            AnalysisResponse result = new();
            _analyzer.Analyze(text, result);

            return Ok(result);
        }
    }
    }
