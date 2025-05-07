using ESGanalyzer.Backend.Models;
using ESGanalyzer.Backend.Services;
using ESGanalyzer.Backend.Services.Analysis;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ESGanalyzer.Backend.Controllers {

    [ApiController]
    [Authorize]
    [Route("/[controller]")]
    public class UploadController : ControllerBase {
        private readonly IParseService _parseService;
        private readonly IAnalyzer _analyzer;

        public UploadController(IParseService parseService, IAnalyzer analyzer) {
            _parseService = parseService;
            _analyzer = analyzer;
        }

        [HttpPost("analyze/docx")]
        public async Task<IActionResult> AnalyzeDocx(IFormFile file) {
            if (file == null || Path.GetExtension(file.FileName)?.ToLower() != ".docx") {
                return BadRequest("Only .docx files are supported.");
            }

            string text = await _parseService.ExtractTextFromDOCXAsync(file);
            ESGAnalysisResult result = _analyzer.Analyze(text);
            return Ok(result);
        }
        [DisableRequestSizeLimit]
        [HttpPost("analyze/pdf")]
        public async Task<IActionResult> AnalyzePdf(IFormFile file) {
            if (file == null || Path.GetExtension(file.FileName)?.ToLower() != ".pdf")
                return BadRequest("Only .pdf files are supported.");

            string text = await _parseService.ExtractTextFromPDFAsync(file);
            ESGAnalysisResult result = _analyzer.Analyze(text);
            return Ok(result);
        }
    }
    }
