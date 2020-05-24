using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using MLNetSamples.Yolo2Coco.Service;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MLNetSamplesApi.Controllers
{
    [Route("api/Yolo")]
    [ApiController]
    public class YoloDetectionController : ControllerBase
    {
        private readonly ILogger<YoloDetectionController> _logger;
        private readonly IYoloDetectionService _yoloDetectionService;

        public YoloDetectionController(IYoloDetectionService yoloDetectionService, ILogger<YoloDetectionController> logger)
        {
            _logger = logger;
            _yoloDetectionService = yoloDetectionService;
        }

        [HttpGet]
        [Route("status")]
        public async Task<IActionResult> Status()
        {
            return Ok();
        }

        [HttpPost]
        [Route("DetectionResults")]
        public async Task<IActionResult> IdentifyObjects(IFormFile imageFile)
        {
            return Ok();
        }
    }
}
