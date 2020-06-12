using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using YoloDotNetApi.Service;
using YoloDotNetApi.Utilities;

namespace YoloDotNetApi.Controllers
{
    [Route("api/Yolo")]
    [ApiController]
    public class YoloDetectionController : ControllerBase
    {
        private readonly IYoloInferenceService _service;

        public YoloDetectionController(IYoloInferenceService service)
        {
            _service = service;
        }

        [HttpGet]
        [Route("status")]
        public async Task<IActionResult> Status()
        {
            return Ok();
        }

        [HttpPost]
        [Route("tensors")]
        public async Task<IActionResult> GetTensors(IFormFile file)
        {
            if (file == null)
            {
                return new BadRequestObjectResult("no file provided");
            }


            using (var stream = new MemoryStream())
            {
                file.CopyTo(stream);
                using (Bitmap bit = new Bitmap(stream))
                {
                    var tensors = _service.GetTensors(bit);
                    return new OkObjectResult(tensors);
                }
            }
        }

        [HttpPost]
        [Route("predictions")]
        public async Task<IActionResult> GetPredictions(IFormFile file)
        {
            if (file == null)
            {
                return new BadRequestObjectResult("no file provided");
            }


            using (var stream = new MemoryStream())
            {
                file.CopyTo(stream);
                using (Bitmap bit = new Bitmap(stream))
                {
                    var tensors = _service.GetTensors(bit);
                    var predictions = _service.ProcessData(tensors);
                    return new OkObjectResult(predictions);
                }
            }
        }

        [HttpPost]
        [Route("topResults")]
        public async Task<IActionResult> GetTopResults(IFormFile file)
        {
            if (file == null)
            {
                return new BadRequestObjectResult("no file provided");
            }


            using (var stream = new MemoryStream())
            {
                file.CopyTo(stream);
                using (Bitmap bit = new Bitmap(stream))
                {
                    var tensors = _service.GetTensors(bit);
                    var predictions = _service.ProcessData(tensors);
                    var topResults = _service.FilterTopResults(predictions);
                    return new OkObjectResult(topResults);
                }
            }
        }

        [HttpPost]
        [Route("Preview")]
        public async Task<IActionResult> GetPreview(IFormFile file)
        {
            if (file == null)
            {
                return new BadRequestObjectResult("no file provided");
            }


            using (var stream = new MemoryStream())
            {
                file.CopyTo(stream);
                using (Bitmap bit = new Bitmap(stream))
                {
                    var tensors = _service.GetTensors(bit);
                    var predictions = _service.ProcessData(tensors);
                    var topResults = _service.FilterTopResults(predictions);
                    var preview = ImageUtils.DrawYoloBoxes(bit, topResults);

                    MemoryStream ms = new MemoryStream();
                    preview.Save(ms, ImageFormat.Jpeg);
                    return File(ms.ToArray(), "image/jpeg");
                }
            }
        }
    }
}
