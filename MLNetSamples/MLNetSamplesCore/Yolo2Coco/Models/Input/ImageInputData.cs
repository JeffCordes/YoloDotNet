using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;
using Microsoft.ML.Transforms.Image;

namespace MLNetSamples.Yolo2Coco.Models.Input
{
    public class ImageInputData
    {
        [ImageType(416, 416)]
        public Bitmap Image { get; set; }
    }
}
