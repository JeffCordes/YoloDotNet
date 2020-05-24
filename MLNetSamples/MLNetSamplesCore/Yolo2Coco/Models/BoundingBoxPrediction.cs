using System;
using System.Collections.Generic;
using System.Text;

namespace MLNetSamples.Yolo2Coco.Models
{
    public class BoundingBoxPrediction : BoundingBoxDimensions
    {
        public float Confidence { get; set; }
    }
}
