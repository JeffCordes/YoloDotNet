using System;
using System.Collections.Generic;
using System.Text;

namespace MLNetSamples.Yolo2Coco.Models.Output
{
    public interface IYoloObjectPrediction
    {
        float[] PredictedLabels { get; set; }
    }
}
