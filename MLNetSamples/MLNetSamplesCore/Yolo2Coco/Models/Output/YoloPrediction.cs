using Microsoft.ML.Data;

namespace MLNetSamples.Yolo2Coco.Models.Output
{
    public class YoloPrediction : IYoloObjectPrediction
    {
        [ColumnName("grid")]
        public float[] PredictedLabels { get; set; }
    }
}
