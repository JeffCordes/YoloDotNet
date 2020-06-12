using System.Collections.Generic;
using System.Drawing;

namespace YoloDotNetApi.Service
{
    public interface IYoloInferenceService
    {
        public float[] GetTensors(Image originalImage);

        public List<Models.YoloPrediction> ProcessData(float[] data);

        public List<Models.YoloPrediction> FilterTopResults(List<Models.YoloPrediction> predictions);
    }
}
