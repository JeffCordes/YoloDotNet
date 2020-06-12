using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using JsonIgnoreAttribute = Newtonsoft.Json.JsonIgnoreAttribute;

namespace YoloDotNetApi.Models
{
    public class YoloPrediction : IComparable
    {
        [JsonProperty("topPrediction")]
        public ClassPrediction TopPrediction { get; set; }
        [JsonProperty("containsObjectConfidence")]
        public float ContainsObjectConfidence { get; set; }
        [JsonProperty("gridX")]
        public int GridX { get; set; }
        [JsonProperty("gridY")]
        public int GridY { get; set; }
        [JsonProperty("anchor")]
        public int Anchor { get; set; }
        [JsonProperty("box")]
        public Rectangle Box { get; set; }
        [JsonProperty("originalTensor")]
        public float[] OriginalTensor { get; set; }
        [JsonProperty("classPredictions")]
        public List<ClassPrediction> ClassPredictions { get; set; }

        [JsonIgnore]
        private readonly float cellWidth;
        [JsonIgnore]
        private readonly float cellHeight;
        [JsonIgnore]
        private readonly (float x, float y)[] BoxAnchors;
        [JsonIgnore]
        private readonly string[] ClassLabels;
        [JsonIgnore]
        private readonly Yolo2CoCoSettings Settings;

        public YoloPrediction(float[] data, int column, int row, int anchor, Yolo2CoCoSettings settings)
        {
            this.ClassLabels = CoCoModel.Labels;
            this.BoxAnchors = CoCoModel.Anchors;
            this.ClassPredictions = new List<ClassPrediction>();
            this.Settings = settings;
            cellHeight = settings.InputHeight / settings.GridHeight;
            cellWidth = settings.InputWidth / settings.GridWidth;
            

            this.GridX = column;
            this.GridY = row;
            this.Anchor = anchor;
            this.OriginalTensor = data;
            this.ContainsObjectConfidence = Sigmoid(data[4]);

            Rectangle rect = new Rectangle();
            rect.X = (int)decimal.Multiply((decimal)(Sigmoid(data[0]) + GridX), (decimal)cellWidth);
            rect.Y = (int)decimal.Multiply((decimal)(Sigmoid(data[1]) + GridY), (decimal)cellHeight);
            rect.Width = (int)(MathF.Exp(data[2]) * cellWidth * BoxAnchors[anchor].x);
            rect.Height = (int)(MathF.Exp(data[3]) * cellHeight * BoxAnchors[anchor].y);

            rect.X = (int)(rect.X - (float)decimal.Divide((decimal)rect.Width, 2));
            rect.Y = (int)(rect.Y - (float)decimal.Divide((decimal)rect.Height, 2));
            this.Box = rect;

            var rawClassProbabilities = data.Skip(5).ToArray();
            var calculatedProbablities = ExtractClassProbabilities(rawClassProbabilities, data[4]);
            for (int i = 0; i < calculatedProbablities.Length; i++)
            {
                ClassPredictions.Add(new ClassPrediction()
                {
                    Label = ClassLabels[i],
                    Confidence = calculatedProbablities[i]
                });
            }

            this.TopPrediction = ClassPredictions.Max();
        }

        /// <summary>
        /// Applies the sigmoid function that outputs a number between 0 and 1.
        /// </summary>
        /// <param name="value">float value</param>
        /// <returns>sigmoid of float</returns>
        private float Sigmoid(float value)
        {
            return 1.0f / (1.0f + (float)MathF.Exp(-value));
        }

        /// <summary>
        /// Get calculated probabilities
        /// </summary>
        /// <param name="rawClassPredictions">the raw data</param>
        /// <param name="confidence">main object detection confidence</param>
        /// <returns>calculated probability</returns>
        public float[] ExtractClassProbabilities(float[] rawClassPredictions, float confidence)
        {
            return Softmax(rawClassPredictions).Select(p => p * confidence).ToArray();
        }

        /// <summary>
        /// Normalization with softmax
        /// </summary>
        /// <param name="classProbabilities"></param>
        /// <returns></returns>
        private float[] Softmax(float[] classProbabilities)
        {
            var exp = classProbabilities.Select(MathF.Exp);
            var sum = exp.Sum();
            var softmax = exp.Select(i => i / sum);
            return softmax.ToArray();
        }

        /// <summary>
        /// custom comparer to compare by confidence
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public int CompareTo(object other)
        {
            return TopPrediction.Confidence.CompareTo(((YoloPrediction)other).TopPrediction.Confidence);
        }
    }

}
