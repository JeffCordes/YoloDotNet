using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using YoloDotNetApi.Models;
using YoloDotNetApi.Utilities;
using Image = System.Drawing.Image;

namespace YoloDotNetApi.Service
{
    public class YoloInferenceService : IYoloInferenceService
    {
        private readonly InferenceSession _inferenceSession;
        private readonly Yolo2CoCoSettings _yolo2CoCoSettings;

        /// <summary>
        /// Default Constructor
        /// </summary>
        /// <param name="modelPath">Relative path to the onnx model file</param>
        public YoloInferenceService(Yolo2CoCoSettings yolo2CoCoSettings)
        {
            _yolo2CoCoSettings = yolo2CoCoSettings;
            _inferenceSession = new InferenceSession(_yolo2CoCoSettings.Model, new SessionOptions());
        }

        /// <summary>
        /// Get the input tensor from an image
        /// </summary>
        /// <param name="originalImage"></param>
        /// <returns></returns>
        public float[] GetTensors(Image image)
        {
            // resize the input image to match the model's input size
            Bitmap resizedImage = ImageUtils.ResizeImage(image, _yolo2CoCoSettings.InputWidth, _yolo2CoCoSettings.InputHeight);

            // convert the image to an array of floats, and convert to a tensor input for the model
            var bitData = ImageUtils.GetNormalizedFloatArray(resizedImage);
            var Inputs = new List<NamedOnnxValue>();
            var tensor = new DenseTensor<float>(bitData, _inferenceSession.InputMetadata["input.1"].Dimensions);
            Inputs.Add(NamedOnnxValue.CreateFromTensor<float>("input.1", tensor));

            // run the inference and convert the Idisposable result back to a float[]
            float[] resultTensor;
            using (var results = _inferenceSession.Run(Inputs))
            {
                resultTensor = results.FirstOrDefault().AsTensor<float>().ToArray();
            }

            // return
            return resultTensor;
        }

        /// <summary>
        /// Convert ray output data to a list of YoloPrediction objects
        /// </summary>
        /// <param name="data">Raw output data from inference</param>
        /// <returns>A list of YoloPredictions</returns>
        public List<Models.YoloPrediction> ProcessData(float[] data)
        {
            // the data is currently out of order, this is the equivalant of numpy.reshape() in python examples
            // convert from 1*425*13*13 t0 1*13*13*425
            var tensorLength = (5 + _yolo2CoCoSettings.ClassCount);
            var columnSpan = tensorLength * _yolo2CoCoSettings.Anchors;
            var rowSpan = columnSpan * _yolo2CoCoSettings.GridWidth;
            float[] target = new float[data.Length];
            int i = 0;
            for (int x = 0; x < columnSpan; x++)
            {
                for(int column = 0; column < _yolo2CoCoSettings.GridWidth; column++)
                {
                    for (int row = 0; row < _yolo2CoCoSettings.GridHeight; row++)
                    {
                        var targetIndex = (row * columnSpan) + (column * rowSpan) + x;
                        target[targetIndex] = data[i];
                        i++;
                    }
                }
            }

            // calculate a prediction for each box/grid combo
            List<Models.YoloPrediction> results = new List<Models.YoloPrediction>();
            for (int column = 0; column < _yolo2CoCoSettings.GridWidth; column++)
            {
                for (int row = 0; row < _yolo2CoCoSettings.GridHeight; row++)
                {
                    for (int box = 0; box < _yolo2CoCoSettings.Anchors; box++)
                    {
                        int startingIndex = (row * rowSpan) + (column * columnSpan) + (box * tensorLength);
                        var cellData = target.Skip(startingIndex).Take(tensorLength).ToArray();
                        if (cellData.Length != tensorLength) throw new Exception();
                        var prediction = new Models.YoloPrediction(cellData, column, row, box, _yolo2CoCoSettings);
                        results.Add(prediction);
                    }

                }
            }

            return results;
        }

        /// <summary>
        /// Filter the top results by confidence, and intersection over union
        /// </summary>
        /// <param name="predictions">a list of yolo predictions</param>
        /// <returns>the filtered results</returns>
        public List<Models.YoloPrediction> FilterTopResults(List<Models.YoloPrediction> predictions)
        {
            var overConfidenceLimit = (from p in predictions where p.ContainsObjectConfidence > _yolo2CoCoSettings.ConfidenceLimit select p).ToList();
            //var overConfidenceLimit = predictions;

            // Intersection Over Union
            if (_yolo2CoCoSettings.IntersectionOverUnion)
            {
                // IOU is unabled, calculate % overlap betweeen predictions with same label.  Trim boxes to remove excess overlap.
                // this makes the prediction boxes tighter to object.
                List<YoloPrediction> duplicates = new List<YoloPrediction>();
                for (int i = 0; i < overConfidenceLimit.Count; i++)
                {
                    var rootPrediction = overConfidenceLimit[i];
                    foreach (var prediction in overConfidenceLimit)
                    {
                        if (rootPrediction.TopPrediction.Label == prediction.TopPrediction.Label)
                        {
                            if (IntersectionOverUnion(rootPrediction.Box, prediction.Box) > _yolo2CoCoSettings.IOUOverlapLimit)
                            {
                                overConfidenceLimit[i].Box = InterSectionOverUnionOverlap(rootPrediction.Box, prediction.Box);
                            }
                        }
                    }
                }

                // remove exact duplicates
                var distinctItems = overConfidenceLimit.GroupBy(x => x.Box).Select(y => y.First()).ToList();

                // Sort and return top results
                distinctItems.Sort();
                distinctItems.Reverse();
                if (distinctItems.Count > _yolo2CoCoSettings.MaxResults)
                {
                    distinctItems = distinctItems.Take(_yolo2CoCoSettings.MaxResults).ToList();
                }
                return distinctItems;
            }
            else
            {
                // no IOU, just sort and take top results
                overConfidenceLimit.Sort();
                overConfidenceLimit.Reverse();
                if (overConfidenceLimit.Count > _yolo2CoCoSettings.MaxResults)
                {
                    overConfidenceLimit = overConfidenceLimit.Take(_yolo2CoCoSettings.MaxResults).ToList();
                }
                return overConfidenceLimit;
            }
        }

        

        private float IntersectionOverUnion(Rectangle boundingBoxA, Rectangle boundingBoxB)
        {
            var areaA = boundingBoxA.Width * boundingBoxA.Height;
            var areaB = boundingBoxB.Width * boundingBoxB.Height;

            if (areaA <= 0 || areaB <= 0)
                return 0;

            var minX = Math.Max(boundingBoxA.Left, boundingBoxB.Left);
            var minY = Math.Max(boundingBoxA.Top, boundingBoxB.Top);
            var maxX = Math.Min(boundingBoxA.Right, boundingBoxB.Right);
            var maxY = Math.Min(boundingBoxA.Bottom, boundingBoxB.Bottom);

            var intersectionArea = MathF.Max(maxY - minY, 0) * MathF.Max(maxX - minX, 0);

            return intersectionArea / (areaA + areaB - intersectionArea);
        }

        private Rectangle InterSectionOverUnionOverlap(Rectangle boundingBoxA, Rectangle boundingBoxB)
        {
            return new Rectangle()
            {
                X = Math.Max(boundingBoxA.Left, boundingBoxB.Left),
                Y = Math.Max(boundingBoxA.Top, boundingBoxB.Top),
                Width = Math.Min(boundingBoxA.Right, boundingBoxB.Right) - Math.Max(boundingBoxA.Left, boundingBoxB.Left),
                Height = Math.Min(boundingBoxA.Bottom, boundingBoxB.Bottom) - Math.Max(boundingBoxA.Top, boundingBoxB.Top)
            };
        }

    }
}
