using Microsoft.ML;
using Microsoft.ML.Transforms.Image;
using MLNetSamples.Yolo2Coco.Models.Input;
using MLNetSamples.Yolo2Coco.Models.Output;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNetSamples.Yolo2Coco.Models
{
    public class YoloModelConfigurator
    {
        private readonly MLContext mlContext;
        private readonly ITransformer mlModel;

        public YoloModelConfigurator(IYoloModel yoloModel)
        {
            mlContext = new MLContext();
            // Model creation and pipeline definition for images needs to run just once,
            // so calling it from the constructor:
            mlModel = SetupMlNetModel(yoloModel);
        }

        private ITransformer SetupMlNetModel(IYoloModel yoloModel)
        {
            var dataView = mlContext.Data.LoadFromEnumerable(new List<ImageInputData>());

            var pipeline = mlContext.Transforms.ResizeImages(resizing: ImageResizingEstimator.ResizingKind.Fill, outputColumnName: yoloModel.ModelInput, imageWidth: 416, imageHeight: 416, inputColumnName: nameof(ImageInputData.Image))
                            .Append(mlContext.Transforms.ExtractPixels(outputColumnName: yoloModel.ModelInput))
                            .Append(mlContext.Transforms.ApplyOnnxModel(modelFile: yoloModel.ModelPath, outputColumnName: yoloModel.ModelOutput, inputColumnName: yoloModel.ModelInput));

            var mlNetModel = pipeline.Fit(dataView);

            return mlNetModel;
        }

        public PredictionEngine<ImageInputData, T> GetMlNetPredictionEngine<T>()
            where T : class, IYoloObjectPrediction, new()
        {
            return mlContext.Model.CreatePredictionEngine<ImageInputData, T>(mlModel);
        }

        public void SaveMLNetModel(string mlnetModelFilePath)
        {
            // Save/persist the model to a .ZIP file to be loaded by the PredictionEnginePool
            mlContext.Model.Save(mlModel, null, mlnetModelFilePath);
        }
    }
}
