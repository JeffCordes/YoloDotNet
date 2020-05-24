using MLNetSamples.Yolo2Coco.Models.Input;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;

namespace MLNetSamples.Yolo2Coco.Service
{
    public interface IYoloDetectionService
    {
        void DetectObjectsUsingModel(ImageInputData imageInputData);
        Image DrawBoundingBox(string imageFilePath);
    }
}
