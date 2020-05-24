using System;
using System.Collections.Generic;
using System.Text;

namespace MLNetSamples.Yolo2Coco.Models
{
    public class YoloModel : IYoloModel
    {
        public string ModelPath { get; private set; }

        public string ModelInput { get; } = "image";
        public string ModelOutput { get; } = "grid";

        public string[] Labels { get; } =
        {
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        };

        public (float, float)[] Anchors { get; } = { (1.08f, 1.19f), (3.42f, 4.41f), (6.63f, 11.38f), (9.42f, 5.11f), (16.62f, 10.52f) };

        public YoloModel(string modelPath)
        {
            ModelPath = modelPath;
        }
    }
}
