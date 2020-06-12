namespace YoloDotNetApi.Models
{
    public class Yolo2CoCoSettings
    {
        public string Model { get; set; }
        public int InputWidth { get; set; }
        public int InputHeight { get; set; }
        public int GridWidth { get; set; }
        public int GridHeight { get; set; }
        public int Anchors { get; set; }
        public int ClassCount { get; set; }
        public float ConfidenceLimit { get; set; }
        public bool IntersectionOverUnion { get; set; }
        public float IOUOverlapLimit { get; set; }
        public int MaxResults { get; set; }
    }
}
