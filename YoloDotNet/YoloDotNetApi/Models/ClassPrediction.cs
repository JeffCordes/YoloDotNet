using System;
using Newtonsoft.Json;

namespace YoloDotNetApi.Models
{
    public class ClassPrediction : IComparable
    {
        [JsonProperty("label")]
        public string Label { get; set; }
        [JsonProperty("confidence")]
        public float Confidence { get; set; }

        public int CompareTo(object other)
        { 
            return Confidence.CompareTo(((ClassPrediction)other).Confidence); 
        }
    }
}
