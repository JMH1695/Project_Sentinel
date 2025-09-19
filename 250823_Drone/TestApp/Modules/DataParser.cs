using System;
using Newtonsoft.Json.Linq;

namespace TestApp.Modules
{
    public class DroneStatus
    {
        public float Battery { get; set; }
        public float Altitude { get; set; }
        public double Latitude { get; set; }
        public double Longitude { get; set; }
        public string EngineStatus { get; set; }
    }

    public class VehicleDetection
    {
        public string VehicleNumber { get; set; }
        public DateTime DetectedTime { get; set; }
        public bool IsIllegal { get; set; }
        public float Confidence { get; set; }
        public double Latitude { get; set; }
        public double Longitude { get; set; }
        public string ImagePath { get; set; }
    }

    public static class DataParser
    {
        public static DroneStatus ParseDroneStatus(string json)
        {
            var data = JObject.Parse(json);

            return new DroneStatus
            {
                Battery = data["Battery"]?.ToObject<float?>() ?? -1f,
                Altitude = data["Altitude"]?.ToObject<float?>() ?? -1f,
                Latitude = data["Latitude"]?.ToObject<double?>() ?? 0,
                Longitude = data["Longitude"]?.ToObject<double?>() ?? 0,
                EngineStatus = data["EngineStatus"]?.ToString() ?? "OFF"
            };
        }

        public static VehicleDetection ParseVehicleDetection(string json)
        {
            var data = JObject.Parse(json);

            return new VehicleDetection
            {
                VehicleNumber = data["vehicle_number"]?.ToString(),
                DetectedTime = DateTime.Parse(data["detected_time"]?.ToString()),
                IsIllegal = data["is_illegal"]?.ToObject<bool>() ?? false,
                Confidence = data["confidence"]?.ToObject<float>() ?? 0f,
                Latitude = data["latitude"]?.ToObject<double>() ?? 0,
                Longitude = data["longitude"]?.ToObject<double>() ?? 0,
                ImagePath = data["image_path"]?.ToString()
            };
        }
    }
}
