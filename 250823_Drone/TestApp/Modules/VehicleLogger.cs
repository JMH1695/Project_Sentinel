using System;
using System.Collections.Generic;
using System.Data;
using System.Drawing;
using System.Windows.Forms;
using MySqlConnector;
using TestApp.Helpers;

namespace TestApp.Modules
{
    public static class VehicleLogger
    {
        private static readonly string connectionString =
            "Server=localhost;Port=3307;Database=drone_parking;Uid=droneuser;Pwd=Dong@0923;";

        public static void InsertVehicleLog(string vehicleNumber, DateTime detectedTime, bool isIllegal,
    float confidence, double latitude, double longitude, string imagePath, RichTextBox logBox = null)
        {
            string query = @"
        INSERT INTO vehicle_detection
        (vehicle_number, detected_time, is_illegal, illegal_confidence, detection_confidence, latitude, longitude, image_path)
        VALUES 
        (@vehicle_number, @detected_time, @is_illegal, @illegal_confidence, @detection_confidence, @latitude, @longitude, @image_path)";

            try
            {
                using (var conn = new MySqlConnection(connectionString))
                {
                    conn.Open();
                    using (var cmd = new MySqlCommand(query, conn))
                    {
                        cmd.Parameters.AddWithValue("@vehicle_number", vehicleNumber);
                        cmd.Parameters.AddWithValue("@detected_time", detectedTime);
                        cmd.Parameters.AddWithValue("@is_illegal", isIllegal);
                        cmd.Parameters.AddWithValue("@illegal_confidence", isIllegal ? confidence : 0f); // 불법일 때만 신뢰도 사용
                        cmd.Parameters.AddWithValue("@detection_confidence", confidence);
                        cmd.Parameters.AddWithValue("@latitude", latitude);
                        cmd.Parameters.AddWithValue("@longitude", longitude);
                        cmd.Parameters.AddWithValue("@image_path", imagePath ?? "");

                        cmd.ExecuteNonQuery();
                    }
                }

                if (logBox != null)
                    LogManager.Log(logBox, $"✅ DB 저장 완료: {vehicleNumber}", LogLevel.Info);
            }
            catch (Exception ex)
            {
                if (logBox != null)
                    LogManager.Log(logBox, "❌ DB 저장 오류: " + ex.Message, LogLevel.Error);
            }
        }


        // MYSQL DB 관련쪽 + GMAP 마커 
        public static void LoadVehicleDetectionData(DataGridView dgv, RichTextBox logBox, Action<List<(double lat, double lng)>> onLocationsLoaded = null)
        {
            try
            {
                dgv.AllowUserToAddRows = false; // ✅ 편집용 빈 행 제거

                using (var connection = new MySqlConnection(connectionString))
                {
                    connection.Open();

                    string query = "SELECT * FROM vehicle_detection ORDER BY detected_time DESC";
                    using (var cmd = new MySqlCommand(query, connection))
                    using (var reader = cmd.ExecuteReader())
                    {
                        // ✅ 컬럼이 없을 경우 기본 컬럼 추가
                        if (dgv.ColumnCount == 0)
                        {
                            dgv.Columns.Add("vehicleNumber", "차량번호");
                            dgv.Columns.Add("detectedTime", "탐지시간");
                            dgv.Columns.Add("isIllegal", "불법 여부");
                            dgv.Columns.Add("confidence", "신뢰도");
                            dgv.Columns.Add("location", "위치");
                        }

                        dgv.Rows.Clear();
                        var locations = new List<(double, double)>();

                        
                        while (reader.Read())
                        {
                            string vehicleNumber = reader.GetString("vehicle_number");
                            DateTime detectedTime = reader.GetDateTime("detected_time");
                            bool isIllegal = reader.GetBoolean("is_illegal");

                            // ❗ null 허용 필드 처리 (예외 방지용)
                            float confidence = reader["confidence"] != DBNull.Value ? Convert.ToSingle(reader["confidence"]) : 0f;
                            double latitude = reader["latitude"] != DBNull.Value ? Convert.ToDouble(reader["latitude"]) : 0.0;
                            double longitude = reader["longitude"] != DBNull.Value ? Convert.ToDouble(reader["longitude"]) : 0.0;
                            string locationStr = $"{latitude}, {longitude}";

                            dgv.Rows.Add(vehicleNumber, detectedTime, isIllegal, confidence, locationStr);

                            var coord = (latitude, longitude);
                            if (!locations.Contains(coord))
                                locations.Add(coord);
                        }

                        logBox.AppendText($"[🗃️] DB에서 {dgv.Rows.Count}개 차량 데이터 로드 완료\n");

                        // 📍 위치 콜백 전달
                        onLocationsLoaded?.Invoke(locations);
                    }
                }
            }
            catch (Exception ex)
            {
                LogManager.Log(logBox, "DB 조회 오류: " + ex.Message, LogLevel.Error);
            }
        }

        public static List<VehicleRecord> FetchNewVehicleData(List<string> knownVehicles)
        {
            List<VehicleRecord> newRecords = new List<VehicleRecord>();

            using (var conn = new MySqlConnection(DBHelper.ConnectionString))
            {
                conn.Open();
                string query = "SELECT vehicle_number, detected_time, location FROM vehicle_detection ORDER BY id DESC LIMIT 50";

                using (var cmd = new MySqlCommand(query, conn))
                using (var reader = cmd.ExecuteReader())
                {
                    while (reader.Read())
                    {
                        string number = reader["vehicle_number"].ToString();

                        if (!knownVehicles.Contains(number))
                        {
                            newRecords.Add(new VehicleRecord
                            {
                                VehicleNumber = number,
                                DetectedTime = reader["detected_time"].ToString(),
                                Location = reader["location"].ToString()
                            });
                        }
                    }
                }
            }

            return newRecords;
        }

        public class VehicleRecord
        {
            public string VehicleNumber { get; set; }
            public string DetectedTime { get; set; }
            public string Location { get; set; }
        }


    }
}
