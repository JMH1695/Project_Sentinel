// DataReceiver.cs
using System;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using MQTTnet;
using MQTTnet.Client;
using MQTTnet.Client.Options;           // (사용 중이면 유지)
using MQTTnet.Formatter;                // 프로토콜 버전 지정용
using Newtonsoft.Json.Linq;
using System.Windows.Forms;
using System.Drawing;
using System.Media;
using TestApp.Helpers;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Data;

namespace TestApp.Modules
{
    public class DataReceiver
    {
        private readonly RichTextBox txtLog;
        private readonly DataGridView dgvData;
        private readonly TextBox textBoxAltitude;
        private readonly TextBox textBoxLocation;
        private readonly Label labelEngineStatus;
        private readonly ProgressBar progressBarBattery;
        private readonly PictureBox pictureBoxVehicle;

        // ===== MQTT =====
        private readonly string mqttServerIp = "43.200.182.226";
        private IMqttClient mqttClient;

        // ===== TCP =====
        private TcpListener tcpListener;
        private CancellationTokenSource tcpCts;
        private bool tcpRunning = false;

        // 🔧 기본 TCP 바인딩(같은 PC 테스트면 127.0.0.1/5059)
        private string tcpBindIp = "127.0.0.1";
        private int tcpPort = 5059;

        private const float BATTERY_THRESHOLD = 20.0f;

        // ===== Grid 제어(샘플링/최대행) =====
        private DateTime _lastVehicleGridUpdate = DateTime.MinValue;
        private static readonly TimeSpan _minVehicleInterval = TimeSpan.FromMilliseconds(300); // 필요 시 조절
        private const int MaxRows = 500; // 최근 500건만 유지

        // (옵션) 상태를 그리드에 표기할지 여부 (기본 false)
        private bool _showStatusInGrid = false;


        //1) 클래스 필드 추가
        // ===== A안: 상태 토픽의 최신 좌표를 저장 → 차량 메시지에 보충 사용 =====
        private double _lastLat = double.NaN;
        private double _lastLng = double.NaN;

        private static bool IsValidCoord(double lat, double lng) =>
            lat >= -90 && lat <= 90 &&
            lng >= -180 && lng <= 180 &&
            !(lat == 0 && lng == 0);

        public DataReceiver(
            RichTextBox logBox,
            DataGridView grid,
            TextBox altitudeBox,
            TextBox locationBox,
            Label engineLabel,
            ProgressBar batteryBar,
            PictureBox vehicleBox)
        {
            txtLog = logBox;
            dgvData = grid;
            textBoxAltitude = altitudeBox;
            textBoxLocation = locationBox;
            labelEngineStatus = engineLabel;
            progressBarBattery = batteryBar;
            pictureBoxVehicle = vehicleBox; // ✅
        }

        // ==========================
        // MQTT
        // ==========================
        public async Task InitMqttAsync()
        {
            try
            {
                var factory = new MqttFactory();
                mqttClient = factory.CreateMqttClient();

                var options = new MqttClientOptionsBuilder()
                    .WithTcpServer(mqttServerIp, 1883)
                    .WithCleanSession()
                    .WithKeepAlivePeriod(TimeSpan.FromSeconds(30))
                    .WithCommunicationTimeout(TimeSpan.FromSeconds(5))
                    .WithProtocolVersion(MqttProtocolVersion.V311) // 호환성 좋음
                    .Build();

                mqttClient.UseConnectedHandler(async e =>
                {
                    LogManager.Log(txtLog, "✅ MQTT 연결 성공");
                    await mqttClient.SubscribeAsync("drone/status");
                    await mqttClient.SubscribeAsync("drone/vehicle");
                    LogManager.Log(txtLog, "📡 구독 시작: drone/status, drone/vehicle");
                });

                mqttClient.UseDisconnectedHandler(async e =>
                {
                    LogManager.Log(txtLog, $"⚠️ MQTT 연결 끊김: {e.Exception?.Message}", LogLevel.Warning);
                    // 간단 재연결 루프
                    await Task.Delay(2000);
                    try { await mqttClient.ConnectAsync(options); } catch { /* 무시/로그 */ }
                });

                mqttClient.UseApplicationMessageReceivedHandler(e =>
                {
                    var payload = Encoding.UTF8.GetString(e.ApplicationMessage.Payload ?? Array.Empty<byte>());
                    var topic = e.ApplicationMessage.Topic ?? "";

                    if (topic == "drone/status")
                        HandleDroneStatus(payload);
                    else if (topic == "drone/vehicle")
                        HandleVehicleDetection(payload);
                    else
                        RouteJson(payload); // 예외적 토픽: JSON 보고 분기
                });

                await mqttClient.ConnectAsync(options);
            }
            catch (Exception ex)
            {
                LogManager.Log(txtLog, "❌ MQTT 초기화 실패: " + ex.Message, LogLevel.Error);
            }
        }

        // ==========================
        // TCP SERVER
        // ==========================
        public async Task InitTcpServerAsync(string bindIp = null, int? port = null)
        {
            if (bindIp != null) tcpBindIp = bindIp;
            if (port.HasValue) tcpPort = port.Value;

            try
            {
                if (tcpRunning)
                {
                    LogManager.Log(txtLog, $"TCP 서버 이미 실행 중: {tcpBindIp}:{tcpPort}", LogLevel.Warning);
                    return;
                }

                tcpCts = new CancellationTokenSource();

                IPAddress ip = (tcpBindIp == "0.0.0.0") ? IPAddress.Any : IPAddress.Parse(tcpBindIp);
                tcpListener = new TcpListener(ip, tcpPort);
                tcpListener.Start();
                tcpRunning = true;

                LogManager.Log(txtLog, $"🟢 TCP 서버 대기 시작: {tcpBindIp}:{tcpPort}");

                // 🔁 Accept 루프를 비블로킹으로 흘려보냄
                _ = AcceptLoopAsync(tcpCts.Token);

                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                LogManager.Log(txtLog, "❌ TCP 서버 초기화 실패: " + ex.Message, LogLevel.Error);
                tcpRunning = false;
            }
        }

        private async Task AcceptLoopAsync(CancellationToken ct)
        {
            while (!ct.IsCancellationRequested)
            {
                try
                {
                    var client = await tcpListener.AcceptTcpClientAsync();
                    _ = HandleClientAsync(client, ct);
                }
                catch (ObjectDisposedException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    LogManager.Log(txtLog, "TCP Accept 오류: " + ex.Message, LogLevel.Error);
                }
            }
        }

        private async Task HandleClientAsync(TcpClient client, CancellationToken ct)
        {
            LogManager.Log(txtLog, "🔌 TCP 클라이언트 연결됨");
            using (client)
            using (var ns = client.GetStream())
            using (var sr = new StreamReader(ns, Encoding.UTF8))
            {
                try
                {
                    while (!ct.IsCancellationRequested)
                    {
                        var line = await sr.ReadLineAsync();
                        if (line == null) break; // 클라이언트 종료

                        LogManager.Log(txtLog, $"[TCP] 수신: {line}");
                        RouteJson(line);
                    }
                }
                catch (IOException)
                {
                    // 연결 종료/중단
                }
                catch (Exception ex)
                {
                    LogManager.Log(txtLog, "TCP 클라이언트 처리 오류: " + ex.Message, LogLevel.Error);
                }
            }
            LogManager.Log(txtLog, "⛔ TCP 클라이언트 종료");
        }

        /// <summary>
        /// 수신한 JSON 라인을 보고 핸들러로 분기.
        /// - 새 스키마(type=violation_finalized)면 옛 스키마로 매핑해서 차량 핸들러 호출
        /// - 그 외는 키 존재 여부로 차량/상태 분기
        /// </summary>
        private void RouteJson(string json)
        {
            try
            {
                var jo = JObject.Parse(json);

                // 새 스키마: violation_finalized (viewer.py에서 전송) → 매핑 후 처리
                var ty = jo["type"]?.ToString();
                if (string.Equals(ty, "violation_finalized", StringComparison.OrdinalIgnoreCase))
                {
                    var legacy = MapViolationToLegacy(jo);
                    HandleVehicleDetection(legacy.ToString());
                    return;
                }

                // 차량/상태 키로 분기
                bool looksVehicle =
                    jo.ContainsKey("vehicle_number") ||
                    jo.ContainsKey("detected_time") ||
                    jo.ContainsKey("is_illegal");

                bool looksStatus =
                    jo.ContainsKey("Battery") ||
                    jo.ContainsKey("Altitude") ||
                    jo.ContainsKey("EngineStatus");

                if (looksVehicle && !looksStatus)
                {
                    HandleVehicleDetection(json);
                }
                else if (looksStatus && !looksVehicle)
                {
                    HandleDroneStatus(json);
                }
                else
                {
                    // 애매하면 차량→실패 시 상태
                    try { HandleVehicleDetection(json); }
                    catch { HandleDroneStatus(json); }
                }
            }
            catch (Exception ex)
            {
                LogManager.Log(txtLog, "JSON 라우팅 오류: " + ex.Message, LogLevel.Error);
            }
        }

        /// <summary>
        /// viewer 새 스키마(violation_finalized)를 WinForms 옛 스키마로 변환
        ///  - vehicle_number ← plate_text
        ///  - detected_time  ← ts (없으면 now)
        ///  - is_illegal     ← true
        ///  - confidence     ← plate_conf
        ///  - latitude/longitude ← src.lat/lon 있으면 사용, 없으면 최근 상태 좌표 보충, 그래도 없으면 0
        ///  - image_path     ← image_path
        /// </summary>
        private JObject MapViolationToLegacy(JObject src)
        {
            string ts = src["ts"]?.ToString();
            string vehicle = src["plate_text"]?.ToString() ?? "";
            float conf = 0f;
            try { conf = src["plate_conf"]?.ToObject<float>() ?? 0f; } catch { /* ignore */ }

            string img = src["image_path"]?.ToString();


            //3) MapViolationToLegacy 수정(좌표 보충)

            // 1) 메시지 자체에 좌표가 있으면 우선 사용 (lat / lon, 혹은 latitude / longitude 키 모두 대응)
            double lat = double.NaN, lng = double.NaN;
            // 다양한 키 대응
            if (!TryReadDouble(src, out lat, "lat", "latitude"))
                lat = double.NaN;
            if (!TryReadDouble(src, out lng, "lon", "lng", "longitude"))
                lng = double.NaN;

            // 2) 없으면 최근 상태 좌표 보충
            if (!IsValidCoord(lat, lng) && IsValidCoord(_lastLat, _lastLng))
            {
                lat = _lastLat;
                lng = _lastLng;
            }

            var legacy = new JObject
            {
                ["vehicle_number"] = vehicle,
                ["detected_time"] = string.IsNullOrWhiteSpace(ts)
                    ? DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")
                    : ts,
                ["is_illegal"] = true,
                ["confidence"] = conf,
                ["latitude"] = IsValidCoord(lat, lng) ? lat : 0.0,
                ["longitude"] = IsValidCoord(lat, lng) ? lng : 0.0,
                ["image_path"] = img ?? ""
            };

            return legacy;
        }

        private static bool TryReadDouble(JObject src, out double value, params string[] keys)
        {
            foreach (var k in keys)
            {
                if (src[k] != null && double.TryParse(src[k]?.ToString(), out value))
                    return true;
            }
            value = double.NaN;
            return false;
        }

        /// <summary>
        /// TCP 서버 중지(폼 종료 시 호출 권장)
        /// </summary>
        public void StopTcpServer()
        {
            try
            {
                tcpRunning = false;
                tcpCts?.Cancel();
                tcpListener?.Stop();
                LogManager.Log(txtLog, "🟥 TCP 서버 중지 완료");
            }
            catch (Exception ex)
            {
                LogManager.Log(txtLog, "TCP 서버 중지 오류: " + ex.Message, LogLevel.Error);
            }
        }

        // ==========================
        // 핸들러
        //2) HandleDroneStatus 수정 (좌표 저장)
        //4) HandleVehicleDetection 수정 (좌표 보충)
        // ==========================
        private void HandleDroneStatus(string json)
        {
            try
            {
                var data = JObject.Parse(json);
                float battery = data["Battery"]?.ToObject<float?>() ?? -1f;
                float altitude = data["Altitude"]?.ToObject<float?>() ?? -1f;

                // 다양한 키에 대응 (Latitude/Longitude, lat/lon)
                double latitude = data["Latitude"]?.ToObject<double?>() ??
                                  data["lat"]?.ToObject<double?>() ?? 0;
                double longitude = data["Longitude"]?.ToObject<double?>() ??
                                   data["lon"]?.ToObject<double?>() ??
                                   data["lng"]?.ToObject<double?>() ?? 0;

                string engineStatus = data["EngineStatus"]?.ToString() ?? "OFF";

                // ✅ 최신 상태 좌표 저장(A안 핵심)
                if (IsValidCoord(latitude, longitude))
                {
                    _lastLat = latitude;
                    _lastLng = longitude;
                }

                UIHelper.SafeUpdate(progressBarBattery, () =>
                {
                    // ProgressBar ForeColor는 VisualStyles에서 반영이 제한적일 수 있음.
                    progressBarBattery.Value = Math.Max(0, Math.Min(100, (int)battery));
                });

                UIHelper.SafeUpdate(textBoxAltitude, () => textBoxAltitude.Text = $"{altitude} m");
                UIHelper.SafeUpdate(textBoxLocation, () => textBoxLocation.Text = $"Lat: {latitude}, Lon: {longitude}");
                UIHelper.SafeUpdate(labelEngineStatus, () => labelEngineStatus.Text = engineStatus);

                // 🔕 상태는 그리드에 추가하지 않음(요청사항 반영)
                if (_showStatusInGrid)
                {
                    UIHelper.SafeUpdate(dgvData, () =>
                    {
                        int rowIndex = dgvData.Rows.Add("드론 상태", DateTime.Now, false, battery, $"{latitude}, {longitude}");
                        dgvData.Rows[rowIndex].DefaultCellStyle.BackColor = Color.LightBlue;

                        if (dgvData.Rows.Count > MaxRows)
                            dgvData.Rows.RemoveAt(dgvData.Rows.Count - 1);
                    });
                }

                LogManager.Log(txtLog, $"드론 상태 수신: 배터리={battery}%, 고도={altitude}m, 위치=({latitude}, {longitude}), 엔진={engineStatus}");
            }
            catch (Exception ex)
            {
                LogManager.Log(txtLog, "드론 상태 처리 오류: " + ex.Message, LogLevel.Error);
            }
        }

        private void HandleVehicleDetection(string json)
        {
            try
            {
                var data = JObject.Parse(json);

                // 🔒 null-safe 파싱
                string vehicleNumber = data["vehicle_number"]?.ToString();
                if (string.IsNullOrWhiteSpace(vehicleNumber))
                    vehicleNumber = "미확인";

                DateTime detectedTime;
                var detectedTimeStr = data["detected_time"]?.ToString();
                if (!DateTime.TryParse(detectedTimeStr, out detectedTime))
                    detectedTime = DateTime.Now;

                bool isIllegal = data["is_illegal"]?.ToObject<bool?>() ?? false;
                float confidence = data["confidence"]?.ToObject<float?>() ?? 0f;

                // 원본 메시지의 좌표 읽기(여러 키 대응)
                double latitude =
                    data["latitude"]?.ToObject<double?>() ??
                    data["lat"]?.ToObject<double?>() ?? 0;
                double longitude =
                    data["longitude"]?.ToObject<double?>() ??
                    data["lon"]?.ToObject<double?>() ??
                    data["lng"]?.ToObject<double?>() ?? 0;

                // ✅ 좌표 보강(A안 핵심): 메시지에 좌표가 없거나(0,0)/유효범위밖이면 최근 상태 좌표로 보충
                if (!IsValidCoord(latitude, longitude) && IsValidCoord(_lastLat, _lastLng))
                {
                    latitude = _lastLat;
                    longitude = _lastLng;
                    LogManager.Log(txtLog, $"ℹ️ 좌표 보충: ({latitude}, {longitude}) - 최근 상태값 사용");
                }

                string imagePath = data["image_path"]?.ToString();

                // 🧹 그리드 업데이트 샘플링/최대행 제한
                var now = DateTime.UtcNow;
                if (now - _lastVehicleGridUpdate >= _minVehicleInterval)
                {
                    _lastVehicleGridUpdate = now;

                    //UIHelper.SafeUpdate(dgvData, () =>
                    //{
                    //    int rowIndex = dgvData.Rows.Add(vehicleNumber, detectedTime, isIllegal, confidence, $"{latitude}, {longitude}");
                    //    if (isIllegal)
                    //        dgvData.Rows[rowIndex].DefaultCellStyle.BackColor = Color.Red;
                    //
                    //    if (dgvData.Rows.Count > MaxRows)
                    //        dgvData.Rows.RemoveAt(dgvData.Rows.Count - 1);
                    //});

                    // (1) 좌표 보강 이후, 표시 문자열 보정 추가
                    // ✅ 좌표가 여전히 무효(0,0 등)라면 "N/A"로 표시
                    string locStr = IsValidCoord(latitude, longitude) ? $"{latitude}, {longitude}" : "N/A";

                    //AppendRowSafe(vehicleNumber, detectedTime, isIllegal, confidence, $"{latitude}, {longitude}");
                    AppendRowSafe(vehicleNumber, detectedTime, isIllegal, confidence, locStr);
                }

                LogManager.Log(txtLog, $"차량 탐지 수신: {vehicleNumber} / {detectedTime}");

                if (isIllegal && confidence >= 0.8f)
                {
                    //MessageBox.Show($"불법 주정차 차량 발견!\n번호: {vehicleNumber}\n신뢰도: {confidence * 100:F1}%", "경고", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    SystemSounds.Beep.Play();
                    LogManager.AppendImportant(txtLog, $"불법 차량 감지: {vehicleNumber}, 신뢰도 {confidence * 100:F1}%");
                }

                // 📷 이미지 표시 (파일 잠김/누수 방지)
                UIHelper.SafeUpdate(pictureBoxVehicle, () =>
                {
                    try
                    {
                        Image old = pictureBoxVehicle.Image;

                        if (!string.IsNullOrEmpty(imagePath) && File.Exists(imagePath))
                        {
                            using (var src = new Bitmap(imagePath))
                            {
                                pictureBoxVehicle.Image = new Bitmap(src); // 복사본으로 교체
                            }
                            LogManager.Log(txtLog, $"✔ 이미지 로드 성공: {imagePath}");
                        }
                        else
                        {
                            pictureBoxVehicle.Image = Properties.Resources.no_image;
                            LogManager.Log(txtLog, $"❌ 이미지 파일 없음, 대체 이미지로 전환", LogLevel.Warning);
                        }

                        old?.Dispose();
                    }
                    catch (Exception ex)
                    {
                        LogManager.Log(txtLog, $"이미지 갱신 오류: {ex.Message}", LogLevel.Error);
                    }
                });
            }
            catch (Exception ex)
            {
                LogManager.Log(txtLog, "차량 탐지 처리 오류: " + ex.Message, LogLevel.Error);
            }
        }

        // ==========================
        // 유틸
        // ==========================
        public void SetTcpEndpoint(string ip, int port)
        {
            tcpBindIp = ip;
            tcpPort = port;
            LogManager.Log(txtLog, $"TCP 엔드포인트 설정: {tcpBindIp}:{tcpPort}");
        }


        // 바인딩/비바인딩 안전 추가 헬퍼
        private void AppendRowSafe(string number, DateTime time, bool illegal, float conf, string loc)
        {
            UIHelper.SafeUpdate(dgvData, () =>
            {
                if (dgvData.DataSource is BindingSource bs)
                {
                    if (bs.DataSource is DataTable dt)
                    {
                        dt.Rows.Add(number, time, illegal, conf, loc);
                    }
                    else if (bs.List is DataView dv && dv.Table != null)
                    {
                        dv.Table.Rows.Add(number, time, illegal, conf, loc);
                    }
                }
                else if (dgvData.DataSource is DataTable dt2)
                {
                    dt2.Rows.Add(number, time, illegal, conf, loc);
                }
                else
                {
                    int rowIndex = dgvData.Rows.Add(number, time, illegal, conf, loc);
                    if (illegal) dgvData.Rows[rowIndex].DefaultCellStyle.BackColor = Color.Red;
                }
                // (선택) 최대 행 제한 등은 필요 시 추가
            });
        }

    }
}
