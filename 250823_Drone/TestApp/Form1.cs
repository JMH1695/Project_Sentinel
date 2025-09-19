using System;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Windows.Forms;
using TestApp.Helpers;
using TestApp.Modules;
using GMap.NET;
using GMap.NET.MapProviders;
using GMap.NET.WindowsForms;
using GMap.NET.WindowsForms.Markers;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using System.Text.Json;
using System.Net.Http;
using Microsoft.Web.WebView2.Core;
using System.Threading.Tasks;

namespace TestApp
{
    public partial class Form1 : Form
    {
        // 🔷 지도 오버레이
        private GMapOverlay markersOverlay;

        // 🔷 수신기 (MQTT/TCP 라우팅 포함)
        private DataReceiver dataReceiver;

        // 🔷 DB 자동 갱신 타이머 및 상태
        private Timer dataUpdateTimer;
        private List<string> knownVehicleNumbers = new List<string>();

        // 🔷 DataGridView 실시간 공유용 데이터 테이블 & 바인딩소스
        private readonly DataTable _vehicleTable = new DataTable("vehicles");
        private readonly BindingSource _vehicleBS = new BindingSource();

        // 🔷 “더 크게 보기” 단일 창 핸들과 재사용
        private TestApp.Modules.VehicleListForm _listForm;


        // Form1 클래스 필드
        private Timer webRefreshTimer;
        private int _webAutoRefreshMs = 15000; // 자동 새로고침 주기(기본 15초, 원하면 바꿔)
        private Uri _webrtcUri;                 // 현재 영상 페이지 URI 저장
        private int _webConsecutiveErrors = 0;  // 연속 오류(백오프용)


        //// ▶ viewer 가용성 감시용 (핵심)
        //private System.Windows.Forms.Timer viewerWatchdogTimer;
        //private static readonly HttpClient _http = new HttpClient() { Timeout = TimeSpan.FromSeconds(1) };
        //private volatile bool _viewerUp = false;      // 현재 상태
        //private volatile bool _viewerEverUp = false;  // 한번이라도 살아난 적 있는지


        public Form1()
        {
            InitializeComponent();

            // 폼 이벤트 연결
            this.Load += Form1_Load;
            this.FormClosing += Form1_FormClosing;

            // DataReceiver 준비(컨트롤 주입)
            dataReceiver = new DataReceiver(
                txtLog,              // RichTextBox (로그)
                dgvData,             // DataGridView (리스트) — 초기 로딩용으로만 사용, 이후 바인딩으로 대체
                textBoxAltitude,     // TextBox (고도)
                textBoxLocation,     // TextBox (위치)
                labelEngineStatus,   // Label (엔진 상태)
                progressBarBattery,  // ProgressBar (배터리)
                pictureBoxVehicle    // PictureBox (차량 이미지)
            );

            // 버튼 이벤트 (디자이너에서 연결했으면 중복 연결 방지)
            btnViewLarge.Click -= btnViewLarge_Click;
            btnViewLarge.Click += btnViewLarge_Click;

            // 필요 시 MQTT도 함께 켤 수 있음 (옵션)
            // _ = dataReceiver.InitMqttAsync();
        }

        private async void Form1_Load(object sender, EventArgs e)
        {
            // ✅ DataGridView 바인딩 준비(실시간 공유)
            SetupVehicleBinding();

            // ✅ GMap 초기 세팅
            InitializeMap();
            AddMapMarker(37.5665, 126.9780, "📍 테스트 마커", GMarkerGoogleType.blue);

            // ✅ CellClick 핸들러 중복방지 후 연결
            dgvData.CellClick -= dgvData_CellClick;
            dgvData.CellClick += dgvData_CellClick;

            // ✅ 로그 히스토리 로드
            LogManager.LoadHistory(txtLog);

            // ✅ WebRTC/HLS WebView 로드
            InitializeWebRTCView();

            // ✅ DB에서 기존 차량 탐지 기록 로딩 + 지도 마커
            //    (VehicleLogger가 dgvData에 직접 채우므로, 바로 아래에서 DataTable로 스냅샷 이관)
            VehicleLogger.LoadVehicleDetectionData(dgvData, txtLog, (locations) =>
            {
                foreach (var (lat, lng) in locations)
                    AddMapMarker(lat, lng, "📍 감지 차량 위치", GMarkerGoogleType.red_dot);
            });

            // ✅ 초기 1회 스냅샷 이관 → 이후엔 DataTable만 사용
            MigrateGridToTableIfNeeded();

            // ✅ 테스트 이미지 로딩(없으면 대체 이미지)
            TryLoadTestImage();

            // ✅ TCP 서버 시작 (같은 PC면 127.0.0.1 / 5059)
            await dataReceiver.InitTcpServerAsync("127.0.0.1", 5059);

            // ✅ 실시간 DB 자동 갱신 타이머 시작
            StartAutoUpdate();

            // ✅ WebView2 URL 설정
            var url = "http://127.0.0.1:8099/"; // 또는 /video 직접
            webView21.Source = new Uri(url);
            LogManager.Log(txtLog, $"📺 WebView2 로드: {url}");
        }

        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            // TCP 서버 정리
            dataReceiver?.StopTcpServer();
        }

        // ─────────────────────────────────────────────────────────────
        // DataGridView 바인딩 구성 (실시간 공유)
        // ─────────────────────────────────────────────────────────────
        private void SetupVehicleBinding()
        {
            if (_vehicleTable.Columns.Count == 0)
            {
                // 컬럼 스키마(표시 순서 포함)
                _vehicleTable.Columns.Add("vehicleNumber", typeof(string));
                _vehicleTable.Columns.Add("detectedTime", typeof(DateTime));
                _vehicleTable.Columns.Add("isIllegal", typeof(bool));
                _vehicleTable.Columns.Add("confidence", typeof(float));
                _vehicleTable.Columns.Add("location", typeof(string));
            }

            _vehicleBS.DataSource = _vehicleTable;

            // 메인 DataGridView에 바인딩
            dgvData.AutoGenerateColumns = true;
            dgvData.DataSource = _vehicleBS;

            // 보기 설정(선택)
            dgvData.AllowUserToAddRows = false;
            dgvData.ReadOnly = true;
            dgvData.AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill;
            dgvData.SelectionMode = DataGridViewSelectionMode.FullRowSelect;
            dgvData.RowHeadersVisible = false;
        }

        // VehicleLogger가 채운 dgvData의 내용을 1회만 DataTable로 이관
        private void MigrateGridToTableIfNeeded()
        {
            // 이미 데이터가 있다면 스킵
            if (_vehicleTable.Rows.Count > 0) return;

            // dgvData가 비어있으면 스킵
            if (dgvData.Rows.Count == 0 || dgvData.Columns.Count < 1) return;

            // 컬럼 키 매핑 시도
            int idxVehicle = GetColIndex(dgvData, "vehicleNumber");
            int idxTime = GetColIndex(dgvData, "detectedTime");
            int idxIllegal = GetColIndex(dgvData, "isIllegal");
            int idxConf = GetColIndex(dgvData, "confidence");
            int idxLoc = GetColIndex(dgvData, "location");

            foreach (DataGridViewRow r in dgvData.Rows)
            {
                if (r.IsNewRow) continue;

                string num = ReadCell<string>(r, idxVehicle);
                DateTime time = ReadCell<DateTime>(r, idxTime, DateTime.Now);
                bool illegal = ReadCell<bool>(r, idxIllegal, false);
                float conf = ReadCell<float>(r, idxConf, 0f);
                string loc = ReadCell<string>(r, idxLoc);

                _vehicleTable.Rows.Add(num, time, illegal, conf, loc);
            }

            // 메인 그리드는 계속 BindingSource 사용
            dgvData.DataSource = _vehicleBS;
        }

        private static int GetColIndex(DataGridView dgv, string name)
        {
            if (dgv.Columns.Contains(name)) return dgv.Columns[name].Index;
            // HeaderText로도 찾아보기(한국어 헤더 대응)
            foreach (DataGridViewColumn c in dgv.Columns)
            {
                if (string.Equals(c.HeaderText, name, StringComparison.OrdinalIgnoreCase))
                    return c.Index;
            }
            return -1;
        }

        private static T ReadCell<T>(DataGridViewRow r, int index, T defaultValue = default)
        {
            try
            {
                if (index < 0) return defaultValue;
                object v = r.Cells[index].Value;
                if (v == null || v == DBNull.Value) return defaultValue;

                if (typeof(T) == typeof(string)) return (T)(object)v.ToString();
                if (typeof(T) == typeof(DateTime))
                {
                    if (v is DateTime dt) return (T)(object)dt;
                    if (DateTime.TryParse(v.ToString(), out var parsed)) return (T)(object)parsed;
                    return defaultValue;
                }
                if (typeof(T) == typeof(bool))
                {
                    if (v is bool b) return (T)(object)b;
                    if (bool.TryParse(v.ToString(), out var parsed)) return (T)(object)parsed;
                    return defaultValue;
                }
                if (typeof(T) == typeof(float))
                {
                    if (v is float f) return (T)(object)f;
                    if (float.TryParse(v.ToString(), out var parsed)) return (T)(object)parsed;
                    return defaultValue;
                }
                // 기타 타입은 Convert 시도
                return (T)Convert.ChangeType(v, typeof(T));
            }
            catch
            {
                return defaultValue;
            }
        }

        // DataTable에 행 추가(공용)
        private void AddVehicleRow(string number, DateTime time, bool illegal, float conf, string loc)
        {
            _vehicleTable.Rows.Add(number, time, illegal, conf, loc);
        }

        // ─────────────────────────────────────────────────────────────
        // 지도 초기화 / 마커 / 좌표 이동
        // ─────────────────────────────────────────────────────────────
        private void InitializeMap()
        {
            try
            {
                // 인터넷 및 캐시
                GMaps.Instance.Mode = AccessMode.ServerAndCache;

                // 지도 제공자
                gmap.MapProvider = GMapProviders.GoogleMap;

                // 기본 위치 (서울 시청)
                gmap.Position = new PointLatLng(37.5665, 126.9780);

                // 줌/상호작용
                gmap.MinZoom = 2;
                gmap.MaxZoom = 18;
                gmap.Zoom = 14;
                gmap.CanDragMap = true;
                gmap.DragButton = MouseButtons.Left;
                gmap.ShowCenter = false;
                gmap.Bearing = 0;
                gmap.Dock = DockStyle.None; // 수동 크기
                gmap.MapScaleInfoEnabled = true;
                gmap.EmptyTileColor = Color.LightGray;
                gmap.RoutesEnabled = true;
                gmap.MarkersEnabled = true;

                LogManager.Log(txtLog, "🗺️ 지도 초기화 완료");
            }
            catch (Exception ex)
            {
                LogManager.Log(txtLog, "❌ 지도 초기화 실패: " + ex.Message, LogLevel.Error);
                MessageBox.Show("지도 초기화 중 오류 발생: " + ex.Message, "오류", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void AddMapMarker(double lat, double lng, string tooltip = "", GMarkerGoogleType markerType = GMarkerGoogleType.red_dot)
        {
            var markerOverlay = new GMapOverlay("markers");
            var point = new PointLatLng(lat, lng);
            var marker = new GMarkerGoogle(point, markerType)
            {
                ToolTipText = "",                         // 툴팁 제거
                ToolTipMode = MarkerTooltipMode.Never     // 툴팁 표시 안 함
            };

            markerOverlay.Markers.Add(marker);
            gmap.Overlays.Add(markerOverlay);
        }

        private void MoveMapToLocation(double lat, double lng)
        {
            gmap.Zoom = 17;  // 더 확대
            gmap.Position = new PointLatLng(lat, lng);
            gmap.Refresh();  // 강제 리렌더
        }

        // ─────────────────────────────────────────────────────────────
        // WebRTC/HLS WebView 로드
        // ─────────────────────────────────────────────────────────────
        private void InitializeWebRTCView()
        {
            try
            {
                var htmlPath = Path.Combine(Application.StartupPath, "webrtc", "index.html");
                if (File.Exists(htmlPath))
                {
                    _webrtcUri = new Uri("file:///" + htmlPath.Replace("\\", "/"));
                    webView21.Source = _webrtcUri;
                    LogManager.Log(txtLog, $"📺 HLS 영상 페이지 로드 완료: {htmlPath}");
                }
                else
                {

                    LogManager.Log(txtLog, $"❌ index.html 파일을 찾을 수 없습니다: {htmlPath}", LogLevel.Warning);
                }

                // CoreWebView2 준비 이벤트(이미 연결돼 있다면 중복 방지)
                webView21.CoreWebView2InitializationCompleted -= WebView21_CoreWebView2InitializationCompleted;
                webView21.CoreWebView2InitializationCompleted += WebView21_CoreWebView2InitializationCompleted;

                // 네비 완료되면 오류 카운터 리셋
                webView21.NavigationCompleted -= WebView21_NavigationCompleted;
                webView21.NavigationCompleted += WebView21_NavigationCompleted;
            }
            catch (Exception ex)
            {
                LogManager.Log(txtLog, $"❌ WebView2 로딩 중 오류 발생: {ex.Message}", LogLevel.Error);
            }
        }

        private void WebView21_CoreWebView2InitializationCompleted(object sender, Microsoft.Web.WebView2.Core.CoreWebView2InitializationCompletedEventArgs e)
        {
            if (e.IsSuccess)
            {
                StartWebAutoRefresh(_webAutoRefreshMs); // ✅ 초기화되면 자동 새로고침 시작
            }
            else
            {
                LogManager.Log(txtLog, $"⚠️ WebView2 초기화 실패: {e.InitializationException?.Message}", LogLevel.Warning);
            }
        }

        private void WebView21_NavigationCompleted(object sender, Microsoft.Web.WebView2.Core.CoreWebView2NavigationCompletedEventArgs e)
        {
            _webConsecutiveErrors = 0; // 새로 로드 성공 시 에러 카운터 초기화
        }

        private void StartWebAutoRefresh(int intervalMs = 15000)
        {
            if (webRefreshTimer == null)
            {
                webRefreshTimer = new Timer();
                webRefreshTimer.Tick += WebRefreshTimer_Tick;
            }
            _webAutoRefreshMs = Math.Max(2000, intervalMs); // 최소 2초
            webRefreshTimer.Interval = _webAutoRefreshMs;
            webRefreshTimer.Start();
            LogManager.Log(txtLog, $"🔁 WebRTC 자동 새로고침 시작 ({_webAutoRefreshMs}ms)");
        }

        private void StopWebAutoRefresh()
        {
            if (webRefreshTimer != null)
            {
                webRefreshTimer.Stop();
                LogManager.Log(txtLog, "⏸️ WebRTC 자동 새로고침 중지");
            }
        }

        private async void WebRefreshTimer_Tick(object sender, EventArgs e)
        {
            try
            {
                // Core 준비 안됐으면 준비시키고 패스
                if (webView21.CoreWebView2 == null)
                {
                    await webView21.EnsureCoreWebView2Async();
                    return;
                }

                // WebRefreshTimer_Tick 안의 js 문자열을 이렇게 교체
                string js = @"
(function(){
  function isPlaying(v){
    try{
      var ok = (v.currentTime > 0) && !v.paused && !v.ended && v.readyState >= 2;
      if(!ok && !v.paused && v.readyState >= 2) ok = true;
      return !!ok;
    }catch(e){ return false; }
  }
  function collectVideosDeep(root, out){
    try{
      var vids = root.querySelectorAll('video, .video-js video, video.vjs-tech');
      vids.forEach(v => out.push(v));
      var treeWalker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT, null);
      while(treeWalker.nextNode()){
        var el = treeWalker.currentNode;
        if (el.shadowRoot) collectVideosDeep(el.shadowRoot, out);
      }
    }catch(e){}
  }
  function scanDocument(doc){
    var arr = [];
    collectVideosDeep(doc, arr);
    var ifs = doc.querySelectorAll('iframe');
    for (var i=0;i<ifs.length;i++){
      try{
        if (ifs[i].contentWindow && ifs[i].contentDocument){
          collectVideosDeep(ifs[i].contentDocument, arr);
        }
      }catch(e){}
    }
    return arr;
  }
  try{
    var videos = scanDocument(document);
    if (videos.length === 0) return {state:'no-video', count:0};
    var anyPlaying = videos.some(isPlaying);
    if (anyPlaying) return {state:'playing', count:videos.length};
    var anyEnded   = videos.some(v => v.ended);
    var anyPaused  = videos.some(v => v.paused);
    var anyBuffer  = videos.some(v => v.readyState <= 2);
    var s = anyEnded ? 'ended' : (anyPaused ? 'paused' : (anyBuffer ? 'buffering' : 'unknown'));
    return {state:s, count:videos.length};
  }catch(e){
    return {state:'error', err: String(e)};
  }
})();";



                string stateJson = await webView21.CoreWebView2.ExecuteScriptAsync(js);
                if (!string.IsNullOrEmpty(stateJson))
                {
                    try
                    {
                        var stateObj = JObject.Parse(stateJson);
                        string state = stateObj["state"]?.ToString();
                        int count = stateObj["count"]?.ToObject<int?>() ?? 0;

                        if (state == "playing")
                        {
                            _webConsecutiveErrors = 0;
                            // ...정상 처리...
                        }
                        else
                        {
                            // reload 처리
                        }
                    }
                    catch
                    {
                        LogManager.Log(txtLog, $"⚠️ WebView 상태 파싱 실패: {stateJson}", LogLevel.Warning);
                    }
                }
            }
            catch (Exception ex)
            {
                _webConsecutiveErrors++;
                LogManager.Log(txtLog, $"⚠️ WebView 새로고침 오류: {ex.Message}", LogLevel.Warning);

                // JS 실패 시 캐시버스터로 강제 새로고침(파일/HTTP 모두 동작)
                try
                {
                    var baseUri = _webrtcUri ?? webView21.Source;
                    if (baseUri != null)
                    {
                        var bust = $"{baseUri}{(baseUri.Query.Length == 0 ? "?" : "&")}t={DateTimeOffset.Now.ToUnixTimeSeconds()}";
                        webView21.Source = new Uri(bust);
                        LogManager.Log(txtLog, "♻️ 캐시버스터로 강제 새로고침");
                    }
                }
                catch { /* ignore */ }

                // 연속 오류가 누적되면 일시적으로 주기를 늘려 서버/네트워크 부담 완화
                if (_webConsecutiveErrors >= 3 && webRefreshTimer.Interval == _webAutoRefreshMs)
                {
                    webRefreshTimer.Interval = Math.Min(_webAutoRefreshMs * 4, 120000); // 최대 2분
                    LogManager.Log(txtLog, $"⏳ 임시 백오프 적용: {webRefreshTimer.Interval}ms");
                }
            }
        }


        /// <summary>
        /// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /// </summary>


        // ─────────────────────────────────────────────────────────────
        // 테스트 이미지 로드
        // ─────────────────────────────────────────────────────────────
        private void TryLoadTestImage()
        {
            string testImage = @"C:\Test Images\sample.jpg";
            if (pictureBoxVehicle == null) return;

            if (File.Exists(testImage))
            {
                pictureBoxVehicle.Image = new Bitmap(testImage);
                pictureBoxVehicle.SizeMode = PictureBoxSizeMode.Zoom;
                LogManager.Log(txtLog, $"✔ 이미지 로드 성공: {testImage}");
            }
            else
            {
                pictureBoxVehicle.Image = Properties.Resources.no_image; // 리소스 등록 필요
                pictureBoxVehicle.SizeMode = PictureBoxSizeMode.Zoom;
                LogManager.Log(txtLog, "❌ 테스트 이미지 없음. 대체 이미지 표시", LogLevel.Warning);
            }
        }

        // ─────────────────────────────────────────────────────────────
        // DataGridView 셀 클릭 → 지도 이동
        // ─────────────────────────────────────────────────────────────
        private void dgvData_CellClick(object sender, DataGridViewCellEventArgs e)
        {
            if (e.RowIndex < 0 || e.ColumnIndex < 0) return;

            string locationStr = dgvData.Rows[e.RowIndex].Cells["location"].Value?.ToString();
            string vehicleNumber = dgvData.Rows[e.RowIndex].Cells["vehicleNumber"].Value?.ToString();
            if (string.IsNullOrWhiteSpace(locationStr) || string.IsNullOrWhiteSpace(vehicleNumber)) return;

            if (!TryParseCoordinates(locationStr, out double lat, out double lng))
            {
                txtLog.AppendText($"[오류] {vehicleNumber} 차량의 좌표 정보를 파싱할 수 없습니다.\n");
                return;
            }

            MoveMapToLocation(lat, lng);
            txtLog.AppendText($"{vehicleNumber} 차량은 {lat}, {lng} 좌표에 위치하고 있습니다.\n");
        }

        private void dgvData_CellFormatting(object sender, DataGridViewCellFormattingEventArgs e)
        {
            if (dgvData.Columns["isIllegal"] != null && e.RowIndex >= 0)
            {
                bool illegal = Convert.ToBoolean(dgvData.Rows[e.RowIndex].Cells["isIllegal"].Value ?? false);
                dgvData.Rows[e.RowIndex].DefaultCellStyle.BackColor = illegal ? Color.MistyRose : Color.White;
            }
        }

        // 좌표 파싱
        private bool TryParseCoordinates(string locationStr, out double lat, out double lng)
        {
            lat = lng = 0;
            var matches = Regex.Matches(locationStr, @"[-+]?[0-9]*\.?[0-9]+");
            if (matches.Count >= 2 &&
                double.TryParse(matches[0].Value, out lat) &&
                double.TryParse(matches[1].Value, out lng))
            {
                return true;
            }
            return false;
        }

        // ─────────────────────────────────────────────────────────────
        // DB 자동 갱신 + 팝업
        // ─────────────────────────────────────────────────────────────
        private void StartAutoUpdate()
        {
            dataUpdateTimer = new Timer();
            dataUpdateTimer.Interval = 60000; // 60초 간격
            dataUpdateTimer.Tick += DataUpdateTimer_Tick;
            dataUpdateTimer.Start();
            LogManager.Log(txtLog, "🔄 실시간 DB 업데이트 타이머 시작됨");
        }

        private void DataUpdateTimer_Tick(object sender, EventArgs e)
        {
            try
            {
                var newRecords = VehicleLogger.FetchNewVehicleData(knownVehicleNumbers);

                foreach (var record in newRecords)
                {
                    // ✅ DataTable에 추가 (실시간 공유)
                    DateTime parsedTime = DateTime.TryParse(record.DetectedTime, out var t) ? t : DateTime.Now;
                    AddVehicleRow(record.VehicleNumber, parsedTime, false, 0f, record.Location);

                    // ✅ 지도 마커 추가
                    if (TryParseCoordinates(record.Location, out double lat, out double lng))
                        AddMapMarker(lat, lng, "🚗 신규 차량", GMarkerGoogleType.green);

                    // ✅ 중복 방지
                    if (!knownVehicleNumbers.Contains(record.VehicleNumber))
                        knownVehicleNumbers.Add(record.VehicleNumber);

                    // ✅ 로그
                    LogManager.Log(txtLog, $"📌 차량 감지됨: {record.VehicleNumber} @ {record.Location}");

                    // ✅ 팝업 알림 (현재는 비활성)
                    // ShowVehicleAlertPopup(record.VehicleNumber, record.Location);
                }
            }
            catch (Exception ex)
            {
                LogManager.Log(txtLog, $"❌ DB 자동 갱신 실패: {ex.Message}", LogLevel.Error);
            }
        }

        private void ShowVehicleAlertPopup(string vehicleNumber, string location)
        {
            string message = $"🚨 신규 차량 감지됨: {vehicleNumber}\n위치: {location}";
            // MessageBox.Show(message, "차량 감지 알림", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        private void webView21_Click(object sender, EventArgs e)
        {
            var url = "http://127.0.0.1:8099/"; // 또는 /video 직접
            webView21.Source = new Uri(url);
            LogManager.Log(txtLog, $"📺 WebView2 로드: {url}");
        }

        // ─────────────────────────────────────────────────────────────
        // “더 크게 보기” 버튼: 단일 창 재사용 + 실시간 바인딩 공유
        // ─────────────────────────────────────────────────────────────
        private void btnViewLarge_Click(object sender, EventArgs e)
        {
            if (_listForm == null || _listForm.IsDisposed)
            {
                _listForm = new TestApp.Modules.VehicleListForm();
                _listForm.FormClosed += (s, _) => _listForm = null;

                // VehicleListForm에 SetDataBinding(BindingSource bs) 메서드가 있어야 함
                _listForm.SetDataBinding(_vehicleBS);

                _listForm.Show(this);
                _listForm.WindowState = FormWindowState.Maximized;
            }
            else
            {
                if (_listForm.WindowState == FormWindowState.Minimized)
                    _listForm.WindowState = FormWindowState.Normal;
                _listForm.Activate();
                _listForm.BringToFront();
            }
        }
    }
}
