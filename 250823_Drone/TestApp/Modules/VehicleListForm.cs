using System;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;

namespace TestApp.Modules
{
    public partial class VehicleListForm : Form
    {
        private readonly DataGridView dgvLarge = new DataGridView();

        public VehicleListForm()
        {
            InitializeComponent();

            // 기존 AboutBox 템플릿의 컨트롤은 모두 제거하고 테이블만 사용
            this.Controls.Clear();

            this.Text = "차량 리스트 (확대 보기)";
            this.WindowState = FormWindowState.Maximized; // 크게 보기
            this.KeyPreview = true;
            this.KeyDown += (s, e) => { if (e.KeyCode == Keys.Escape) this.Close(); }; // ESC로 닫기

            // DataGridView 기본 설정
            dgvLarge.Dock = DockStyle.Fill;
            dgvLarge.AllowUserToAddRows = false;
            dgvLarge.ReadOnly = true;
            dgvLarge.MultiSelect = true;
            dgvLarge.SelectionMode = DataGridViewSelectionMode.FullRowSelect;
            dgvLarge.AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill;
            dgvLarge.RowHeadersVisible = false;
            dgvLarge.BorderStyle = BorderStyle.None;
            dgvLarge.EnableHeadersVisualStyles = true;

            this.Controls.Add(dgvLarge);
        }

        /// <summary>
        /// 메인 폼의 DataGridView 내용을 이 창에 표시.
        /// 바인딩(=DataSource)된 경우엔 같은 DataSource를 공유하고,
        /// 수동으로 Rows.Add 한 경우에는 스냅샷 복사.
        /// </summary>
        public void SetData(DataGridView src)
        {
            if (src == null) return;

            if (src.DataSource != null)
            {
                // 같은 데이터 소스를 공유 (실시간 반영)
                dgvLarge.DataSource = null; // 안전을 위해 한 번 비워줌
                dgvLarge.AutoGenerateColumns = true;
                dgvLarge.DataSource = src.DataSource;
            }
            else
            {
                // 수동행: 스냅샷 복사
                dgvLarge.DataSource = null;
                dgvLarge.Columns.Clear();

                // 컬럼 복사 (HeaderText 유지)
                foreach (DataGridViewColumn c in src.Columns)
                {
                    var col = new DataGridViewTextBoxColumn
                    {
                        Name = c.Name,
                        HeaderText = c.HeaderText,
                        ReadOnly = true
                    };
                    dgvLarge.Columns.Add(col);
                }

                // 행 복사
                foreach (DataGridViewRow r in src.Rows)
                {
                    if (r.IsNewRow) continue;
                    var vals = new object[src.Columns.Count];
                    for (int i = 0; i < src.Columns.Count; i++)
                        vals[i] = r.Cells[i].Value;
                    dgvLarge.Rows.Add(vals);
                }
            }

            // 보기 좋게 정렬/스타일 약간
            dgvLarge.AutoResizeColumns();
            dgvLarge.AutoResizeRows();
        }

        // 디자이너에서 연결돼 있을 수 있는 Load 핸들러는 비워둠
        private void VehicleListForm_Load(object sender, EventArgs e) { }



        //BindingSource 기반 실시간 동기화 + “더 크게 보기” 단일 창 재사용
        public void SetDataBinding(BindingSource bs)
        {
            if (bs == null) return;
            var dgv = this.Controls.OfType<DataGridView>().FirstOrDefault();
            if (dgv == null) return;
            dgv.DataSource = null;
            dgv.AutoGenerateColumns = true;
            dgv.DataSource = bs;   // ✅ 메인과 같은 소스 공유 → 실시간 반영
        }

        private void dataGridView1_CellContentClick(object sender, DataGridViewCellEventArgs e)
        {

        }
    }
}
