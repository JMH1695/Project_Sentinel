using System;
using System.Windows.Forms;

namespace TestApp.Helpers
{
    public static class UIHelper
    {
        /// <summary>
        /// UI 컨트롤을 스레드 안전하게 업데이트합니다.
        /// </summary>
        public static void SafeUpdate(Control control, Action action)
        {
            if (control == null || action == null) return;

            if (control.InvokeRequired)
                control.Invoke((MethodInvoker)(() => action()));
            else
                action();
        }
    }
}
