using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;

namespace TestApp.Helpers
{
    public enum LogLevel
    {
        Info,
        Warning,
        Error
    }

    public static class LogManager
    {
        private static readonly List<(string line, LogLevel level)> logHistory = new List<(string, LogLevel)>();

        public static void Log(RichTextBox txtLog, string message, LogLevel level = LogLevel.Info)
        {
            if (txtLog == null) return;

            string logLine = $"[{DateTime.Now:HH:mm:ss}] {message}\r\n";
            logHistory.Add((logLine, level));  // 히스토리에 저장

            Color color;
            switch (level)
            {
                case LogLevel.Warning:
                    color = Color.Orange;
                    break;
                case LogLevel.Error:
                    color = Color.Red;
                    break;
                default:
                    color = Color.White;
                    break;
            }

            txtLog.Invoke((MethodInvoker)(() =>
            {
                int start = txtLog.TextLength;
                txtLog.AppendText(logLine);
                txtLog.Select(start, logLine.Length);
                txtLog.SelectionColor = color;
                txtLog.ScrollToCaret();
            }));
        }

        public static void AppendImportant(RichTextBox txtLog, string message)
        {
            Log(txtLog, "⚠️ " + message, LogLevel.Warning);
        }

        public static void LoadHistory(RichTextBox txtLog)
        {
            if (txtLog == null) return;

            txtLog.Invoke((MethodInvoker)(() =>
            {
                foreach (var item in logHistory)
                {
                    string line = item.line;
                    LogLevel level = item.level;

                    Color color;
                    switch (level)
                    {
                        case LogLevel.Warning:
                            color = Color.Orange;
                            break;
                        case LogLevel.Error:
                            color = Color.Red;
                            break;
                        default:
                            color = Color.White;
                            break;
                    }

                    int start = txtLog.TextLength;
                    txtLog.AppendText(line);
                    txtLog.Select(start, line.Length);
                    txtLog.SelectionColor = color;
                }

                txtLog.ScrollToCaret();
            }));
        }
    }
}
