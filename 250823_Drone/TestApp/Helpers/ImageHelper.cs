using OpenCvSharp;
using OpenCvSharp.Extensions;
using System;
using System.Drawing;
using System.IO;
using System.Windows.Forms;

namespace TestApp.Helpers
{
    public static class ImageHelper
    {
        public static void ShowVehicleImage(PictureBox pictureBox, string imagePath)
        {
            if (string.IsNullOrEmpty(imagePath) || !File.Exists(imagePath))
                return;

            try
            {
                using var mat = Cv2.ImRead(imagePath); // OpenCV로 이미지 읽기
                if (mat.Empty())
                    return;

                var bitmap = BitmapConverter.ToBitmap(mat); // Bitmap 변환
                pictureBox.Invoke((MethodInvoker)(() =>
                {
                    pictureBox.Image = bitmap;
                }));
            }
            catch (Exception ex)
            {
                MessageBox.Show("이미지 로딩 실패: " + ex.Message);
            }
        }
    }
}
