import subprocess
import os

class CameraManager:
    def __init__(self, width=1280, height=720, device="/dev/video0"):
        self.width = width
        self.height = height
        self.yuv_frame_size = self.width * self.height * 3 // 2  # YUV420p (I420)

        # Arducam YUV420p 프레임 송출 파이프 실행
        self.proc = subprocess.Popen(
            [
                'python3',
                '/home/jetson/streaming/arducam_tostdout.py',
                '--width', str(self.width),
                '--height', str(self.height)
            ],
            stdout=subprocess.PIPE,
            bufsize=self.yuv_frame_size
        )

    def read(self):
        raw = self.proc.stdout.read(self.yuv_frame_size)
        if len(raw) != self.yuv_frame_size:
            return False, None
        return True, raw

    def is_opened(self):
        return self.proc and self.proc.poll() is None

    def release(self):
        if self.proc:
            self.proc.terminate()
            self.proc.wait()


def main():
    RTMP_URL = "rtmp://54.180.119.91/live/stream"
    WIDTH, HEIGHT, FPS = 1280, 720, 30
    FRAME_SIZE = WIDTH * HEIGHT * 3 // 2

    ffmpeg_cmd = [
        '/usr/local/bin/ffmpeg',
        '-f', 'rawvideo',
        '-pix_fmt', 'yuv420p',
        '-s', f'{WIDTH}x{HEIGHT}',
        '-r', str(FPS),
        '-i', '-',
        '-c:v', 'h264_nvmpi',
        '-b:v', '4M',
        '-g', '15',
        '-bf', '0',
        '-f', 'flv',
        RTMP_URL
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    cam = CameraManager(width=WIDTH, height=HEIGHT)

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                continue
            process.stdin.write(frame)
    except KeyboardInterrupt:
        print("중단됨")
    finally:
        cam.release()
        if process.stdin:
            process.stdin.close()
        process.wait()


if __name__ == "__main__":
    main()
