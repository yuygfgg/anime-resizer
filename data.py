import os
import subprocess
import random
import uuid
import numpy as np
import logging
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# ANSI 转义序列，用于彩色输出日志
COLOR_GREEN = "\033[92m"
COLOR_RESET = "\033[0m"

# 配置日志格式与级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 全局计数器，用于统计已提取的有效帧数
global_frame_counter = 0
counter_lock = None

# 定义检测帧亮度的函数，返回 True 表示帧亮度正常，False 表示过黑或过白
def is_valid_frame(image_path, brightness_threshold=0.05):
    try:
        img = Image.open(image_path).convert('L')  # 转换为灰度图像
        histogram = np.array(img.histogram())
        total_pixels = histogram.sum()

        # 统计低亮度和高亮度像素的比例
        low_brightness_ratio = histogram[:10].sum() / total_pixels
        high_brightness_ratio = histogram[-10:].sum() / total_pixels

        # 日志输出帧信息
        logging.debug(f"检测帧: {image_path}, 低亮度比例: {low_brightness_ratio:.3f}, 高亮度比例: {high_brightness_ratio:.3f}")

        # 判断帧是否过黑或过白
        if low_brightness_ratio > brightness_threshold:
            logging.info(f"跳过帧: {image_path}，原因：过黑")
            return False
        if high_brightness_ratio > brightness_threshold:
            logging.info(f"跳过帧: {image_path}，原因：过白")
            return False

        return True
    except Exception as e:
        logging.error(f"检测帧亮度失败: {image_path}, 错误: {e}")
        return False

# 随机提取帧的函数
def extract_random_frames(video_path, output_dir, num_frames=5):
    global global_frame_counter
    global counter_lock

    try:
        os.makedirs(output_dir, exist_ok=True)

        # 我们不使用 ffprobe，而是直接用 ffmpeg 随机抽取帧
        total_random_attempts = num_frames * 5  # 给定一定的重试次数，以确保最终能获得足够的有效帧
        valid_frames_extracted = 0

        for attempt in range(total_random_attempts):
            if valid_frames_extracted >= num_frames:
                break

            # 随机选择一个时间点来提取帧
            random_time = random.uniform(0, 15*20)  # 假设随机选择前 10 秒，适当调整范围
            unique_filename = str(uuid.uuid4())  # 生成 UUID 作为文件名
            output_frame_path = os.path.join(output_dir, f'{unique_filename}.png')

            # 使用 ffmpeg 从随机时间点提取帧
            extract_cmd = [
                'ffmpeg', '-ss', f'{random_time}', '-i', video_path,
                '-vf', 'scale=-1:1080',
                '-vframes', '1', output_frame_path, '-hide_banner', '-loglevel', 'error'
            ]
            logging.info(f"尝试提取随机帧: 时间 {random_time:.2f}s -> {output_frame_path}")
            subprocess.run(extract_cmd, check=True)

            # 检测该帧是否为有效帧
            if is_valid_frame(output_frame_path):
                valid_frames_extracted += 1
                logging.info(f"有效帧: {output_frame_path} (已提取 {valid_frames_extracted}/{num_frames} 帧)")

                # 更新全局有效帧计数器
                with counter_lock:
                    global_frame_counter += 1
                    if global_frame_counter % 100 == 0:
                        # 每 100 帧输出彩色日志
                        logging.info(f"{COLOR_GREEN}已成功提取 {global_frame_counter} 帧！{COLOR_RESET}")
            else:
                os.remove(output_frame_path)  # 删除无效帧
                logging.info(f"删除无效帧: {output_frame_path}")

        if valid_frames_extracted < num_frames:
            logging.warning(f"视频 {video_path} 中没有足够的有效帧 (已提取 {valid_frames_extracted}/{num_frames})")

        logging.info(f"视频处理完成: {video_path}")
    except Exception as e:
        logging.error(f"帧提取失败: {video_path}, 错误: {e}")

# 递归查找 .mkv 文件并提取帧
def find_mkv_files_and_extract_frames(root_dir, output_dir, num_frames=5, max_workers=4):
    global counter_lock
    video_files = []

    # 遍历查找所有 .mkv 文件
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.mkv'):
                video_path = os.path.join(dirpath, filename)
                video_files.append(video_path)

    # 使用多线程处理，每个视频文件一个线程
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        from threading import Lock
        counter_lock = Lock()  # 初始化一个线程锁，用于保护计数器的更新
        futures = []
        for video_path in video_files:
            futures.append(executor.submit(extract_random_frames, video_path, output_dir, num_frames))

        # 处理所有的 future 任务
        for future in as_completed(futures):
            try:
                future.result()  # 如果有异常，将在这里抛出
            except Exception as e:
                logging.error(f"处理任务失败: {e}")

if __name__ == '__main__':
    # 定义根目录和输出目录
    root_directory = '.'  # 当前目录
    output_directory = '/Volumes/untitled/data'  # 输出目录
    num_frames_to_extract = 2  # 每个视频提取的帧数
    max_threads = 8  # 定义最大线程数

    logging.info("开始递归查找 .mkv 文件并提取随机帧")
    find_mkv_files_and_extract_frames(root_directory, output_directory, num_frames=num_frames_to_extract, max_workers=max_threads)
    logging.info("所有视频处理完成")
