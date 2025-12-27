# -*- coding: utf-8 -*-
"""
MP4 HintBox Pro - Parallel/Serial selectable processing
Refactored from mp4hint_1.0.4.py to allow choosing between:
 - Serial processing (original ProcessingWorker using QThread)
 - Parallel processing (ProcessWorker using QRunnable + QThreadPool)

Save this file as mp4hint_parallelized.py and run with a Python environment
that has PyQt5, psutil, and ffmpeg/MP4Box available in PATH if you want to test.
"""
import sys
import os
import shutil
import subprocess
import json
import logging
import uuid
import struct
import platform
import psutil
from datetime import datetime
from typing import List, Optional, Tuple
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QListWidget, QPushButton, QLabel, QProgressBar,
    QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QMessageBox,
    QTextEdit, QSplitter, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QThreadPool, QRunnable, QObject, QUrl
from PyQt5.QtGui import QFont, QPainter, QPen, QColor

# 콘솔 숨기기(Windows)
startupinfo = None
if sys.platform == "win32":
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

# 로깅 설정
def setup_logging():
    log_file = os.path.join(os.getcwd(), "mp4_processor.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# 설정 클래스
class Config:
    MAX_PARALLEL = 4
    BACKUP_ENABLED = False
    TEMP_DIR = os.path.join(os.getcwd(), "temp_mp4_processor")
    BACKUP_DIR = os.path.join(os.getcwd(), "backup_mp4_processor")
    TIMEOUT = 3600
    PROCESS_MODE = "serial"  # "serial" or "parallel"

    def __init__(self):
        os.makedirs(self.TEMP_DIR, exist_ok=True)
        if self.BACKUP_ENABLED:
            os.makedirs(self.BACKUP_DIR, exist_ok=True)

# Drag & Drop QListWidget
class DragDropListWidget(QListWidget):
    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DropOnly)
        self.drag_active = False

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            valid_files = [url.toLocalFile() for url in urls if url.isLocalFile() and url.toLocalFile().lower().endswith('.mp4')]
            if valid_files:
                event.acceptProposedAction()
                self.drag_active = True
                self.update()
            else:
                event.ignore()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.drag_active = False
        self.update()
        event.accept()

    def dropEvent(self, event):
        self.drag_active = False
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            file_paths = [url.toLocalFile() for url in urls if url.isLocalFile() and url.toLocalFile().lower().endswith('.mp4')]
            if file_paths:
                self.files_dropped.emit(file_paths)
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.drag_active:
            painter = QPainter(self.viewport())
            painter.setRenderHint(QPainter.Antialiasing)
            pen = QPen(QColor(0, 123, 255, 180), 3, Qt.DashLine)
            painter.setPen(pen)
            rect = self.viewport().rect()
            rect.adjust(5, 5, -5, -5)
            painter.drawRect(rect)
            if self.count() == 0:
                painter.setPen(QColor(108, 117, 125))
                painter.drawText(rect, Qt.AlignCenter,
                    "MP4 파일을 여기에 드래그하세요\n또는 '파일 추가' 버튼을 사용하세요")
            painter.end()

# MP4 utilities
class MP4Utils:
    @staticmethod
    def check_hint_track_with_mp4box(filepath: str, mp4box_path: str) -> bool:
        try:
            result = subprocess.run(
                [mp4box_path, '-info', filepath],
                capture_output=True, text=True,
                encoding='utf-8', errors='ignore',
                timeout=30, startupinfo=startupinfo
            )
            if result.returncode != 0:
                logger.warning(f"MP4Box info failed for {filepath}: {result.stderr}")
                return False
            lines = result.stdout.split('\n')
            for line in lines:
                line_lower = line.lower()
                if ('hint' in line_lower and 'track' in line_lower) or 'hinting' in line_lower:
                    logger.info(f"Hint track found in {filepath}")
                    return True
            return False
        except subprocess.TimeoutExpired:
            logger.error(f"MP4Box timeout for {filepath}")
            return False
        except FileNotFoundError:
            logger.error(f"MP4Box not found at {mp4box_path}")
            return False
        except Exception as e:
            logger.error(f"MP4Box hint check error for {filepath}: {e}")
            return False

    @staticmethod
    def check_faststart_with_ffprobe(filepath: str, ffprobe_path: str) -> bool:
        try:
            result = subprocess.run([
                ffprobe_path, '-v', 'quiet', '-print_format', 'json',
                '-show_format', filepath
            ], capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=30, startupinfo=startupinfo)
            if result.returncode != 0:
                logger.warning(f"FFprobe format check failed for {filepath}")
                return MP4Utils._check_faststart_binary(filepath)
            data = json.loads(result.stdout)
            format_info = data.get('format', {})
            start_time = float(format_info.get('start_time', 1.0))
            duration = float(format_info.get('duration', 0))
            if duration > 0 and abs(start_time) < 0.1:
                return MP4Utils._check_faststart_binary(filepath)
            return False
        except Exception as e:
            logger.error(f"FFprobe faststart check error for {filepath}: {e}")
            return MP4Utils._check_faststart_binary(filepath)

    @staticmethod
    def _check_faststart_binary(filepath: str) -> bool:
        try:
            with open(filepath, 'rb') as f:
                data = f.read(8192)
            if len(data) < 16:
                return False
            pos = 0
            found_ftyp = False
            found_moov = False
            found_mdat = False
            while pos < len(data) - 8:
                if pos + 8 > len(data):
                    break
                try:
                    atom_size = int.from_bytes(data[pos:pos+4], 'big')
                    atom_type = data[pos+4:pos+8].decode('ascii', errors='ignore')
                    if atom_type == 'ftyp':
                        found_ftyp = True
                    elif atom_type == 'moov':
                        found_moov = True
                        if not found_mdat:
                            logger.info(f"Faststart detected for {filepath} (moov before mdat)")
                            return True
                    elif atom_type == 'mdat':
                        found_mdat = True
                        if not found_moov:
                            logger.info(f"Non-faststart detected for {filepath} (mdat before moov)")
                            return False
                    if atom_size < 8 or atom_size > len(data):
                        break
                    pos += atom_size
                except (ValueError, UnicodeDecodeError, struct.error):
                    pos += 1
                    continue
            if not found_moov and not found_mdat:
                return MP4Utils._deep_faststart_check(filepath)
            return False
        except Exception as e:
            logger.error(f"Binary faststart check error for {filepath}: {e}")
            return False

    @staticmethod
    def _deep_faststart_check(filepath: str) -> bool:
        try:
            with open(filepath, 'rb') as f:
                data = f.read(65536)
            if len(data) < 16:
                return False
            pos = 0
            moov_pos = -1
            mdat_pos = -1
            while pos < len(data) - 8:
                try:
                    atom_size = int.from_bytes(data[pos:pos+4], 'big')
                    atom_type = data[pos+4:pos+8].decode('ascii', errors='ignore')
                    if atom_type == 'moov' and moov_pos == -1:
                        moov_pos = pos
                    elif atom_type == 'mdat' and mdat_pos == -1:
                        mdat_pos = pos
                    if moov_pos != -1 and mdat_pos != -1:
                        break
                    if atom_size < 8 or pos + atom_size > len(data):
                        pos += 8
                    else:
                        pos += atom_size
                except (ValueError, UnicodeDecodeError, struct.error):
                    pos += 1
                    continue
            if moov_pos != -1 and mdat_pos != -1:
                is_faststart = moov_pos < mdat_pos
                logger.info(f"Deep faststart check for {filepath}: {is_faststart}")
                return is_faststart
            elif moov_pos != -1 and mdat_pos == -1:
                logger.info(f"Likely faststart for {filepath} (moov found early, mdat not in first 64KB)")
                return True
            else:
                logger.warning(f"Could not determine faststart status for {filepath}")
                return False
        except Exception as e:
            logger.error(f"Deep faststart check error for {filepath}: {e}")
            return False

    @staticmethod
    def get_comprehensive_status(filepath: str, mp4box_path: str, ffprobe_path: str = None) -> dict:
        status = {
            'faststart': False,
            'hint_track': False,
            'faststart_status': 'faststart 필요',
            'hint_status': 'hint track 필요'
        }
        try:
            if ffprobe_path:
                status['faststart'] = MP4Utils.check_faststart_with_ffprobe(filepath, ffprobe_path)
            else:
                status['faststart'] = MP4Utils._check_faststart_binary(filepath)
            if status['faststart']:
                status['faststart_status'] = 'faststart 적용됨'
            status['hint_track'] = MP4Utils.check_hint_track_with_mp4box(filepath, mp4box_path)
            if status['hint_track']:
                status['hint_status'] = 'hint track 존재'
        except Exception as e:
            logger.error(f"Comprehensive status check error for {filepath}: {e}")
            status['faststart_status'] = '체크 실패'
            status['hint_status'] = '체크 실패'
        return status

    @staticmethod
    def get_processing_status(filepath: str, engine: str, mp4box_path: str, ffprobe_path: str = None) -> str:
        try:
            comprehensive = MP4Utils.get_comprehensive_status(filepath, mp4box_path, ffprobe_path)
            if engine.lower() == "ffmpeg":
                if comprehensive['faststart']:
                    if comprehensive['hint_track']:
                        return "faststart 적용됨 + hint track"
                    else:
                        return "faststart 적용됨"
                else:
                    if comprehensive['hint_track']:
                        return "faststart 필요 (hint track 존재)"
                    else:
                        return "faststart 필요"
            elif engine.lower() == "mp4box":
                if comprehensive['hint_track']:
                    if comprehensive['faststart']:
                        return "hint track 존재 + faststart"
                    else:
                        return "hint track 존재"
                else:
                    if comprehensive['faststart']:
                        return "hint track 필요 (faststart 적용됨)"
                    else:
                        return "hint track 필요"
            else:
                return "알 수 없는 엔진"
        except Exception as e:
            logger.error(f"Status check error for {filepath}: {e}")
            return "체크 실패"

    @staticmethod
    def create_backup(filepath: str, backup_dir: str) -> Optional[str]:
        try:
            filename = os.path.basename(filepath)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{timestamp}_{filename}"
            backup_path = os.path.join(backup_dir, backup_name)
            shutil.copy2(filepath, backup_path)
            logger.info(f"Backup created: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Backup creation failed for {filepath}: {e}")
            return None

# Signals for QRunnable workers
class WorkerSignals(QObject):
    result = pyqtSignal(str, str)  # filepath, status
    error = pyqtSignal(str, str)   # filepath, error
    finished = pyqtSignal(str, bool)  # filepath, success
    log = pyqtSignal(str)

# CheckWorker remains (uses QRunnable)
class CheckWorker(QRunnable):
    def __init__(self, filepath: str, engine: str, mp4box_path: str, ffprobe_path: str = None):
        super().__init__()
        self.filepath = filepath
        self.engine = engine
        self.mp4box_path = mp4box_path
        self.ffprobe_path = ffprobe_path
        self.signals = WorkerSignals()

    def run(self):
        try:
            status = MP4Utils.get_processing_status(self.filepath, self.engine, self.mp4box_path, self.ffprobe_path)
            self.signals.result.emit(self.filepath, status)
        except Exception as e:
            self.signals.error.emit(self.filepath, str(e))

# Serial processing worker (original)
class ProcessingWorker(QThread):
    progress_signal = pyqtSignal(int, int, str, str)
    finished_signal = pyqtSignal(bool, str)
    log_signal = pyqtSignal(str)

    def __init__(self, file_list: List[str], engine: str, config: Config,
                 ffmpeg_path: str, mp4box_path: str, ffprobe_path: str,
                 hw_accel_option: str):
        super().__init__()
        self.file_list = file_list
        self.engine = engine
        self.config = config
        self.ffmpeg_path = ffmpeg_path
        self.mp4box_path = mp4box_path
        self.ffprobe_path = ffprobe_path
        self.hw_accel_option = hw_accel_option
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        success_count = 0
        error_count = 0
        skipped_count = 0

        for idx, filepath in enumerate(self.file_list, 1):
            if self._stop_requested:
                self.progress_signal.emit(idx, len(self.file_list), filepath, "취소됨")
                break

            try:
                current_status = MP4Utils.get_processing_status(filepath, self.engine, self.mp4box_path, self.ffprobe_path)

                engine_specific_done = (self.engine.lower() == "ffmpeg" and "faststart 적용됨" in current_status) or \
                                       (self.engine.lower() == "mp4box" and "hint track 존재" in current_status)

                if engine_specific_done:
                    self.progress_signal.emit(idx, len(self.file_list), filepath, "이미 처리됨 ⏭️")
                    skipped_count += 1
                    success_count += 1
                    continue

                self.progress_signal.emit(idx, len(self.file_list), filepath, "처리 중...")

                backup_path = None
                if self.config.BACKUP_ENABLED:
                    backup_path = MP4Utils.create_backup(filepath, self.config.BACKUP_DIR)
                    if not backup_path:
                        self.progress_signal.emit(idx, len(self.file_list), filepath, "백업 실패 ❌")
                        error_count += 1
                        continue

                temp_filename = f"{uuid.uuid4().hex}.mp4"
                temp_path = os.path.join(self.config.TEMP_DIR, temp_filename)

                success = self._process_file(filepath, temp_path)

                if success and os.path.exists(temp_path):
                    shutil.move(temp_path, filepath)
                    self.progress_signal.emit(idx, len(self.file_list), filepath, "완료 ✅")
                    success_count += 1
                else:
                    self.progress_signal.emit(idx, len(self.file_list), filepath, "처리 실패 ❌")
                    error_count += 1

                    if backup_path and os.path.exists(backup_path):
                        try:
                            shutil.copy2(backup_path, filepath)
                            self.log_signal.emit(f"백업에서 복원: {filepath}")
                        except Exception as e:
                            self.log_signal.emit(f"백업 복원 실패: {e}")

                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception as e:
                        self.log_signal.emit(f"임시 파일 삭제 실패: {e}")

            except Exception as e:
                logger.error(f"Processing error for {filepath}: {e}")
                self.progress_signal.emit(idx, len(self.file_list), filepath, f"오류 ❌ ({type(e).__name__})")
                error_count += 1

        if self._stop_requested:
            message = "처리가 취소되었습니다."
        else:
            if skipped_count > 0:
                message = f"처리 완료: 성공 {success_count - skipped_count}개, 건너뜀 {skipped_count}개, 실패 {error_count}개"
            else:
                message = f"처리 완료: 성공 {success_count}개, 실패 {error_count}개"

        self.finished_signal.emit(error_count == 0, message)

def _process_file(self, input_path: str, output_path: str) -> bool:
        try:
            if self.engine.lower() == "ffmpeg":
                # 입력 큐 사이즈를 늘려 대용량 파일 처리 시 버퍼 문제 예방
                cmd = [self.ffmpeg_path, '-y', '-thread_queue_size', '512', '-i', input_path]

                if self.hw_accel_option == 'nvidia':
                    cmd.extend(['-c:v', 'h264_nvenc', '-c:a', 'copy'])
                elif self.hw_accel_option == 'amd':
                    cmd.extend(['-c:v', 'h264_amf', '-c:a', 'copy'])
                elif self.hw_accel_option == 'intel':
                    cmd.extend(['-hwaccel', 'qsv', '-c:v', 'h264_qsv', '-c:a', 'copy'])
                else:
                    cmd.extend(['-c', 'copy'])

                cmd.extend(['-movflags', '+faststart', output_path])
            else:
                cmd = [
                    self.mp4box_path, '-hint', input_path,
                    '-out', output_path
                ]

            self.signals.log.emit(f"실행 명령: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True, text=True,
                encoding='utf-8', errors='ignore',
                timeout=self.config.TIMEOUT, startupinfo=startupinfo
            )

            if result.returncode != 0:
                self.signals.log.emit(f"처리 실패: {result.stderr}")
                return False

            return True

        except subprocess.TimeoutExpired:
            self.signals.log.emit(f"처리 시간 초과: {input_path}")
            return False
        except FileNotFoundError as e:
            self.signals.log.emit(f"실행 파일을 찾을 수 없음: {e}")
            return False
        except Exception as e:
            self.signals.log.emit(f"처리 중 오류: {e}")
            return False

# Parallel single-file worker (QRunnable)
class ProcessWorker(QRunnable):
    def __init__(self, filepath: str, engine: str, config: Config,
                 ffmpeg_path: str, mp4box_path: str, ffprobe_path: str,
                 hw_accel_option: str):
        super().__init__()
        self.filepath = filepath
        self.engine = engine
        self.config = config
        self.ffmpeg_path = ffmpeg_path
        self.mp4box_path = mp4box_path
        self.ffprobe_path = ffprobe_path
        self.hw_accel_option = hw_accel_option
        self.signals = WorkerSignals()
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        # Each ProcessWorker handles exactly one file
        filepath = self.filepath
        try:
            current_status = MP4Utils.get_processing_status(filepath, self.engine, self.mp4box_path, self.ffprobe_path)

            engine_specific_done = (self.engine.lower() == "ffmpeg" and "faststart 적용됨" in current_status) or \
                                   (self.engine.lower() == "mp4box" and "hint track 존재" in current_status)

            if engine_specific_done:
                self.signals.result.emit(filepath, "이미 처리됨 ⏭️")
                self.signals.finished.emit(filepath, True)
                return

            self.signals.result.emit(filepath, "처리 중...")

            backup_path = None
            if self.config.BACKUP_ENABLED:
                backup_path = MP4Utils.create_backup(filepath, self.config.BACKUP_DIR)
                if not backup_path:
                    self.signals.result.emit(filepath, "백업 실패 ❌")
                    self.signals.finished.emit(filepath, False)
                    return

            temp_filename = f"{uuid.uuid4().hex}.mp4"
            temp_path = os.path.join(self.config.TEMP_DIR, temp_filename)

            success = self._process_file(filepath, temp_path)

            if success and os.path.exists(temp_path):
                shutil.move(temp_path, filepath)
                self.signals.result.emit(filepath, "완료 ✅")
                self.signals.finished.emit(filepath, True)
            else:
                self.signals.result.emit(filepath, "처리 실패 ❌")
                self.signals.finished.emit(filepath, False)
                if backup_path and os.path.exists(backup_path):
                    try:
                        shutil.copy2(backup_path, filepath)
                        self.signals.log.emit(f"백업에서 복원: {filepath}")
                    except Exception as e:
                        self.signals.log.emit(f"백업 복원 실패: {e}")

            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    self.signals.log.emit(f"임시 파일 삭제 실패: {e}")

        except Exception as e:
            logger.error(f"Parallel processing error for {filepath}: {e}")
            self.signals.error.emit(filepath, str(e))
            self.signals.finished.emit(filepath, False)

    def _process_file(self, input_path: str, output_path: str) -> bool:
        try:
            if self.engine.lower() == "ffmpeg":
                cmd = [self.ffmpeg_path, '-y', '-i', input_path]

                if self.hw_accel_option == 'nvidia':
                    cmd.extend(['-c:v', 'h264_nvenc', '-c:a', 'copy'])
                elif self.hw_accel_option == 'amd':
                    cmd.extend(['-c:v', 'h264_amf', '-c:a', 'copy'])
                elif self.hw_accel_option == 'intel':
                    cmd.extend(['-hwaccel', 'qsv', '-c:v', 'h264_qsv', '-c:a', 'copy'])
                else:
                    cmd.extend(['-c', 'copy'])

                cmd.extend(['-movflags', '+faststart', output_path])
            else:
                cmd = [
                    self.mp4box_path, '-hint', input_path,
                    '-out', output_path
                ]

            self.signals.log.emit(f"실행 명령: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True, text=True,
                encoding='utf-8', errors='ignore',
                timeout=self.config.TIMEOUT, startupinfo=startupinfo
            )

            if result.returncode != 0:
                self.signals.log.emit(f"처리 실패: {result.stderr}")
                return False

            return True

        except subprocess.TimeoutExpired:
            self.signals.log.emit(f"처리 시간 초과: {input_path}")
            return False
        except FileNotFoundError as e:
            self.signals.log.emit(f"실행 파일을 찾을 수 없음: {e}")
            return False
        except Exception as e:
            self.signals.log.emit(f"처리 중 오류: {e}")
            return False

# Main GUI
class MP4ProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(self.config.MAX_PARALLEL)
        self.ffmpeg_path = shutil.which("ffmpeg") or "ffmpeg.exe"
        self.ffprobe_path = shutil.which("ffprobe") or "ffprobe.exe"
        self.mp4box_path = shutil.which("MP4Box") or "MP4Box.exe"

        # For serial worker
        self.processing_worker = None

        # For parallel tracking
        self._parallel_total = 0
        self._parallel_done = 0
        self._parallel_success = 0
        self._parallel_errors = 0

        self.hw_accel_type = 'none'
        self.init_ui()
        self.check_dependencies()
        self.detect_hw_accel()

    def detect_hw_accel(self):
        self.hw_accel_type = 'none'
        self.hw_accel_combo.setEnabled(False)
        vendor = "none"
        codecs = []
        cpu_cores = psutil.cpu_count(logical=True) or 1

        if platform.system() == "Windows":
            try:
                import winreg
                found_gpu = False
                # 0000부터 0003까지 순회 (최대 4개 GPU 슬롯 확인)
                for i in range(4):
                    try:
                        key_path = rf"SYSTEM\CurrentControlSet\Control\Class\{{4d36e968-e325-11ce-bfc1-08002be10318}}\{i:04d}"
                        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
                        gpu_desc = winreg.QueryValueEx(key, "DriverDesc")[0].upper()
                        winreg.CloseKey(key)
                        
                        if "NVIDIA" in gpu_desc or "INTEL" in gpu_desc or "AMD" in gpu_desc:
                            if "NVIDIA" in gpu_desc:
                                vendor = "nvidia"
                            elif "INTEL" in gpu_desc:
                                vendor = "intel"
                            elif "AMD" in gpu_desc:
                                vendor = "amd"
                            self.log(f"GPU 감지: {gpu_desc}")
                            found_gpu = True
                            break
                    except FileNotFoundError:
                        continue
                    except Exception:
                        continue
                
                if not found_gpu:
                    self.log("레지스트리에서 GPU를 찾을 수 없습니다")
            except Exception as e:
                self.log(f"레지스트리 GPU 감지 실패: {e}")
				
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-hide_banner", "-encoders"],
                capture_output=True, text=True, encoding="utf-8", errors="ignore",
                timeout=10, startupinfo=startupinfo
            )
            if result.returncode == 0:
                output = result.stdout.lower()
                if "h264_nvenc" in output:
                    codecs.append("nvidia")
                if "h264_qsv" in output:
                    codecs.append("intel")
                if "h264_amf" in output:
                    codecs.append("amd")
        except Exception as e:
            self.log(f"FFmpeg 인코더 감지 실패: {e}")

        if vendor == "nvidia" and "nvidia" in codecs:
            self.hw_accel_type = "nvidia"
        elif vendor == "intel" and "intel" in codecs:
            self.hw_accel_type = "intel"
        elif vendor == "amd" and "amd" in codecs:
            self.hw_accel_type = "amd"
        else:
            self.hw_accel_type = "none"

        if self.hw_accel_type != "none":
            self.hw_accel_combo.setEnabled(True)
            self.log(f"{vendor.upper()} GPU 감지됨 → 하드웨어 가속({self.hw_accel_type}) 사용 가능")
        else:
            self.log(f"지원 GPU 없음 또는 FFmpeg 미지원. CPU 모드 ({cpu_cores} 쓰레드)")

        self.log(f"CPU 코어 수: {cpu_cores}")

    def init_ui(self):
        self.setWindowTitle("MP4 HintBox Pro - by suldo.com")
        self.setGeometry(100, 100, 800, 400)
        self.setAcceptDrops(True)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)

        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        button_layout1 = QHBoxLayout()
        self.add_btn = QPushButton("파일 추가")
        self.add_btn.clicked.connect(self.add_files)

        self.process_selected_btn = QPushButton("선택 파일 처리")
        self.process_selected_btn.clicked.connect(self.process_selected)

        self.process_all_btn = QPushButton("전체 파일 처리")
        self.process_all_btn.clicked.connect(self.process_all)

        self.cancel_btn = QPushButton("취소")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_processing)

        button_layout1.addWidget(self.add_btn)
        button_layout1.addWidget(self.process_selected_btn)
        button_layout1.addWidget(self.process_all_btn)
        button_layout1.addWidget(self.cancel_btn)

        button_layout2 = QHBoxLayout()
        button_layout2.addWidget(QLabel("처리 엔진:"))

        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["FFmpeg (FastStart)", "MP4Box (Hint Track)"])
        self.engine_combo.currentTextChanged.connect(self.on_engine_changed)
        button_layout2.addWidget(self.engine_combo)

        self.hw_accel_combo = QComboBox()
        self.hw_accel_combo.addItems(["CPU (Copy)", "GPU (Transcode)"])
        self.hw_accel_combo.setEnabled(False)
        button_layout2.addWidget(self.hw_accel_combo)

        self.backup_checkbox = QCheckBox("백업 생성")
        self.backup_checkbox.setChecked(self.config.BACKUP_ENABLED)
        self.backup_checkbox.toggled.connect(self.on_backup_toggled)
        button_layout2.addWidget(self.backup_checkbox)

        # New: processing mode selector
        button_layout2.addWidget(QLabel("처리 방식:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["직렬 처리 (Serial)", "병렬 처리 (Parallel)"])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        button_layout2.addWidget(self.mode_combo)

        self.del_selected_btn = QPushButton("선택 삭제")
        self.del_selected_btn.clicked.connect(self.delete_selected)

        self.del_all_btn = QPushButton("전체 삭제")
        self.del_all_btn.clicked.connect(self.delete_all)

        button_layout2.addWidget(self.del_selected_btn)
        button_layout2.addWidget(self.del_all_btn)
        button_layout2.addStretch()

        control_layout.addLayout(button_layout1)
        control_layout.addLayout(button_layout2)

        file_label = QLabel("파일 목록 (MP4 파일을 드래그 앤 드롭으로 추가할 수 있습니다):")
        self.file_list = DragDropListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_list.itemDoubleClicked.connect(self.retry_file)
        self.file_list.files_dropped.connect(self.on_files_dropped)

        control_layout.addWidget(file_label)
        control_layout.addWidget(self.file_list)

        self.progress_label = QLabel("준비됨")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_label)
        control_layout.addWidget(self.progress_bar)

        splitter.addWidget(control_widget)

        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.addWidget(QLabel("처리 로그:"))

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        font = QFont("Consolas", 9)
        self.log_text.setFont(font)
        log_layout.addWidget(self.log_text)

        splitter.addWidget(log_widget)
        splitter.setSizes([500, 200])

# 하단 정보 영역을 위한 수평 레이아웃 생성
        bottom_layout = QHBoxLayout()

        # 왼쪽: Official Website 링크
        website_label = QLabel('<a href="https://hint.ev7.net">Official Website</a>')
        website_label.setOpenExternalLinks(True)
        website_label.setAlignment(Qt.AlignLeft)
        bottom_layout.addWidget(website_label)

        # 오른쪽: 기존 버전 정보
        info_label = QLabel(f"{QApplication.applicationName()} v{QApplication.applicationVersion()} © {QApplication.organizationName()}")
        info_label.setAlignment(Qt.AlignRight)
        bottom_layout.addWidget(info_label)

        main_layout.addLayout(bottom_layout)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            for url in urls:
                if url.isLocalFile() and url.toLocalFile().lower().endswith('.mp4'):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            file_paths = []
            for url in urls:
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    if file_path.lower().endswith('.mp4'):
                        file_paths.append(file_path)
            if file_paths:
                self.on_files_dropped(file_paths)
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()

    def on_files_dropped(self, file_paths: List[str]):
        added_count = 0
        duplicate_count = 0
        for filepath in file_paths:
            is_duplicate = any(self.file_list.item(i).text().split(" | ")[0] == filepath for i in range(self.file_list.count()))
            if not is_duplicate:
                self.add_file(filepath)
                added_count += 1
            else:
                duplicate_count += 1
        if added_count > 0:
            self.log(f"드래그 앤 드롭으로 {added_count}개 파일 추가")
        if duplicate_count > 0:
            self.log(f"중복된 파일 {duplicate_count}개는 건너뜀")
        if added_count == 0 and duplicate_count == 0:
            self.log("유효한 MP4 파일을 찾을 수 없습니다")

    def check_dependencies(self):
        missing = []
        if not shutil.which(self.ffmpeg_path.split()[0]):
            missing.append("FFmpeg")
        if not shutil.which(self.mp4box_path.split()[0]):
            missing.append("MP4Box")
        if missing:
            QMessageBox.warning(
                self, "의존성 프로그램 누락",
                f"다음 프로그램이 설치되어 있지 않습니다:\n{', '.join(missing)}\n\n"
                f"해당 엔진을 사용할 수 없습니다."
            )

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "MP4 파일 선택", "",
            "MP4 Files (*.mp4);;All Files (*)"
        )
        if files:
            self.on_files_dropped(files)

    def add_file(self, filepath: str):
        self.file_list.addItem(f"{filepath} | 확인 중...")
        self.log(f"파일 추가: {os.path.basename(filepath)}")
        engine = self.get_current_engine()
        worker = CheckWorker(filepath, engine, self.mp4box_path, self.ffprobe_path)
        worker.signals.result.connect(self.update_file_status)
        worker.signals.error.connect(self.handle_check_error)
        self.threadpool.start(worker)

    def update_file_status(self, filepath: str, status: str):
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.text().split(" | ")[0] == filepath:
                item.setText(f"{filepath} | {status}")
                break

    def handle_check_error(self, filepath: str, error: str):
        self.update_file_status(filepath, "체크 실패")
        self.log(f"상태 확인 실패 - {os.path.basename(filepath)}: {error}")

    def get_current_engine(self) -> str:
        text = self.engine_combo.currentText()
        return "ffmpeg" if "FFmpeg" in text else "mp4box"

    def get_current_hw_accel_option(self) -> str:
        text = self.hw_accel_combo.currentText()
        return self.hw_accel_type if "GPU" in text else "none"

    def on_engine_changed(self):
        is_ffmpeg = self.get_current_engine() == "ffmpeg"
        self.hw_accel_combo.setEnabled(is_ffmpeg and self.hw_accel_type != 'none')
        if self.file_list.count() == 0:
            return
        engine = self.get_current_engine()
        self.log(f"엔진 변경: {self.engine_combo.currentText()}")
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            filepath = item.text().split(" | ")[0]
            item.setText(f"{filepath} | 재확인 중...")
            worker = CheckWorker(filepath, engine, self.mp4box_path, self.ffprobe_path)
            worker.signals.result.connect(self.update_file_status)
            worker.signals.error.connect(self.handle_check_error)
            self.threadpool.start(worker)

    def on_backup_toggled(self, checked: bool):
        self.config.BACKUP_ENABLED = checked
        self.log(f"백업 {'활성화' if checked else '비활성화'}")

    def on_mode_changed(self, index: int):
        mode = "serial" if index == 0 else "parallel"
        self.config.PROCESS_MODE = mode
        self.log(f"처리 방식 변경: {'직렬' if mode=='serial' else '병렬'}")

    def delete_selected(self):
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            return
        for item in selected_items:
            row = self.file_list.row(item)
            self.file_list.takeItem(row)
        self.log(f"{len(selected_items)}개 파일 제거")

    def delete_all(self):
        count = self.file_list.count()
        if count == 0:
            return
        reply = QMessageBox.question(
            self, "확인", f"모든 파일({count}개)을 목록에서 제거하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.file_list.clear()
            self.log(f"{count}개 파일 전체 제거")

    def process_selected(self):
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "알림", "처리할 파일을 선택해주세요.")
            return
        filepaths = [item.text().split(" | ")[0] for item in selected_items]
        self.start_processing(filepaths)

    def process_all(self):
        if self.file_list.count() == 0:
            QMessageBox.information(self, "알림", "처리할 파일이 없습니다.")
            return
        filepaths = [self.file_list.item(i).text().split(" | ")[0] for i in range(self.file_list.count())]
        self.start_processing(filepaths)

    def start_processing(self, filepaths: List[str]):
        if not filepaths:
            return
        engine = self.get_current_engine()
        hw_accel_option = self.get_current_hw_accel_option()
        if engine == "ffmpeg" and not shutil.which(self.ffmpeg_path.split()[0]):
            QMessageBox.warning(self, "오류", "FFmpeg를 찾을 수 없습니다.")
            return
        elif engine == "mp4box" and not shutil.which(self.mp4box_path.split()[0]):
            QMessageBox.warning(self, "오류", "MP4Box를 찾을 수 없습니다.")
            return

        files_to_process = []
        already_processed = []
        for filepath in filepaths:
            try:
                status = MP4Utils.get_processing_status(filepath, engine, self.mp4box_path, self.ffprobe_path)
                engine_specific_done = (engine.lower() == "ffmpeg" and "faststart 적용됨" in status) or \
                                       (engine.lower() == "mp4box" and "hint track 존재" in status)
                if engine_specific_done:
                    already_processed.append(os.path.basename(filepath))
                    for i in range(self.file_list.count()):
                        item = self.file_list.item(i)
                        if item.text().split(" | ")[0] == filepath:
                            item.setText(f"{filepath} | 이미 처리됨 ⏭️")
                            break
                else:
                    files_to_process.append(filepath)
            except Exception as e:
                self.log(f"상태 확인 실패 - {os.path.basename(filepath)}: {e}")
                files_to_process.append(filepath)

        if already_processed:
            if len(already_processed) == len(filepaths):
                QMessageBox.information(self, "알림", f"선택된 모든 파일이 이미 처리되어 있습니다.\n처리된 파일: {len(already_processed)}개")
                return
            else:
                reply = QMessageBox.question(
                    self, "확인",
                    f"이미 처리된 파일 {len(already_processed)}개를 발견했습니다.\n"
                    f"처리되지 않은 {len(files_to_process)}개 파일만 처리하시겠습니까?\n\n"
                    f"이미 처리된 파일:\n" + "\n".join(already_processed[:5]) +
                    (f"\n... 외 {len(already_processed)-5}개" if len(already_processed) > 5 else ""),
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return

        if not files_to_process:
            return

        # UI lock and progress setup
        self.toggle_ui(False)
        self.progress_bar.setVisible(True)
        total_count = len(filepaths)
        self.progress_bar.setMaximum(total_count)
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"처리 중 (0/{total_count})")

        if self.config.PROCESS_MODE == "serial":
            # Use original serial worker
            self.processing_worker = ProcessingWorker(
                filepaths, engine, self.config,
                self.ffmpeg_path, self.mp4box_path, self.ffprobe_path,
                hw_accel_option=hw_accel_option
            )
            self.processing_worker.progress_signal.connect(self.update_progress)
            self.processing_worker.finished_signal.connect(self.finish_processing)
            self.processing_worker.log_signal.connect(self.log)
            self.processing_worker.start()
            self.log(f"직렬 처리 시작: 전체 {len(filepaths)}개 파일 (처리 대상: {len(files_to_process)}개), 엔진: {self.engine_combo.currentText()}, 가속: {self.hw_accel_combo.currentText()}")
        else:
            # Parallel mode using QThreadPool and ProcessWorker per file
            self._parallel_total = len(files_to_process)
            self._parallel_done = 0
            self._parallel_success = 0
            self._parallel_errors = 0

            # For progress display we count completions (including skipped)
            for filepath in files_to_process:
                pw = ProcessWorker(filepath, engine, self.config,
                                   self.ffmpeg_path, self.mp4box_path, self.ffprobe_path,
                                   hw_accel_option=hw_accel_option)
                pw.signals.result.connect(self.update_file_status)
                pw.signals.log.connect(self.log)
                pw.signals.error.connect(self.handle_parallel_error)
                pw.signals.finished.connect(self.handle_parallel_finished)
                self.threadpool.start(pw)

            self.log(f"병렬 처리 시작: 전체 {len(filepaths)}개 파일 (병렬 대상: {len(files_to_process)}개), 최대 동시 작업: {self.threadpool.maxThreadCount()}, 엔진: {self.engine_combo.currentText()}, 가속: {self.hw_accel_combo.currentText()}")

    def handle_parallel_error(self, filepath: str, error: str):
        self.log(f"병렬 처리 오류 - {os.path.basename(filepath)}: {error}")

    def handle_parallel_finished(self, filepath: str, success: bool):
        # Update counters and progress bar
        self._parallel_done += 1
        if success:
            self._parallel_success += 1
        else:
            self._parallel_errors += 1

        # Update progress UI (value is count of completed files among total requested)
        self.progress_bar.setValue(self._parallel_done)
        self.progress_label.setText(f"처리 중 ({self._parallel_done}/{self.progress_bar.maximum()})")

        # Log short summary per completion is handled in update_file_status already
        if self._parallel_done >= self.progress_bar.maximum():
            # All done; finalize
            self.toggle_ui(True)
            self.progress_bar.setValue(self.progress_bar.maximum())
            self.progress_bar.setVisible(False)
            if self._parallel_errors == 0:
                QMessageBox.information(self, "완료", f"병렬 처리 완료: 성공 {self._parallel_success}개, 실패 {self._parallel_errors}개")
            else:
                QMessageBox.warning(self, "완료", f"병렬 처리 완료: 성공 {self._parallel_success}개, 실패 {self._parallel_errors}개")
            self.log(f"병렬 처리 완료: 성공 {self._parallel_success}개, 실패 {self._parallel_errors}개")
            # reset parallel counters
            self._parallel_total = 0
            self._parallel_done = 0
            self._parallel_success = 0
            self._parallel_errors = 0

    def cancel_processing(self):
        # For serial processing
        if self.processing_worker and isinstance(self.processing_worker, ProcessingWorker):
            self.processing_worker.stop()
            self.log("직렬 처리 취소 요청됨")
        # For parallel processing: we can't easily stop already-started subprocesses here,
        # but we can clear the threadpool queue (PyQt5 doesn't provide direct cancellation for running QRunnable).
        # For best-effort, set a flag on queued workers is needed; but we don't maintain references to all.
        # Inform user:
        self.log("병렬 처리 도중에는 즉시 취소가 제한적입니다 (진행 중인 작업은 완료 또는 타임아웃 후 중지됩니다).")

    def retry_file(self, item):
        filepath = item.text().split(" | ")[0]
        status = item.text().split(" | ")[1]
        engine = self.get_current_engine()
        engine_specific_done = (engine.lower() == "ffmpeg" and "faststart 적용됨" in status) or \
                               (engine.lower() == "mp4box" and "hint track 존재" in status) or \
                               "이미 처리됨" in status
        if engine_specific_done:
            filename = os.path.basename(filepath)
            engine_name = "FFmpeg (FastStart)" if engine == "ffmpeg" else "MP4Box (Hint Track)"
            QMessageBox.information(
                self, "알림",
                f"'{filename}' 파일은 현재 엔진({engine_name})으로 이미 처리되어 있습니다.\n"
                f"현재 상태: {status}\n\n"
                f"다른 종류의 처리가 필요한 경우 엔진을 변경하세요:\n"
                f"• FastStart → MP4Box 엔진으로 변경하여 Hint Track 추가\n"
                f"• Hint Track → FFmpeg 엔진으로 변경하여 FastStart 적용"
            )
            return
        if "실패" in status or "체크 실패" in status or "취소됨" in status or "필요" in status:
            filename = os.path.basename(filepath)
            reply = QMessageBox.question(
                self, "재처리",
                f"'{filename}' 파일을 다시 처리하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.start_processing([filepath])

    def toggle_ui(self, enabled: bool):
        self.add_btn.setEnabled(enabled)
        self.process_selected_btn.setEnabled(enabled)
        self.process_all_btn.setEnabled(enabled)
        self.cancel_btn.setEnabled(not enabled)
        self.engine_combo.setEnabled(enabled)
        self.hw_accel_combo.setEnabled(enabled and self.hw_accel_type != 'none' and self.get_current_engine() == 'ffmpeg')
        self.backup_checkbox.setEnabled(enabled)
        self.del_selected_btn.setEnabled(enabled)
        self.del_all_btn.setEnabled(enabled)
        self.mode_combo.setEnabled(enabled)

    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        self.log_text.append(log_message)
        logger.info(message)

    def update_progress(self, current: int, total: int, filepath: str, status: str):
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.text().split(" | ")[0] == filepath:
                item.setText(f"{filepath} | {status}")
                break

        if current == total and "완료" not in status:
            self.progress_bar.setValue(99)
        else:
            self.progress_bar.setValue(current)

        filename = os.path.basename(filepath)
        self.progress_label.setText(f"처리 중 ({current}/{total}): {filename}")
        if "✅" in status or "❌" in status or "⏭️" in status:
            self.log(f"[{current}/{total}] {filename}: {status}")

    def finish_processing(self, success: bool, message: str):
        self.toggle_ui(True)
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.progress_bar.setVisible(False)
        self.progress_label.setText("처리 완료")
        if success:
            QMessageBox.information(self, "완료", message)
        else:
            QMessageBox.warning(self, "완료", message)
        self.log(f"처리 완료: {message}")
        self.processing_worker = None

    def update_file_status(self, filepath: str, status: str):
        # update list item text
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.text().split(" | ")[0] == filepath:
                item.setText(f"{filepath} | {status}")
                break
        # Also log succinctly when finished/failed
        if "완료" in status or "처리 실패" in status or "이미 처리됨" in status:
            filename = os.path.basename(filepath)
            self.log(f"{filename}: {status}")

    def closeEvent(self, event):
        if self.processing_worker and isinstance(self.processing_worker, ProcessingWorker) and self.processing_worker.isRunning():
            reply = QMessageBox.question(
                self, "종료", "처리가 진행 중입니다. 종료하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.processing_worker.stop()
                self.processing_worker.wait(3000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setApplicationName("MP4 HintBox Pro")
    app.setApplicationVersion("1.0.4")
    app.setOrganizationName("suldo.com")

    try:
        from PyQt5.QtGui import QIcon
        app.setWindowIcon(QIcon("app_icon.ico"))
    except Exception as e:
        print(f"아이콘 설정 실패: {e}")

    window = MP4ProcessorApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
