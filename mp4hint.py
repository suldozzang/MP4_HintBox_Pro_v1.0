# -*- coding: utf-8 -*-
import sys
import os
import shutil
import subprocess
import json
import logging
import uuid
import struct
from datetime import datetime
from typing import List, Optional, Tuple
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QListWidget, QPushButton, QLabel, QProgressBar,
    QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QMessageBox,
    QTextEdit, QSplitter, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QThreadPool, QRunnable, QObject, QUrl
from PyQt5.QtGui import QFont, QPainter, QPen, QColor

# 파일체크
def check_external_tools(self):
    missing_tools = []
    if not shutil.which(self.ffmpegpath):
        missing_tools.append("FFmpeg")
    if not shutil.which(self.mp4boxpath):
        missing_tools.append("MP4Box")
    if missing_tools:
        msg = f"필수 외부 프로그램이 설치 또는 경로에 없습니다: {', '.join(missing_tools)}\n"
        msg += "아래 링크에서 설치 파일 및 안내를 확인하세요:\nhttps://ev7.net/hint/info.php"
        QMessageBox.warning(self, "외부 프로그램 누락", msg)
        return False
    return True
#콘솔x
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
    TIMEOUT = 120
    
    def __init__(self):
        os.makedirs(self.TEMP_DIR, exist_ok=True)
        if self.BACKUP_ENABLED:
            os.makedirs(self.BACKUP_DIR, exist_ok=True)

# 드래그 앤 드롭을 지원하는 QListWidget 클래스
class DragDropListWidget(QListWidget):
    files_dropped = pyqtSignal(list)  # 드롭된 파일 목록을 전달하는 시그널
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DropOnly)
        self.drag_active = False
        
    def dragEnterEvent(self, event):
        """드래그가 시작될 때"""
        if event.mimeData().hasUrls():
            # URL들 중에서 MP4 파일이 있는지 확인
            urls = event.mimeData().urls()
            valid_files = []
            for url in urls:
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    if file_path.lower().endswith('.mp4'):
                        valid_files.append(file_path)
            
            if valid_files:
                event.acceptProposedAction()
                self.drag_active = True
                self.update()  # 화면 다시 그리기
            else:
                event.ignore()
        else:
            event.ignore()
    
    def dragMoveEvent(self, event):
        """드래그가 이동할 때"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def dragLeaveEvent(self, event):
        """드래그가 영역을 벗어날 때"""
        self.drag_active = False
        self.update()
        event.accept()
    
    def dropEvent(self, event):
        """파일이 드롭될 때"""
        self.drag_active = False
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            file_paths = []
            
            for url in urls:
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    if file_path.lower().endswith('.mp4'):
                        file_paths.append(file_path)
            
            if file_paths:
                self.files_dropped.emit(file_paths)
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()
        
        self.update()
    
    def paintEvent(self, event):
        """리스트 위젯 그리기 (드래그 상태 표시 포함)"""
        super().paintEvent(event)
        
        # 드래그 중일 때 점선 테두리 표시
        if self.drag_active:
            painter = QPainter(self.viewport())
            painter.setRenderHint(QPainter.Antialiasing)
            
            # 점선 스타일 설정
            pen = QPen(QColor(0, 123, 255, 180), 3, Qt.DashLine)
            painter.setPen(pen)
            
            # 테두리 그리기
            rect = self.viewport().rect()
            rect.adjust(5, 5, -5, -5)
            painter.drawRect(rect)
            
            # 드롭 안내 텍스트 표시 (리스트가 비어있을 때만)
            if self.count() == 0:
                painter.setPen(QColor(108, 117, 125))
                painter.drawText(rect, Qt.AlignCenter, 
                    "MP4 파일을 여기에 드래그하세요\n또는 '파일 추가' 버튼을 사용하세요")
            
            painter.end()

# MP4 분석 유틸리티
class MP4Utils:
    @staticmethod
    def check_hint_track_with_mp4box(filepath: str, mp4box_path: str) -> bool:
        """MP4Box를 사용하여 hint track 존재 여부 확인"""
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
        """FFprobe로 faststart 여부 확인 (더 정확한 방법)"""
        try:
            # 방법 1: ffprobe로 format 정보 확인
            result = subprocess.run([
                ffprobe_path, '-v', 'quiet', '-print_format', 'json',
                '-show_format', filepath
            ], capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=30, startupinfo=startupinfo)
            
            if result.returncode != 0:
                logger.warning(f"FFprobe format check failed for {filepath}")
                return MP4Utils._check_faststart_binary(filepath)
            
            data = json.loads(result.stdout)
            format_info = data.get('format', {})
            
            # start_time이 0에 가깝고, duration이 있으면 일반적으로 faststart
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
        """바이너리 분석으로 faststart 여부 확인 (가장 정확한 방법)"""
        try:
            with open(filepath, 'rb') as f:
                # MP4 파일의 첫 8KB 정도를 읽어서 atom 구조 확인
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
                    
                # atom 크기와 타입 읽기
                try:
                    atom_size = int.from_bytes(data[pos:pos+4], 'big')
                    atom_type = data[pos+4:pos+8].decode('ascii', errors='ignore')
                    
                    if atom_type == 'ftyp':
                        found_ftyp = True
                    elif atom_type == 'moov':
                        found_moov = True
                        # moov가 mdat보다 먼저 나오면 faststart
                        if not found_mdat:
                            logger.info(f"Faststart detected for {filepath} (moov before mdat)")
                            return True
                    elif atom_type == 'mdat':
                        found_mdat = True
                        # mdat가 moov보다 먼저 나오면 non-faststart
                        if not found_moov:
                            logger.info(f"Non-faststart detected for {filepath} (mdat before moov)")
                            return False
                    
                    # atom 크기가 유효하지 않으면 중단
                    if atom_size < 8 or atom_size > len(data):
                        break
                        
                    pos += atom_size
                    
                except (ValueError, UnicodeDecodeError, struct.error):
                    pos += 1
                    continue
            
            # 8KB 내에서 moov를 찾지 못했다면 더 많이 읽어서 확인
            if not found_moov and not found_mdat:
                return MP4Utils._deep_faststart_check(filepath)
            
            return False
            
        except Exception as e:
            logger.error(f"Binary faststart check error for {filepath}: {e}")
            return False

    @staticmethod
    def _deep_faststart_check(filepath: str) -> bool:
        """더 깊은 바이너리 분석 (큰 파일용)"""
        try:
            with open(filepath, 'rb') as f:
                # 처음 64KB 읽기
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
                # moov는 찾았지만 mdat을 64KB 내에서 못찾음 -> 보통 faststart
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
        """파일의 종합적인 처리 상태 확인"""
        status = {
            'faststart': False,
            'hint_track': False,
            'faststart_status': 'faststart 필요',
            'hint_status': 'hint track 필요'
        }
        
        try:
            # FastStart 확인
            if ffprobe_path:
                status['faststart'] = MP4Utils.check_faststart_with_ffprobe(filepath, ffprobe_path)
            else:
                status['faststart'] = MP4Utils._check_faststart_binary(filepath)
            
            if status['faststart']:
                status['faststart_status'] = 'faststart 적용됨'
                
            # Hint Track 확인  
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
        """엔진에 따른 파일 처리 상태 확인 (종합 상태 포함)"""
        try:
            comprehensive = MP4Utils.get_comprehensive_status(filepath, mp4box_path, ffprobe_path)
            
            if engine.lower() == "ffmpeg":
                # FFmpeg 엔진: faststart 상태 확인하되, hint track 정보도 표시
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
                # MP4Box 엔진: hint track 상태 확인하되, faststart 정보도 표시
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
        """파일 백업 생성"""
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

# 신호를 위한 QObject 클래스
class WorkerSignals(QObject):
    result = pyqtSignal(str, str)  # filepath, status
    error = pyqtSignal(str, str)   # filepath, error_message

# 체크 작업 클래스
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

# 처리 작업 스레드
class ProcessingWorker(QThread):
    progress_signal = pyqtSignal(int, int, str, str)  # current, total, filepath, status
    finished_signal = pyqtSignal(bool, str)  # success, message
    log_signal = pyqtSignal(str)

    def __init__(self, file_list: List[str], engine: str, config: Config, 
                 ffmpeg_path: str, mp4box_path: str, ffprobe_path: str = None):
        super().__init__()
        self.file_list = file_list
        self.engine = engine
        self.config = config
        self.ffmpeg_path = ffmpeg_path
        self.mp4box_path = mp4box_path
        self.ffprobe_path = ffprobe_path
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
                # 이미 처리된 파일 체크 - 현재 엔진에 맞는 처리만 확인
                current_status = MP4Utils.get_processing_status(filepath, self.engine, self.mp4box_path, 
                                                              self.ffprobe_path if hasattr(self, 'ffprobe_path') else None)
                
                # 현재 엔진에 해당하는 처리가 이미 완료된 경우만 스킵
                engine_specific_done = False
                if self.engine.lower() == "ffmpeg" and ("faststart 적용됨" in current_status):
                    engine_specific_done = True
                elif self.engine.lower() == "mp4box" and ("hint track 존재" in current_status):
                    engine_specific_done = True
                    
                if engine_specific_done:
                    self.progress_signal.emit(idx, len(self.file_list), filepath, "이미 처리됨 ⏭️")
                    skipped_count += 1
                    success_count += 1  # 스킵된 파일도 성공으로 간주
                    continue

                # 진행 중 표시
                self.progress_signal.emit(idx, len(self.file_list), filepath, "처리 중...")

                # 백업 생성
                backup_path = None
                if self.config.BACKUP_ENABLED:
                    backup_path = MP4Utils.create_backup(filepath, self.config.BACKUP_DIR)
                    if not backup_path:
                        self.progress_signal.emit(idx, len(self.file_list), filepath, "백업 실패 ❌")
                        error_count += 1
                        continue

                # 임시 파일 경로 생성 (UUID 사용으로 충돌 방지)
                temp_filename = f"{uuid.uuid4().hex}.mp4"
                temp_path = os.path.join(self.config.TEMP_DIR, temp_filename)

                # 처리 명령 실행
                success = self._process_file(filepath, temp_path)
                
                if success and os.path.exists(temp_path):
                    # 원본 파일 교체
                    shutil.move(temp_path, filepath)
                    self.progress_signal.emit(idx, len(self.file_list), filepath, "완료 ✅")
                    success_count += 1
                else:
                    self.progress_signal.emit(idx, len(self.file_list), filepath, "처리 실패 ❌")
                    error_count += 1
                    
                    # 백업에서 복원
                    if backup_path and os.path.exists(backup_path):
                        try:
                            shutil.copy2(backup_path, filepath)
                            self.log_signal.emit(f"백업에서 복원: {filepath}")
                        except Exception as e:
                            self.log_signal.emit(f"백업 복원 실패: {e}")

                # 임시 파일 정리
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception as e:
                        self.log_signal.emit(f"임시 파일 삭제 실패: {e}")

            except Exception as e:
                logger.error(f"Processing error for {filepath}: {e}")
                self.progress_signal.emit(idx, len(self.file_list), filepath, f"오류 ❌ ({type(e).__name__})")
                error_count += 1

        # 완료 메시지
        if self._stop_requested:
            message = "처리가 취소되었습니다."
        else:
            if skipped_count > 0:
                message = f"처리 완료: 성공 {success_count - skipped_count}개, 건너뜀 {skipped_count}개, 실패 {error_count}개"
            else:
                message = f"처리 완료: 성공 {success_count}개, 실패 {error_count}개"
        
        self.finished_signal.emit(error_count == 0, message)

    def _process_file(self, input_path: str, output_path: str) -> bool:
        """파일 처리 실행"""
        try:
            if self.engine.lower() == "ffmpeg":
                cmd = [
                    self.ffmpeg_path, '-y', '-i', input_path, 
                    '-c', 'copy', '-movflags', '+faststart', 
                    output_path
                ]
            else:  # MP4Box
                cmd = [
                    self.mp4box_path, '-hint', input_path, 
                    '-out', output_path
                ]

            self.log_signal.emit(f"실행 명령: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd, 
                capture_output=True, text=True,
                encoding='utf-8', errors='ignore',
                timeout=self.config.TIMEOUT, startupinfo=startupinfo
            )

            if result.returncode != 0:
                self.log_signal.emit(f"처리 실패: {result.stderr}")
                return False

            return True

        except subprocess.TimeoutExpired:
            self.log_signal.emit(f"처리 시간 초과: {input_path}")
            return False
        except FileNotFoundError as e:
            self.log_signal.emit(f"실행 파일을 찾을 수 없음: {e}")
            return False
        except Exception as e:
            self.log_signal.emit(f"처리 중 오류: {e}")
            return False

# 메인 GUI 클래스
class MP4ProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(self.config.MAX_PARALLEL)
        
        # 실행 파일 경로
        self.ffmpeg_path = shutil.which("ffmpeg") or "ffmpeg.exe"
        self.ffprobe_path = shutil.which("ffprobe") or "ffprobe.exe"
        self.mp4box_path = shutil.which("MP4Box") or "MP4Box.exe"
        
        self.processing_worker = None
        self.init_ui()
        self.check_dependencies()

    def init_ui(self):
        self.setWindowTitle("MP4 HintBox Pro v1.0 - by suldo.com")
        self.setGeometry(100, 100, 1000, 600)
        
        # 드래그 앤 드롭을 전체 윈도우에서 허용
        self.setAcceptDrops(True)

        # 중앙 위젯과 스플리터
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)

        # 상단 컨트롤 패널
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        
        # 버튼 행 1
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

        # 버튼 행 2
        button_layout2 = QHBoxLayout()
        button_layout2.addWidget(QLabel("처리 엔진:"))
        
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["FFmpeg (FastStart)", "MP4Box (Hint Track)"])
        self.engine_combo.currentTextChanged.connect(self.on_engine_changed)
        button_layout2.addWidget(self.engine_combo)
        
        self.backup_checkbox = QCheckBox("백업 생성")
        self.backup_checkbox.setChecked(self.config.BACKUP_ENABLED)
        self.backup_checkbox.toggled.connect(self.on_backup_toggled)
        button_layout2.addWidget(self.backup_checkbox)
        
        self.del_selected_btn = QPushButton("선택 삭제")
        self.del_selected_btn.clicked.connect(self.delete_selected)
        
        self.del_all_btn = QPushButton("전체 삭제")
        self.del_all_btn.clicked.connect(self.delete_all)
        
        button_layout2.addWidget(self.del_selected_btn)
        button_layout2.addWidget(self.del_all_btn)
        button_layout2.addStretch()

        control_layout.addLayout(button_layout1)
        control_layout.addLayout(button_layout2)

        # 파일 리스트 (드래그 앤 드롭 지원)
        file_label = QLabel("파일 목록 (MP4 파일을 드래그 앤 드롭으로 추가할 수 있습니다):")
        self.file_list = DragDropListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_list.itemDoubleClicked.connect(self.retry_file)
        
        # 드래그 앤 드롭 시그널 연결
        self.file_list.files_dropped.connect(self.on_files_dropped)
        
        control_layout.addWidget(file_label)
        control_layout.addWidget(self.file_list)

        # 진행률 표시
        self.progress_label = QLabel("준비됨")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_label)
        control_layout.addWidget(self.progress_bar)

        splitter.addWidget(control_widget)

        # 로그 영역
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
        
        # 스플리터 비율 설정
        splitter.setSizes([500, 200])

    def dragEnterEvent(self, event):
        """메인 윈도우 드래그 엔터 이벤트"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            for url in urls:
                if url.isLocalFile() and url.toLocalFile().lower().endswith('.mp4'):
                    event.acceptProposedAction()
                    return
        event.ignore()
    
    def dropEvent(self, event):
        """메인 윈도우 드롭 이벤트"""
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
        """드롭된 파일들을 처리하는 함수"""
        added_count = 0
        duplicate_count = 0
        
        for filepath in file_paths:
            # 중복 확인
            is_duplicate = False
            for i in range(self.file_list.count()):
                if self.file_list.item(i).text().split(" | ")[0] == filepath:
                    is_duplicate = True
                    duplicate_count += 1
                    break
            
            if not is_duplicate:
                self.add_file(filepath)
                added_count += 1
        
        # 결과 로그 출력
        if added_count > 0:
            self.log(f"드래그 앤 드롭으로 {added_count}개 파일 추가")
        
        if duplicate_count > 0:
            self.log(f"중복된 파일 {duplicate_count}개는 건너뜀")
            
        if added_count == 0 and duplicate_count == 0:
            self.log("유효한 MP4 파일을 찾을 수 없습니다")

    def check_dependencies(self):
        """의존성 프로그램 확인"""
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
        """파일 추가"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "MP4 파일 선택", "", 
            "MP4 Files (*.mp4);;All Files (*)"
        )
        
        if files:
            self.on_files_dropped(files)

    def add_file(self, filepath: str):
        """개별 파일 추가 및 상태 확인"""
        # 리스트에 추가
        self.file_list.addItem(f"{filepath} | 확인 중...")
        self.log(f"파일 추가: {os.path.basename(filepath)}")
        
        # 백그라운드에서 상태 확인
        engine = self.get_current_engine()
        worker = CheckWorker(filepath, engine, self.mp4box_path, self.ffprobe_path)
        worker.signals.result.connect(self.update_file_status)
        worker.signals.error.connect(self.handle_check_error)
        self.threadpool.start(worker)

    def update_file_status(self, filepath: str, status: str):
        """파일 상태 업데이트"""
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.text().split(" | ")[0] == filepath:
                item.setText(f"{filepath} | {status}")
                break

    def handle_check_error(self, filepath: str, error: str):
        """상태 확인 오류 처리"""
        self.update_file_status(filepath, "체크 실패")
        self.log(f"상태 확인 실패 - {os.path.basename(filepath)}: {error}")

    def get_current_engine(self) -> str:
        """현재 선택된 엔진 반환"""
        text = self.engine_combo.currentText()
        if "FFmpeg" in text:
            return "ffmpeg"
        else:
            return "mp4box"

    def on_engine_changed(self):
        """엔진 변경 시 모든 파일 상태 재확인"""
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
        """백업 옵션 토글"""
        self.config.BACKUP_ENABLED = checked
        self.log(f"백업 {'활성화' if checked else '비활성화'}")

    def delete_selected(self):
        """선택된 파일 삭제"""
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            return
            
        for item in selected_items:
            row = self.file_list.row(item)
            self.file_list.takeItem(row)
            
        self.log(f"{len(selected_items)}개 파일 제거")

    def delete_all(self):
        """모든 파일 삭제"""
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
        """선택된 파일 처리"""
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "알림", "처리할 파일을 선택해주세요.")
            return
            
        filepaths = [item.text().split(" | ")[0] for item in selected_items]
        self.start_processing(filepaths)

    def process_all(self):
        """모든 파일 처리"""
        if self.file_list.count() == 0:
            QMessageBox.information(self, "알림", "처리할 파일이 없습니다.")
            return
            
        filepaths = []
        for i in range(self.file_list.count()):
            filepath = self.file_list.item(i).text().split(" | ")[0]
            filepaths.append(filepath)
            
        self.start_processing(filepaths)

    def start_processing(self, filepaths: List[str]):
        """처리 시작"""
        if not filepaths:
            return
            
        engine = self.get_current_engine()
        
        # 실행 파일 경로 확인
        if engine == "ffmpeg" and not shutil.which(self.ffmpeg_path.split()[0]):
            QMessageBox.warning(self, "오류", "FFmpeg를 찾을 수 없습니다.")
            return
        elif engine == "mp4box" and not shutil.which(self.mp4box_path.split()[0]):
            QMessageBox.warning(self, "오류", "MP4Box를 찾을 수 없습니다.")
            return

        # 처리가 필요한 파일만 필터링
        files_to_process = []
        already_processed = []
        
        for filepath in filepaths:
            try:
                status = MP4Utils.get_processing_status(filepath, engine, self.mp4box_path, self.ffprobe_path)
                
                # 현재 엔진에 해당하는 처리가 이미 완료되었는지 확인
                engine_specific_done = False
                if engine.lower() == "ffmpeg" and ("faststart 적용됨" in status):
                    engine_specific_done = True
                elif engine.lower() == "mp4box" and ("hint track 존재" in status):
                    engine_specific_done = True
                    
                if engine_specific_done:
                    already_processed.append(os.path.basename(filepath))
                    # UI에서 상태 업데이트
                    for i in range(self.file_list.count()):
                        item = self.file_list.item(i)
                        if item.text().split(" | ")[0] == filepath:
                            item.setText(f"{filepath} | 이미 처리됨 ⏭️")
                            break
                else:
                    files_to_process.append(filepath)
            except Exception as e:
                self.log(f"상태 확인 실패 - {os.path.basename(filepath)}: {e}")
                files_to_process.append(filepath)  # 확인 실패 시에는 처리 시도
        
        # 이미 처리된 파일이 있으면 알림
        if already_processed:
            if len(already_processed) == len(filepaths):
                QMessageBox.information(
                    self, "알림", 
                    f"선택된 모든 파일이 이미 처리되어 있습니다.\n"
                    f"처리된 파일: {len(already_processed)}개"
                )
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
        
        # 처리할 파일이 없으면 종료
        if not files_to_process:
            return

        # UI 상태 변경
        self.toggle_ui(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(filepaths))  # 전체 파일 수로 설정 (스킵 포함)
        self.progress_bar.setValue(0)
        
        # 처리 시작
        self.processing_worker = ProcessingWorker(
            filepaths, engine, self.config,  # 전체 파일 리스트 전달 (스킵 처리용)
            self.ffmpeg_path, self.mp4box_path, self.ffprobe_path
        )
        
        self.processing_worker.progress_signal.connect(self.update_progress)
        self.processing_worker.finished_signal.connect(self.finish_processing)
        self.processing_worker.log_signal.connect(self.log)
        self.processing_worker.start()
        
        self.log(f"처리 시작: 전체 {len(filepaths)}개 파일 (처리 대상: {len(files_to_process)}개, 건너뜀: {len(already_processed)}개), 엔진: {self.engine_combo.currentText()}")

    def cancel_processing(self):
        """처리 취소"""
        if self.processing_worker:
            self.processing_worker.stop()
            self.log("처리 취소 요청됨")

    def update_progress(self, current: int, total: int, filepath: str, status: str):
        """진행률 업데이트"""
        # 파일 상태 업데이트
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.text().split(" | ")[0] == filepath:
                item.setText(f"{filepath} | {status}")
                break
        
        # 진행률 바 업데이트
        self.progress_bar.setValue(current)
        filename = os.path.basename(filepath)
        self.progress_label.setText(f"처리 중 ({current}/{total}): {filename}")
        
        # 로그 출력 (완료/실패/스킵 시에만)
        if "✅" in status or "❌" in status or "⏭️" in status:
            self.log(f"[{current}/{total}] {filename}: {status}")

    def finish_processing(self, success: bool, message: str):
        """처리 완료"""
        self.toggle_ui(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setText("처리 완료")
        
        # 완료 메시지
        if success:
            QMessageBox.information(self, "완료", message)
        else:
            QMessageBox.warning(self, "완료", message)
        
        self.log(f"처리 완료: {message}")
        self.processing_worker = None

    def retry_file(self, item):
        """파일 재처리"""
        filepath = item.text().split(" | ")[0]
        status = item.text().split(" | ")[1]
        engine = self.get_current_engine()
        
        # 현재 엔진에 해당하는 처리가 이미 완료된 파일은 재처리 불가
        engine_specific_done = False
        if engine.lower() == "ffmpeg" and ("faststart 적용됨" in status):
            engine_specific_done = True
        elif engine.lower() == "mp4box" and ("hint track 존재" in status):
            engine_specific_done = True
        elif "이미 처리됨" in status:
            engine_specific_done = True
            
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
        """UI 컨트롤 활성화/비활성화"""
        self.add_btn.setEnabled(enabled)
        self.process_selected_btn.setEnabled(enabled)
        self.process_all_btn.setEnabled(enabled)
        self.cancel_btn.setEnabled(not enabled)
        self.engine_combo.setEnabled(enabled)
        self.backup_checkbox.setEnabled(enabled)
        self.del_selected_btn.setEnabled(enabled)
        self.del_all_btn.setEnabled(enabled)

    def log(self, message: str):
        """로그 메시지 출력"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        self.log_text.append(log_message)
        logger.info(message)

    def closeEvent(self, event):
        """앱 종료 시 정리"""
        if self.processing_worker and self.processing_worker.isRunning():
            reply = QMessageBox.question(
                self, "종료", "처리가 진행 중입니다. 종료하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.processing_worker.stop()
                self.processing_worker.wait(3000)  # 3초 대기
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 모던한 스타일
    
    # 앱 정보 설정
    app.setApplicationName("MP4 Processor Pro")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("suldo.com")
    
    window = MP4ProcessorApp()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()