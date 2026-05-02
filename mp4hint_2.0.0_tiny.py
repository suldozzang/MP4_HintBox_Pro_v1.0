# -*- coding: utf-8 -*-
"""
==========================================================================
 MP4 HintBox Pro Tiny v2.0.0
--------------------------------------------------------------------------
  Author : EV7lab
  Site   : https://www.mp4hintbox.co.kr
  Engine : NativeFastStart 단독 (FFmpeg / MP4Box / psutil 모두 불필요)
==========================================================================
  • 의존성 : PyQt5 만
  • 기능   : MP4 faststart 처리 (qtfaststart 알고리즘 - 순수 Python)
  • 특징   : 단일 파일, 외부 실행파일 동봉 불필요, 가벼움
==========================================================================
"""
from __future__ import annotations

import sys
import os
import re
import json
import time
import uuid
import shutil
import struct
import logging
import threading
import mmap
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any, Callable

from PyQt5.QtCore import (
    Qt, pyqtSignal, QThreadPool, QRunnable, QObject, QMutex,
    QMutexLocker, QSettings, QSize, QRect, pyqtSlot
)
from PyQt5.QtGui import (
    QFont, QPainter, QPen, QColor, QPalette, QIcon, QBrush,
    QKeySequence, QLinearGradient
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QProgressBar,
    QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QMessageBox,
    QTextEdit, QSplitter, QCheckBox, QTreeWidget, QTreeWidgetItem, QHeaderView,
    QLineEdit, QDialog, QDialogButtonBox, QFormLayout, QSpinBox, QGroupBox,
    QStyledItemDelegate, QStyle, QMenu, QAction, QStatusBar,
    QGridLayout, QRadioButton, QButtonGroup, QAbstractItemView, QShortcut,
    QDoubleSpinBox
)


# =====================================================================
APP_NAME       = "MP4 HintBox Pro Tiny"
APP_VERSION    = "2.0.5"
APP_ORG        = "EV7lab"
APP_WEBSITE    = "https://www.mp4hintbox.co.kr"
APP_AUTHOR     = "EV7lab"
APP_BUILD_YEAR = "2026"

COL_NAME, COL_SIZE, COL_DURATION, COL_STATUS, COL_PROGRESS, COL_PATH = range(6)


# =====================================================================
def app_base_dir() -> str:
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def user_data_dir() -> str:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    else:
        base = os.path.expanduser("~/.local/share")
    p = os.path.join(base, "MP4HintBoxProTiny")
    os.makedirs(p, exist_ok=True)
    return p


def setup_logging() -> logging.Logger:
    log_file = os.path.join(user_data_dir(), "mp4hintbox_tiny.log")
    handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    return logging.getLogger("MP4HintBoxTiny")


logger = setup_logging()


# =====================================================================
class OutputMode(Enum):
    OVERWRITE    = "overwrite"
    SEPARATE_DIR = "separate_dir"
    SUFFIX       = "suffix"


class TaskState(Enum):
    PENDING    = "대기 중"
    CHECKING   = "확인 중..."
    QUEUED     = "큐 대기"
    PROCESSING = "처리 중"
    DONE       = "완료 ✓"
    SKIPPED    = "이미 faststart ⏭"
    FAILED     = "실패 ✗"
    CANCELLED  = "취소됨"


@dataclass
class MediaInfo:
    duration: float = 0.0
    size: int = 0
    has_faststart: bool = False


@dataclass
class AppConfig:
    process_mode: str = "serial"
    max_parallel: int = 4
    output_mode: str = OutputMode.OVERWRITE.value
    output_dir: str = ""
    output_suffix: str = "_fs"
    audio_sync_offset_sec: float = 0.00
    show_log: bool = False
    theme: str = "dark"
    recent_dirs: List[str] = field(default_factory=list)


# =====================================================================
class SettingsManager:
    def __init__(self):
        self.q = QSettings(APP_ORG, APP_NAME)
        self.cfg = AppConfig()
        self.load()

    def load(self):
        c = self.cfg
        c.process_mode  = self.q.value("process_mode", c.process_mode, str)
        c.max_parallel  = int(self.q.value("max_parallel", c.max_parallel))
        c.output_mode   = self.q.value("output_mode", c.output_mode, str)
        c.output_dir    = self.q.value("output_dir", c.output_dir, str)
        c.output_suffix = self.q.value("output_suffix", c.output_suffix, str)
        try:
            c.audio_sync_offset_sec = float(self.q.value("audio_sync_offset_sec",
                                                          c.audio_sync_offset_sec))
        except (TypeError, ValueError):
            pass
        c.show_log = self.q.value("show_log", c.show_log, bool)
        c.theme    = self.q.value("theme", c.theme, str)
        recent     = self.q.value("recent_dirs", "", str)
        c.recent_dirs = [p for p in (recent or "").split("|") if p]

    def save(self):
        c = self.cfg
        self.q.setValue("process_mode", c.process_mode)
        self.q.setValue("max_parallel", c.max_parallel)
        self.q.setValue("output_mode", c.output_mode)
        self.q.setValue("output_dir", c.output_dir)
        self.q.setValue("output_suffix", c.output_suffix)
        self.q.setValue("audio_sync_offset_sec", c.audio_sync_offset_sec)
        self.q.setValue("show_log", c.show_log)
        self.q.setValue("theme", c.theme)
        self.q.setValue("recent_dirs", "|".join(c.recent_dirs[:10]))
        self.q.sync()


# =====================================================================
class Theme:
    @staticmethod
    def apply_dark(app: QApplication):
        app.setStyle("Fusion")
        p = QPalette()
        p.setColor(QPalette.Window,           QColor(32, 33, 36))
        p.setColor(QPalette.WindowText,       QColor(232, 234, 237))
        p.setColor(QPalette.Base,             QColor(40, 42, 46))
        p.setColor(QPalette.AlternateBase,    QColor(48, 50, 54))
        p.setColor(QPalette.ToolTipBase,      QColor(45, 47, 51))
        p.setColor(QPalette.ToolTipText,      QColor(232, 234, 237))
        p.setColor(QPalette.Text,             QColor(232, 234, 237))
        p.setColor(QPalette.Button,           QColor(48, 50, 54))
        p.setColor(QPalette.ButtonText,       QColor(232, 234, 237))
        p.setColor(QPalette.BrightText,       QColor(255, 80, 80))
        p.setColor(QPalette.Link,             QColor(102, 178, 255))
        p.setColor(QPalette.Highlight,        QColor(0, 122, 204))
        p.setColor(QPalette.HighlightedText,  QColor(255, 255, 255))
        p.setColor(QPalette.Disabled, QPalette.Text,       QColor(120, 124, 130))
        p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(120, 124, 130))
        app.setPalette(p)
        app.setStyleSheet("""
            QToolTip { color:#e8eaed; background:#2d2f33; border:1px solid #444; }
            QPushButton { padding:6px 14px; border-radius:4px;
                          background:#3a3d42; border:1px solid #4a4d52; }
            QPushButton:hover { background:#4a5057; }
            QPushButton:pressed { background:#2c2e32; }
            QPushButton:disabled { color:#777; background:#2c2e32; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                padding:5px; border:1px solid #4a4d52;
                border-radius:4px; background:#2a2c30; }
            QHeaderView::section { background:#2c2e32; color:#cfd2d6;
                                    padding:6px; border:0; border-right:1px solid #1a1b1d; }
            QTreeWidget { background:#1f2125; alternate-background-color:#26282c;
                          gridline-color:#2c2e32; border:1px solid #2c2e32; }
            QTreeWidget::item { padding:3px; }
            QTreeWidget::item:selected { background:#0a4d8c; }
            QProgressBar { background:#1f2125; border:1px solid #2c2e32;
                            border-radius:3px; text-align:center; color:#e8eaed; }
            QProgressBar::chunk { background:#FB8C00; border-radius:2px; }
            QStatusBar { background:#26282c; }
            QMenu { background:#2a2c30; border:1px solid #444; }
            QMenu::item:selected { background:#0a4d8c; }
        """)

    @staticmethod
    def apply_light(app: QApplication):
        app.setStyle("Fusion")
        app.setPalette(app.style().standardPalette())
        app.setStyleSheet("")


# =====================================================================
#  MP4 atom 파서 + 미디어 정보 (모두 순수 Python)
# =====================================================================
class MP4Reader:
    @staticmethod
    def iter_top_atoms(mm_or_data, end_pos: int):
        size_total = end_pos
        pos = 0
        while pos + 8 <= size_total:
            try:
                size = struct.unpack(">I", mm_or_data[pos:pos+4])[0]
                atype = mm_or_data[pos+4:pos+8]
                if isinstance(atype, (bytes, bytearray)):
                    atype = bytes(atype).decode("ascii", errors="ignore")
            except Exception:
                return
            header_size = 8
            if size == 1:
                if pos + 16 > size_total:
                    return
                size = struct.unpack(">Q", mm_or_data[pos+8:pos+16])[0]
                header_size = 16
            elif size == 0:
                size = size_total - pos
            if size < 8 or pos + size > size_total:
                return
            yield (pos, atype, size, header_size)
            pos += size

    @staticmethod
    def find_box(data, start: int, end: int, target: str,
                 recurse_into: tuple = ()) -> Optional[Tuple[int, int, int]]:
        pos = start
        while pos + 8 <= end:
            try:
                size = struct.unpack(">I", data[pos:pos+4])[0]
                atype = bytes(data[pos+4:pos+8]).decode("ascii", errors="ignore")
            except Exception:
                return None
            header_size = 8
            if size == 1:
                if pos + 16 > end:
                    return None
                size = struct.unpack(">Q", data[pos+8:pos+16])[0]
                header_size = 16
            if size < 8 or pos + size > end:
                return None
            if atype == target:
                return (pos, size, header_size)
            if atype in recurse_into:
                found = MP4Reader.find_box(data, pos + header_size,
                                            pos + size, target, recurse_into)
                if found:
                    return found
            pos += size
        return None

    @staticmethod
    def has_faststart(filepath: str) -> bool:
        try:
            sz = os.path.getsize(filepath)
            if sz < 16:
                return False
            with open(filepath, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    moov_pos = -1
                    mdat_pos = -1
                    for offset, atype, _size, _hs in MP4Reader.iter_top_atoms(mm, sz):
                        if atype == "moov" and moov_pos == -1:
                            moov_pos = offset
                        elif atype == "mdat" and mdat_pos == -1:
                            mdat_pos = offset
                        if moov_pos != -1 and mdat_pos != -1:
                            break
                    if moov_pos == -1:
                        return False
                    if mdat_pos == -1:
                        return True
                    return moov_pos < mdat_pos
        except Exception as e:
            logger.warning(f"faststart 검사 실패 ({filepath}): {e}")
            return False

    @staticmethod
    def get_info(filepath: str) -> MediaInfo:
        info = MediaInfo()
        try:
            info.size = os.path.getsize(filepath)
        except Exception:
            pass
        info.has_faststart = MP4Reader.has_faststart(filepath)
        # mvhd 에서 duration 추출
        try:
            with open(filepath, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    sz = len(mm)
                    moov = None
                    for offset, atype, asize, hs in MP4Reader.iter_top_atoms(mm, sz):
                        if atype == "moov":
                            moov = (offset, asize, hs)
                            break
                    if moov:
                        mvhd = MP4Reader.find_box(mm, moov[0] + moov[2],
                                                   moov[0] + moov[1], "mvhd")
                        if mvhd:
                            ver = mm[mvhd[0] + mvhd[2]]
                            if ver == 1:
                                ts_pos = mvhd[0] + mvhd[2] + 4 + 8 + 8
                                dur_pos = ts_pos + 4
                                if dur_pos + 8 <= mvhd[0] + mvhd[1]:
                                    timescale = struct.unpack(">I", mm[ts_pos:ts_pos+4])[0]
                                    duration  = struct.unpack(">Q", mm[dur_pos:dur_pos+8])[0]
                                    if timescale > 0:
                                        info.duration = duration / timescale
                            else:
                                ts_pos = mvhd[0] + mvhd[2] + 4 + 4 + 4
                                dur_pos = ts_pos + 4
                                if dur_pos + 4 <= mvhd[0] + mvhd[1]:
                                    timescale = struct.unpack(">I", mm[ts_pos:ts_pos+4])[0]
                                    duration  = struct.unpack(">I", mm[dur_pos:dur_pos+4])[0]
                                    if timescale > 0:
                                        info.duration = duration / timescale
        except Exception as e:
            logger.warning(f"mvhd 파싱 실패 ({filepath}): {e}")
        return info


# =====================================================================
#  처리 컨트롤러 (Python 레벨 pause/cancel)
# =====================================================================
class ProcessController(QObject):
    state_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._paused = threading.Event()
        self._paused.set()
        self._cancelled = False

    def pause(self):
        with self._lock:
            self._paused.clear()
        self.state_changed.emit("paused")
        logger.info("일시정지")

    def resume(self):
        with self._lock:
            self._paused.set()
        self.state_changed.emit("running")
        logger.info("재개")

    def cancel(self):
        with self._lock:
            self._cancelled = True
            self._paused.set()
        self.state_changed.emit("cancelled")
        logger.info("취소")

    def reset(self):
        with self._lock:
            self._cancelled = False
            self._paused.set()
        self.state_changed.emit("idle")

    def is_cancelled(self) -> bool:
        return self._cancelled

    def wait_if_paused(self, timeout: float = 0.5) -> None:
        while not self._paused.wait(timeout):
            if self._cancelled:
                return


# =====================================================================
#  NativeFastStart  (qtfaststart + 오디오 싱크 보정)
# =====================================================================
class NativeFastStart:
    CHUNK_SIZE = 8 * 1024 * 1024

    # ── 안전성 검사 ──
    @staticmethod
    def _meta_has_iloc(filepath: str, atoms) -> bool:
        try:
            for offset, atype, size, hs in atoms:
                if atype != "meta":
                    continue
                with open(filepath, "rb") as f:
                    f.seek(offset + hs + 4)
                    end = offset + size
                    pos = offset + hs + 4
                    while pos + 8 <= end:
                        f.seek(pos)
                        h = f.read(8)
                        if len(h) < 8:
                            break
                        bsize = struct.unpack(">I", h[:4])[0]
                        btype = h[4:8].decode("ascii", errors="ignore")
                        if btype == "iloc":
                            return True
                        if bsize < 8:
                            break
                        pos += bsize
            return False
        except Exception:
            return False

    # ── stco/co64 보정 ──
    @staticmethod
    def _adjust_offsets(moov_bytes: bytes, delta: int) -> bytes:
        result = bytearray(moov_bytes)

        def recurse(pos: int, end: int):
            while pos + 8 <= end:
                size = struct.unpack(">I", result[pos:pos+4])[0]
                atype = bytes(result[pos+4:pos+8]).decode("ascii", errors="ignore")
                hs = 8
                if size == 1:
                    if pos + 16 > end:
                        return
                    size = struct.unpack(">Q", result[pos+8:pos+16])[0]
                    hs = 16
                if size < 8 or pos + size > end:
                    return
                if atype in ("stco", "co64"):
                    cpos = pos + hs + 4
                    epos = cpos + 4
                    cnt = struct.unpack(">I", result[cpos:cpos+4])[0]
                    if atype == "stco":
                        for i in range(cnt):
                            ep = epos + i * 4
                            old = struct.unpack(">I", result[ep:ep+4])[0]
                            new = old + delta
                            if new > 0xFFFFFFFF:
                                raise ValueError("stco 32bit 오버플로 (co64 필요)")
                            struct.pack_into(">I", result, ep, new)
                    else:
                        for i in range(cnt):
                            ep = epos + i * 8
                            old = struct.unpack(">Q", result[ep:ep+8])[0]
                            struct.pack_into(">Q", result, ep, old + delta)
                elif atype in ("trak", "mdia", "minf", "stbl", "edts", "udta"):
                    recurse(pos + hs, pos + size)
                pos += size

        recurse(8, len(result))
        return bytes(result)

    # ── 오디오 싱크 오프셋 (audio elst.media_time in-place) ──
    @staticmethod
    def _find_audio_trak(moov) -> Optional[Tuple[int, int, int]]:
        pos, end = 8, len(moov)
        while pos + 8 <= end:
            try:
                size = struct.unpack(">I", moov[pos:pos+4])[0]
                atype = bytes(moov[pos+4:pos+8]).decode("ascii", errors="ignore")
            except Exception:
                return None
            hs = 8
            if size == 1:
                if pos + 16 > end:
                    return None
                size = struct.unpack(">Q", moov[pos+8:pos+16])[0]
                hs = 16
            if size < 8 or pos + size > end:
                return None
            if atype == "trak":
                mdia = MP4Reader.find_box(moov, pos + hs, pos + size, "mdia")
                if mdia:
                    hdlr = MP4Reader.find_box(moov, mdia[0] + mdia[2],
                                               mdia[0] + mdia[1], "hdlr")
                    if hdlr:
                        ht_pos = hdlr[0] + hdlr[2] + 4 + 4
                        if ht_pos + 4 <= hdlr[0] + hdlr[1]:
                            ht = bytes(moov[ht_pos:ht_pos+4]).decode("ascii", errors="ignore")
                            if ht == "soun":
                                return (pos, size, hs)
            pos += size
        return None

    @staticmethod
    def _audio_timescale(moov, trak_start, trak_size, trak_hs) -> Optional[int]:
        mdia = MP4Reader.find_box(moov, trak_start + trak_hs,
                                   trak_start + trak_size, "mdia")
        if not mdia:
            return None
        mdhd = MP4Reader.find_box(moov, mdia[0] + mdia[2],
                                   mdia[0] + mdia[1], "mdhd")
        if not mdhd:
            return None
        version = moov[mdhd[0] + mdhd[2]]
        if version == 1:
            ts_pos = mdhd[0] + mdhd[2] + 4 + 8 + 8
        else:
            ts_pos = mdhd[0] + mdhd[2] + 4 + 4 + 4
        if ts_pos + 4 > mdhd[0] + mdhd[1]:
            return None
        return struct.unpack(">I", moov[ts_pos:ts_pos+4])[0]

    @staticmethod
    def _modify_audio_elst(moov, trak_start, trak_size, trak_hs, units: int) -> bool:
        edts = MP4Reader.find_box(moov, trak_start + trak_hs,
                                   trak_start + trak_size, "edts")
        if not edts:
            return False
        elst = MP4Reader.find_box(moov, edts[0] + edts[2],
                                   edts[0] + edts[1], "elst")
        if not elst:
            return False
        version = moov[elst[0] + elst[2]]
        cpos = elst[0] + elst[2] + 4
        if cpos + 4 > elst[0] + elst[1]:
            return False
        cnt = struct.unpack(">I", moov[cpos:cpos+4])[0]
        if cnt == 0:
            return False
        epos = cpos + 4
        if version == 0:
            mt_pos = epos + 4
            if mt_pos + 4 > elst[0] + elst[1]:
                return False
            old = struct.unpack(">i", moov[mt_pos:mt_pos+4])[0]
            if old == -1:
                return False
            struct.pack_into(">i", moov, mt_pos, max(0, old + units))
            return True
        elif version == 1:
            mt_pos = epos + 8
            if mt_pos + 8 > elst[0] + elst[1]:
                return False
            old = struct.unpack(">q", moov[mt_pos:mt_pos+8])[0]
            if old == -1:
                return False
            struct.pack_into(">q", moov, mt_pos, max(0, old + units))
            return True
        return False

    @staticmethod
    def _apply_audio_offset(moov_bytes: bytes, offset_sec: float,
                             on_log: Callable[[str], None]) -> bytes:
        if abs(offset_sec) < 0.001:
            return moov_bytes
        result = bytearray(moov_bytes)
        audio = NativeFastStart._find_audio_trak(result)
        if not audio:
            on_log("⚠ audio trak 미발견 - 싱크 보정 스킵")
            return moov_bytes
        ts_start, ts_size, ts_hs = audio
        ts = NativeFastStart._audio_timescale(result, ts_start, ts_size, ts_hs)
        if not ts:
            on_log("⚠ mdhd timescale 획득 실패 - 싱크 보정 스킵")
            return moov_bytes
        units = int(round(offset_sec * ts))
        ok = NativeFastStart._modify_audio_elst(result, ts_start, ts_size, ts_hs, units)
        if not ok:
            on_log("⚠ audio elst 미발견 - 싱크 보정 스킵")
            return moov_bytes
        on_log(f"✓ 오디오 싱크 보정 +{offset_sec:.3f}s ({units} units @ {ts}Hz)")
        return bytes(result)

    # ── 메인 처리 ──
    @staticmethod
    def process(input_path: str, output_path: str, ctrl: ProcessController,
                on_progress: Callable[[float, float, float], None],
                on_log: Callable[[str], None],
                audio_sync_offset_sec: float = 0.0) -> Tuple[bool, str]:
        try:
            file_size = os.path.getsize(input_path)
            if file_size < 32:
                return False, "파일 너무 작음"

            with open(input_path, "rb") as fin:
                atoms = list(MP4Reader.iter_top_atoms(
                    open(input_path, "rb").read(min(file_size, 1024*1024)) if False
                    else NativeFastStart._read_atoms_full(fin), file_size))

            ftyp = next((a for a in atoms if a[1] == "ftyp"), None)
            moov = next((a for a in atoms if a[1] == "moov"), None)
            mdat = next((a for a in atoms if a[1] == "mdat"), None)
            if not (ftyp and moov and mdat):
                return False, "ftyp/moov/mdat 누락 (유효한 MP4 가 아님)"

            # 안전성 검사
            mdat_count = sum(1 for a in atoms if a[1] == "mdat")
            moov_count = sum(1 for a in atoms if a[1] == "moov")
            moof_count = sum(1 for a in atoms if a[1] == "moof")
            if moof_count > 0:
                return False, f"fragmented MP4 (moof × {moof_count}) - 처리 불가"
            if mdat_count > 1:
                return False, f"다중 mdat ({mdat_count}) - 처리 불가"
            if moov_count > 1:
                return False, f"다중 moov ({moov_count}) - 처리 불가"
            if any(a[1] == "meta" for a in atoms):
                if NativeFastStart._meta_has_iloc(input_path, atoms):
                    return False, "최상위 meta/iloc 감지 - 처리 불가"

            # 이미 faststart
            if moov[0] < mdat[0]:
                on_log(f"  ⏭ 이미 faststart - 단순 복사")
                NativeFastStart._copy(input_path, output_path, file_size, ctrl, on_progress)
                return True, "이미 faststart (복사 완료)"

            on_log(f"  moov={moov[2]/1024:.1f}KB @ {moov[0]} → 선두로 이동")

            # moov 읽기
            with open(input_path, "rb") as fin:
                fin.seek(moov[0])
                moov_data = fin.read(moov[2])

            # 오디오 싱크 보정 (in-place, 박스 크기 동일)
            if abs(audio_sync_offset_sec) > 0.001:
                moov_data = NativeFastStart._apply_audio_offset(
                    moov_data, audio_sync_offset_sec, on_log)

            # stco/co64 보정
            try:
                moov_new = NativeFastStart._adjust_offsets(moov_data, moov[2])
            except ValueError as e:
                return False, f"native 처리 불가 ({e})"

            # 출력
            start = time.monotonic()
            written = 0

            with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
                fin.seek(ftyp[0])
                fout.write(fin.read(ftyp[2]))
                written += ftyp[2]

                fout.write(moov_new)
                written += len(moov_new)

                for atom in atoms:
                    if atom == ftyp or atom == moov:
                        continue
                    offset, atype, size, _hs = atom
                    fin.seek(offset)
                    remaining = size
                    while remaining > 0:
                        if ctrl.is_cancelled():
                            return False, "취소됨"
                        ctrl.wait_if_paused(0.2)
                        n = min(NativeFastStart.CHUNK_SIZE, remaining)
                        buf = fin.read(n)
                        if not buf:
                            break
                        fout.write(buf)
                        written += len(buf)
                        remaining -= len(buf)
                        elapsed = max(0.001, time.monotonic() - start)
                        bps = written / elapsed
                        pct = max(0.0, min(99.5, written / file_size * 100.0))
                        eta = (file_size - written) / bps if bps > 0 else 0
                        on_progress(pct, eta, bps / (1024 * 1024))

            on_progress(100.0, 0, 0)
            elapsed = time.monotonic() - start
            on_log(f"  완료 ({elapsed:.2f}s, 평균 {written/elapsed/1024/1024:.1f} MB/s)")
            return True, "OK"
        except Exception as e:
            return False, f"처리 예외: {e}"

    @staticmethod
    def _read_atoms_full(fin):
        """전체 파일에 대해 mmap-like 인터페이스 (open 한 핸들에 read all 후 반환)"""
        # 큰 파일에서 전부 읽어오면 메모리 부담 → mmap 사용
        try:
            fin.seek(0, 2)
            sz = fin.tell()
            fin.seek(0)
            return mmap.mmap(fin.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            fin.seek(0)
            return fin.read()

    @staticmethod
    def _copy(src: str, dst: str, size: int, ctrl: ProcessController,
              on_progress: Callable[[float, float, float], None]):
        copied = 0
        start = time.monotonic()
        with open(src, "rb") as fin, open(dst, "wb") as fout:
            while True:
                if ctrl.is_cancelled():
                    return
                ctrl.wait_if_paused(0.2)
                buf = fin.read(NativeFastStart.CHUNK_SIZE)
                if not buf:
                    break
                fout.write(buf)
                copied += len(buf)
                elapsed = max(0.001, time.monotonic() - start)
                bps = copied / elapsed
                pct = max(0.0, min(99.5, copied / size * 100.0)) if size else 0
                eta = (size - copied) / bps if bps > 0 else 0
                on_progress(pct, eta, bps / (1024 * 1024))


# =====================================================================
class WorkerSignals(QObject):
    state    = pyqtSignal(str, str)
    progress = pyqtSignal(str, float, float, float)
    log      = pyqtSignal(str)
    finished = pyqtSignal(str, bool, str)


class FileProcessor(QRunnable):
    def __init__(self, filepath: str, cfg: AppConfig, ctrl: ProcessController,
                 temp_dir: str):
        super().__init__()
        self.setAutoDelete(True)
        self.filepath = filepath
        self.cfg = cfg
        self.ctrl = ctrl
        self.temp_dir = temp_dir
        self.signals = WorkerSignals()

    def _resolve_output(self) -> str:
        src = Path(self.filepath)
        if self.cfg.output_mode == OutputMode.SEPARATE_DIR.value and self.cfg.output_dir:
            outdir = Path(self.cfg.output_dir)
            outdir.mkdir(parents=True, exist_ok=True)
            return str(outdir / src.name)
        if self.cfg.output_mode == OutputMode.SUFFIX.value:
            suf = self.cfg.output_suffix or "_fs"
            return str(src.with_name(src.stem + suf + src.suffix))
        return self.filepath

    def run(self):
        path = self.filepath
        try:
            if self.ctrl.is_cancelled():
                self.signals.state.emit(path, TaskState.CANCELLED.value)
                self.signals.finished.emit(path, False, "취소됨")
                return

            self.signals.state.emit(path, TaskState.CHECKING.value)
            info = MP4Reader.get_info(path)

            if info.has_faststart:
                # 출력이 별도 위치면 복사라도 진행
                final_out = self._resolve_output()
                if final_out == path:
                    self.signals.state.emit(path, TaskState.SKIPPED.value)
                    self.signals.progress.emit(path, 100.0, 0, 0)
                    self.signals.finished.emit(path, True, "이미 faststart")
                    return

            need = max(int(info.size * 1.2), 64 * 1024 * 1024)
            try:
                free = shutil.disk_usage(self.temp_dir).free
                if free < need:
                    self.signals.finished.emit(path, False,
                        f"디스크 여유 부족 (필요 {need//1024//1024}MB)")
                    self.signals.state.emit(path, TaskState.FAILED.value)
                    return
            except Exception:
                pass

            self.signals.state.emit(path, TaskState.PROCESSING.value)

            final_out = self._resolve_output()
            if final_out == path:
                temp_out = os.path.join(self.temp_dir, f"{uuid.uuid4().hex}.mp4")
            else:
                tmpdir = os.path.dirname(final_out) or self.temp_dir
                os.makedirs(tmpdir, exist_ok=True)
                temp_out = os.path.join(tmpdir, f".{uuid.uuid4().hex}.tmp.mp4")

            def on_progress(pct, eta, speed):
                self.signals.progress.emit(path, pct, eta, speed)

            def on_log(msg):
                self.signals.log.emit(msg)

            ok, msg = NativeFastStart.process(
                path, temp_out, self.ctrl, on_progress, on_log,
                audio_sync_offset_sec=self.cfg.audio_sync_offset_sec)

            if self.ctrl.is_cancelled():
                self._cleanup(temp_out)
                self.signals.state.emit(path, TaskState.CANCELLED.value)
                self.signals.finished.emit(path, False, "취소됨")
                return

            if not ok or not os.path.isfile(temp_out):
                self._cleanup(temp_out)
                self.signals.state.emit(path, TaskState.FAILED.value)
                self.signals.finished.emit(path, False, msg or "처리 실패")
                return

            # 출력 검증 (네이티브: moov 가 mdat 보다 앞에 있는지)
            if not MP4Reader.has_faststart(temp_out):
                self._cleanup(temp_out)
                self.signals.state.emit(path, TaskState.FAILED.value)
                self.signals.finished.emit(path, False, "출력 무결성 검증 실패")
                return

            try:
                if final_out == path:
                    try:
                        os.replace(temp_out, path)
                    except OSError:
                        shutil.move(temp_out, path)
                else:
                    if os.path.exists(final_out):
                        try: os.remove(final_out)
                        except Exception: pass
                    if os.path.dirname(temp_out) == os.path.dirname(final_out):
                        os.replace(temp_out, final_out)
                    else:
                        shutil.move(temp_out, final_out)
            except Exception as e:
                self._cleanup(temp_out)
                self.signals.state.emit(path, TaskState.FAILED.value)
                self.signals.finished.emit(path, False, f"파일 교체 실패: {e}")
                return

            self.signals.progress.emit(path, 100.0, 0, 0)
            self.signals.state.emit(path, TaskState.DONE.value)
            self.signals.finished.emit(path, True, "OK")
        except Exception as e:
            logger.exception(f"FileProcessor 예외: {path}")
            self.signals.state.emit(path, TaskState.FAILED.value)
            self.signals.finished.emit(path, False, f"예외: {e}")

    @staticmethod
    def _cleanup(path):
        try:
            if path and os.path.isfile(path):
                os.remove(path)
        except Exception:
            pass


class CheckWorker(QRunnable):
    def __init__(self, filepath: str):
        super().__init__()
        self.setAutoDelete(True)
        self.filepath = filepath
        self.signals = WorkerSignals()

    def run(self):
        try:
            info = MP4Reader.get_info(self.filepath)
            state = TaskState.SKIPPED.value if info.has_faststart else TaskState.PENDING.value
            payload = json.dumps({
                "path": self.filepath,
                "meta": {"size": info.size, "duration": info.duration,
                         "fs": info.has_faststart}
            }, ensure_ascii=False)
            self.signals.log.emit(f"__META__{payload}")
            self.signals.state.emit(self.filepath, state)
        except Exception as e:
            self.signals.state.emit(self.filepath, TaskState.FAILED.value)


# =====================================================================
class ProgressDelegate(QStyledItemDelegate):
    PROG_ROLE = Qt.UserRole + 1

    def paint(self, painter, option, index):
        progress = index.data(self.PROG_ROLE)
        if progress is None:
            super().paint(painter, option, index)
            return
        rect = option.rect.adjusted(4, 3, -4, -3)
        if rect.height() < 4 or rect.width() < 8:
            return
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(QColor(60, 62, 66), 1))
        painter.setBrush(QBrush(QColor(31, 33, 37)))
        painter.drawRoundedRect(rect, 3, 3)
        pct = max(0, min(100, int(progress)))
        if pct > 0:
            fill_w = max(0, int(rect.width() * pct / 100.0))
            fill_rect = QRect(rect.x(), rect.y(), fill_w, rect.height())
            grad = QLinearGradient(fill_rect.topLeft(), fill_rect.bottomLeft())
            if pct < 100:
                grad.setColorAt(0.0, QColor(255, 167, 38))
                grad.setColorAt(1.0, QColor(251, 140, 0))
            else:
                grad.setColorAt(0.0, QColor(251, 140, 0))
                grad.setColorAt(1.0, QColor(230, 81, 0))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(grad))
            painter.drawRoundedRect(fill_rect, 3, 3)
        target_px = max(8, min(13, int(rect.height() * 0.72)))
        f = painter.font()
        f.setPixelSize(target_px)
        painter.setFont(f)
        painter.setPen(QColor(232, 234, 237))
        painter.drawText(rect, Qt.AlignCenter, f"{pct}%")
        painter.restore()

    def sizeHint(self, option, index):
        s = super().sizeHint(option, index)
        if s.height() < 22:
            s.setHeight(22)
        return s


class DnDTreeWidget(QTreeWidget):
    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DropOnly)
        self._drag = False

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            urls = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
            if any(p.lower().endswith(".mp4") or os.path.isdir(p) for p in urls):
                e.acceptProposedAction()
                self._drag = True
                self.viewport().update()
                return
        e.ignore()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragLeaveEvent(self, e):
        self._drag = False
        self.viewport().update()
        e.accept()

    def dropEvent(self, e):
        self._drag = False
        self.viewport().update()
        if e.mimeData().hasUrls():
            paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
            if paths:
                self.files_dropped.emit(paths)
                e.acceptProposedAction()
                return
        e.ignore()

    def paintEvent(self, e):
        super().paintEvent(e)
        if self._drag:
            p = QPainter(self.viewport())
            p.setRenderHint(QPainter.Antialiasing)
            p.setPen(QPen(QColor(251, 140, 0, 200), 3, Qt.DashLine))
            r = self.viewport().rect().adjusted(4, 4, -4, -4)
            p.drawRect(r)
            p.end()


# =====================================================================
class SettingsDialog(QDialog):
    def __init__(self, cfg: AppConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowTitle("설정")
        self.setMinimumWidth(500)
        layout = QVBoxLayout(self)

        gb_out = QGroupBox("출력 동작")
        ol = QGridLayout(gb_out)
        self.rb_overwrite = QRadioButton("원본 덮어쓰기 (atomic replace)")
        self.rb_separate  = QRadioButton("별도 폴더 출력")
        self.rb_suffix    = QRadioButton("같은 폴더 + 접미사")
        bg = QButtonGroup(self)
        bg.addButton(self.rb_overwrite); bg.addButton(self.rb_separate); bg.addButton(self.rb_suffix)
        ol.addWidget(self.rb_overwrite, 0, 0, 1, 3)
        ol.addWidget(self.rb_separate,  1, 0, 1, 3)
        self.le_outdir = QLineEdit(cfg.output_dir)
        self.btn_browse = QPushButton("폴더 선택...")
        self.btn_browse.clicked.connect(self._browse)
        ol.addWidget(QLabel("    출력 폴더:"), 2, 0)
        ol.addWidget(self.le_outdir, 2, 1)
        ol.addWidget(self.btn_browse, 2, 2)
        ol.addWidget(self.rb_suffix, 3, 0, 1, 3)
        self.le_suffix = QLineEdit(cfg.output_suffix)
        ol.addWidget(QLabel("    접미사:"), 4, 0)
        ol.addWidget(self.le_suffix, 4, 1, 1, 2)
        layout.addWidget(gb_out)

        gb_proc = QGroupBox("처리 옵션")
        pl = QFormLayout(gb_proc)
        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["직렬 (Serial)", "병렬 (Parallel)"])
        self.cb_mode.setCurrentIndex(0 if cfg.process_mode == "serial" else 1)
        pl.addRow("처리 방식:", self.cb_mode)
        self.sp_parallel = QSpinBox()
        self.sp_parallel.setRange(1, 32)
        self.sp_parallel.setValue(cfg.max_parallel)
        pl.addRow("최대 동시 작업:", self.sp_parallel)

        self.sp_audio = QDoubleSpinBox()
        self.sp_audio.setRange(0.000, 5.000)
        self.sp_audio.setSingleStep(0.05)
        self.sp_audio.setDecimals(3)
        self.sp_audio.setSuffix("  초")
        self.sp_audio.setValue(float(cfg.audio_sync_offset_sec))
        self.sp_audio.setToolTip("audio elst.media_time 을 지정 초만큼 앞당겨\n"
                                  "음성 싱크를 보정합니다 (양수=오디오 빨라짐).")
        pl.addRow("오디오 싱크 보정:", self.sp_audio)
        layout.addWidget(gb_proc)

        gb_ui = QGroupBox("외관")
        ul = QFormLayout(gb_ui)
        self.cb_theme = QComboBox()
        self.cb_theme.addItems(["다크", "라이트"])
        self.cb_theme.setCurrentIndex(0 if cfg.theme == "dark" else 1)
        ul.addRow("테마:", self.cb_theme)
        layout.addWidget(gb_ui)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

        if cfg.output_mode == OutputMode.SEPARATE_DIR.value:
            self.rb_separate.setChecked(True)
        elif cfg.output_mode == OutputMode.SUFFIX.value:
            self.rb_suffix.setChecked(True)
        else:
            self.rb_overwrite.setChecked(True)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "출력 폴더 선택", self.le_outdir.text() or "")
        if d:
            self.le_outdir.setText(d)
            self.rb_separate.setChecked(True)

    def apply_to(self, cfg: AppConfig):
        if self.rb_separate.isChecked():
            cfg.output_mode = OutputMode.SEPARATE_DIR.value
        elif self.rb_suffix.isChecked():
            cfg.output_mode = OutputMode.SUFFIX.value
        else:
            cfg.output_mode = OutputMode.OVERWRITE.value
        cfg.output_dir            = self.le_outdir.text().strip()
        cfg.output_suffix         = self.le_suffix.text().strip() or "_fs"
        cfg.process_mode          = "serial" if self.cb_mode.currentIndex() == 0 else "parallel"
        cfg.max_parallel          = self.sp_parallel.value()
        cfg.audio_sync_offset_sec = float(self.sp_audio.value())
        cfg.theme                 = "dark" if self.cb_theme.currentIndex() == 0 else "light"


# =====================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = SettingsManager()
        self.cfg = self.settings.cfg
        self.temp_dir = os.path.join(user_data_dir(), "temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        self._cleanup_orphan()

        self.controller = ProcessController()
        self.pool = QThreadPool()
        self._update_pool()

        self._mutex = QMutex()
        self._total = 0
        self._done = 0
        self._success = 0
        self._fail = 0
        self._processing_active = False
        self._meta: Dict[str, Dict[str, Any]] = {}

        self._build_ui()
        self._build_menu()
        self._build_shortcuts()

    def _build_ui(self):
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1000, 620)
        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout(central)
        v.setContentsMargins(8, 8, 8, 4)

        top = QHBoxLayout()
        self.btn_add      = QPushButton("📁 파일 추가")
        self.btn_add_dir  = QPushButton("📂 폴더 추가 (재귀)")
        self.btn_proc_sel = QPushButton("▶ 선택 처리")
        self.btn_proc_all = QPushButton("⏵ 전체 처리")
        self.btn_pause    = QPushButton("⏸ 일시정지")
        self.btn_cancel   = QPushButton("⏹ 취소")
        self.btn_settings = QPushButton("⚙ 설정")
        for b in (self.btn_add, self.btn_add_dir, self.btn_proc_sel, self.btn_proc_all,
                  self.btn_pause, self.btn_cancel, self.btn_settings):
            b.setMinimumHeight(32)
            top.addWidget(b)
        self.btn_pause.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.btn_add.clicked.connect(self._on_add_files)
        self.btn_add_dir.clicked.connect(self._on_add_dir)
        self.btn_proc_sel.clicked.connect(lambda: self._start(True))
        self.btn_proc_all.clicked.connect(lambda: self._start(False))
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_settings.clicked.connect(self._on_settings)
        top.addStretch()
        v.addLayout(top)

        opt = QHBoxLayout()
        opt.addWidget(QLabel("방식:"))
        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["직렬 처리", "병렬 처리"])
        self.cb_mode.setCurrentIndex(0 if self.cfg.process_mode == "serial" else 1)
        self.cb_mode.currentIndexChanged.connect(self._on_mode_changed)
        opt.addWidget(self.cb_mode)
        opt.addWidget(QLabel("필터:"))
        self.le_filter = QLineEdit()
        self.le_filter.setPlaceholderText("파일명/상태로 검색...")
        self.le_filter.textChanged.connect(self._apply_filter)
        opt.addWidget(self.le_filter, 1)
        self.btn_remove = QPushButton("🗑 선택 제거")
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_clear  = QPushButton("🗑 전체 제거")
        self.btn_clear.clicked.connect(self._clear_all)
        opt.addWidget(self.btn_remove)
        opt.addWidget(self.btn_clear)
        v.addLayout(opt)

        self.splitter = QSplitter(Qt.Vertical)
        v.addWidget(self.splitter, 1)

        self.tree = DnDTreeWidget()
        self.tree.setColumnCount(6)
        self.tree.setHeaderLabels(["파일명", "크기", "길이", "상태", "진행률", "경로"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setSortingEnabled(True)
        self.tree.setItemDelegateForColumn(COL_PROGRESS, ProgressDelegate(self))
        h = self.tree.header()
        h.setSectionResizeMode(COL_NAME,     QHeaderView.Stretch)
        h.setSectionResizeMode(COL_SIZE,     QHeaderView.ResizeToContents)
        h.setSectionResizeMode(COL_DURATION, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(COL_STATUS,   QHeaderView.ResizeToContents)
        h.setSectionResizeMode(COL_PROGRESS, QHeaderView.Fixed)
        self.tree.setColumnWidth(COL_PROGRESS, 130)
        self.tree.setColumnHidden(COL_PATH, True)
        self.tree.files_dropped.connect(self._on_paths_dropped)
        self.tree.itemDoubleClicked.connect(self._on_dbl)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        self.splitter.addWidget(self.tree)

        self.log_box = QWidget()
        lv = QVBoxLayout(self.log_box)
        lv.setContentsMargins(0, 4, 0, 0)
        lv.addWidget(QLabel("처리 로그:"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 9))
        self.log.document().setMaximumBlockCount(5000)
        lv.addWidget(self.log)
        self.splitter.addWidget(self.log_box)
        if self.cfg.show_log:
            self.splitter.setSizes([460, 200])
        else:
            self.log_box.setVisible(False)
            self.splitter.setSizes([1, 0])

        bot = QHBoxLayout()
        self.lbl_progress = QLabel("준비됨")
        self.pbar = QProgressBar()
        self.pbar.setVisible(False)
        self.pbar.setMaximumHeight(18)
        bot.addWidget(self.lbl_progress, 1)
        bot.addWidget(self.pbar, 2)
        v.addLayout(bot)

        sb = QStatusBar()
        self.setStatusBar(sb)
        self._site = QLabel(f'<a href="{APP_WEBSITE}" style="color:#FFA726;">Official Website</a>')
        self._site.setOpenExternalLinks(True)
        sb.addPermanentWidget(self._site)

    def _build_menu(self):
        mb = self.menuBar()
        m_file = mb.addMenu("&파일")
        a_add = QAction("파일 추가...", self); a_add.setShortcut("Ctrl+O"); a_add.triggered.connect(self._on_add_files)
        a_dir = QAction("폴더 추가 (재귀)...", self); a_dir.setShortcut("Ctrl+Shift+O"); a_dir.triggered.connect(self._on_add_dir)
        a_quit = QAction("종료", self); a_quit.setShortcut("Ctrl+Q"); a_quit.triggered.connect(self.close)
        m_file.addAction(a_add); m_file.addAction(a_dir); m_file.addSeparator(); m_file.addAction(a_quit)

        m_run = mb.addMenu("&처리")
        a_run = QAction("전체 처리", self); a_run.setShortcut("F5"); a_run.triggered.connect(lambda: self._start(False))
        a_sel = QAction("선택 처리", self); a_sel.setShortcut("Ctrl+R"); a_sel.triggered.connect(lambda: self._start(True))
        a_pau = QAction("일시정지/재개", self); a_pau.setShortcut("Space"); a_pau.triggered.connect(self._on_pause)
        a_cc  = QAction("취소", self); a_cc.setShortcut("Esc"); a_cc.triggered.connect(self._on_cancel)
        m_run.addAction(a_run); m_run.addAction(a_sel); m_run.addSeparator(); m_run.addAction(a_pau); m_run.addAction(a_cc)

        m_view = mb.addMenu("&보기")
        self.act_show_log = QAction("처리 로그", self)
        self.act_show_log.setCheckable(True)
        self.act_show_log.setChecked(self.cfg.show_log)
        self.act_show_log.setShortcut("Ctrl+L")
        self.act_show_log.toggled.connect(self._on_toggle_log)
        m_view.addAction(self.act_show_log)

        m_set = mb.addMenu("&설정")
        a_set = QAction("환경설정...", self); a_set.setShortcut("Ctrl+,"); a_set.triggered.connect(self._on_settings)
        m_set.addAction(a_set)

        m_help = mb.addMenu("&도움말")
        a_about = QAction("정보", self); a_about.triggered.connect(self._on_about)
        m_help.addAction(a_about)

    def _build_shortcuts(self):
        self._sc_del = QShortcut(QKeySequence("Delete"), self)
        self._sc_del.activated.connect(self._remove_selected)
        self._sc_all = QShortcut(QKeySequence("Ctrl+A"), self.tree)
        self._sc_all.activated.connect(self.tree.selectAll)

    def _on_toggle_log(self, checked: bool):
        self.cfg.show_log = checked
        self.settings.save()
        self.log_box.setVisible(checked)
        if checked:
            h = max(400, self.splitter.height())
            self.splitter.setSizes([int(h * 0.7), int(h * 0.3)])
        else:
            self.splitter.setSizes([1, 0])

    def _update_pool(self):
        if self.cfg.process_mode == "parallel":
            self.pool.setMaxThreadCount(max(1, min(self.cfg.max_parallel, 32)))
        else:
            self.pool.setMaxThreadCount(1)

    def _on_add_files(self):
        start = self.cfg.recent_dirs[0] if self.cfg.recent_dirs else ""
        files, _ = QFileDialog.getOpenFileNames(self, "MP4 파일 선택", start,
                                                 "MP4 Files (*.mp4);;All Files (*)")
        if files:
            self._on_paths_dropped(files)

    def _on_add_dir(self):
        start = self.cfg.recent_dirs[0] if self.cfg.recent_dirs else ""
        d = QFileDialog.getExistingDirectory(self, "폴더 선택", start)
        if d:
            self._on_paths_dropped([d])

    def _on_paths_dropped(self, paths: List[str]):
        files: List[str] = []
        dirs: List[str] = []
        for p in paths:
            try:
                if os.path.isdir(p):
                    dirs.append(p)
                    for root, _, fnames in os.walk(p):
                        for fn in fnames:
                            if fn.lower().endswith(".mp4"):
                                files.append(os.path.join(root, fn))
                elif os.path.isfile(p) and p.lower().endswith(".mp4"):
                    files.append(p)
            except Exception as e:
                self._log(f"스캔 오류 {p}: {e}")

        if dirs:
            for d in dirs:
                if d in self.cfg.recent_dirs:
                    self.cfg.recent_dirs.remove(d)
                self.cfg.recent_dirs.insert(0, d)
            self.cfg.recent_dirs = self.cfg.recent_dirs[:10]
            self.settings.save()

        if not files:
            self._log("추가할 MP4 없음")
            return

        def norm(p):
            try:
                return os.path.normcase(os.path.normpath(os.path.abspath(p)))
            except Exception:
                return p

        existing = set(norm(p) for p in self._existing_paths())
        added = dup = 0
        for f in files:
            f = os.path.normpath(os.path.abspath(f))
            if norm(f) in existing:
                dup += 1
                continue
            self._add_item(f)
            existing.add(norm(f))
            added += 1
            cw = CheckWorker(f)
            cw.signals.state.connect(self._on_state)
            cw.signals.log.connect(self._on_check_log)
            self.pool.start(cw)
        self._log(f"추가: {added}개 / 중복: {dup}개")

    def _existing_paths(self):
        return [self.tree.topLevelItem(i).text(COL_PATH)
                for i in range(self.tree.topLevelItemCount())]

    def _add_item(self, filepath: str):
        it = QTreeWidgetItem()
        it.setText(COL_NAME,     os.path.basename(filepath))
        it.setToolTip(COL_NAME,  filepath)
        it.setText(COL_SIZE,     "...")
        it.setText(COL_DURATION, "...")
        it.setText(COL_STATUS,   TaskState.CHECKING.value)
        it.setText(COL_PATH,     filepath)
        it.setData(COL_PROGRESS, ProgressDelegate.PROG_ROLE, 0)
        self.tree.addTopLevelItem(it)

    def _find_item(self, filepath: str) -> Optional[QTreeWidgetItem]:
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            if it.text(COL_PATH) == filepath:
                return it
        return None

    def _remove_selected(self):
        items = self.tree.selectedItems()
        if not items: return
        if self._processing_active:
            QMessageBox.information(self, "알림", "처리 중에는 제거 불가")
            return
        for it in items:
            self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(it))

    def _clear_all(self):
        if self.tree.topLevelItemCount() == 0: return
        if self._processing_active:
            QMessageBox.information(self, "알림", "처리 중에는 제거 불가")
            return
        n = self.tree.topLevelItemCount()
        if QMessageBox.question(self, "확인", f"전체 {n}개 제거할까요?") == QMessageBox.Yes:
            self.tree.clear()

    def _on_mode_changed(self, idx):
        self.cfg.process_mode = "serial" if idx == 0 else "parallel"
        self._update_pool()
        self.settings.save()

    def _on_settings(self):
        dlg = SettingsDialog(self.cfg, self)
        if dlg.exec_() == QDialog.Accepted:
            old_theme = self.cfg.theme
            dlg.apply_to(self.cfg)
            self.settings.save()
            self._update_pool()
            self.cb_mode.setCurrentIndex(0 if self.cfg.process_mode == "serial" else 1)
            if old_theme != self.cfg.theme:
                if self.cfg.theme == "dark":
                    Theme.apply_dark(QApplication.instance())
                else:
                    Theme.apply_light(QApplication.instance())

    def _on_about(self):
        QMessageBox.about(self, f"{APP_NAME} 정보",
            f"<h3>{APP_NAME} v{APP_VERSION}</h3>"
            f"<p>© {APP_BUILD_YEAR} {APP_AUTHOR} - "
            f'<a href="{APP_WEBSITE}" style="color:#FFA726;">{APP_WEBSITE}</a></p>')

    def _selected_or_all_paths(self, sel_only: bool) -> List[str]:
        if sel_only:
            its = self.tree.selectedItems()
        else:
            its = [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        return [it.text(COL_PATH) for it in its]

    def _start(self, sel_only: bool):
        if self._processing_active:
            QMessageBox.information(self, "알림", "이미 처리 중")
            return
        paths = self._selected_or_all_paths(sel_only)
        if not paths:
            QMessageBox.information(self, "알림", "처리할 파일 없음")
            return
        if self.cfg.output_mode == OutputMode.SEPARATE_DIR.value:
            if not self.cfg.output_dir:
                QMessageBox.warning(self, "출력 폴더 미지정", "설정에서 폴더를 지정하세요.")
                return
            try:
                os.makedirs(self.cfg.output_dir, exist_ok=True)
            except Exception as e:
                QMessageBox.warning(self, "오류", f"출력 폴더 생성 실패: {e}")
                return

        self.controller.reset()
        with QMutexLocker(self._mutex):
            self._total = len(paths); self._done = 0
            self._success = 0; self._fail = 0
            self._processing_active = True

        self.pbar.setVisible(True)
        self.pbar.setMaximum(self._total)
        self.pbar.setValue(0)
        self._toggle_buttons(True)
        self._log(f"=== 처리 시작 (총 {self._total}개, 모드={self.cfg.process_mode}, "
                  f"출력={self.cfg.output_mode}, 오디오 보정={self.cfg.audio_sync_offset_sec:+.3f}s) ===")

        for p in paths:
            it = self._find_item(p)
            if it: it.setText(COL_STATUS, TaskState.QUEUED.value)
            w = FileProcessor(p, self.cfg, self.controller, self.temp_dir)
            w.signals.state.connect(self._on_state)
            w.signals.progress.connect(self._on_progress)
            w.signals.log.connect(self._log)
            w.signals.finished.connect(self._on_finished)
            self.pool.start(w)

    def _on_pause(self):
        if not self._processing_active: return
        if self.btn_pause.text().startswith("⏸"):
            self.controller.pause()
            self.btn_pause.setText("▶ 재개")
            self.lbl_progress.setText("일시정지 중...")
        else:
            self.controller.resume()
            self.btn_pause.setText("⏸ 일시정지")
            self.lbl_progress.setText(f"처리 중 ({self._done}/{self._total})")

    def _on_cancel(self):
        if not self._processing_active: return
        if QMessageBox.question(self, "취소", "진행 중인 작업을 취소할까요?") != QMessageBox.Yes:
            return
        self.controller.cancel()

    def _on_state(self, path: str, state: str):
        it = self._find_item(path)
        if it:
            it.setText(COL_STATUS, state)

    def _on_progress(self, path: str, pct: float, eta: float, speed: float):
        it = self._find_item(path)
        if it:
            it.setData(COL_PROGRESS, ProgressDelegate.PROG_ROLE, pct)
        if eta > 0 and pct < 99.5:
            eta_s = str(timedelta(seconds=int(eta)))
            self.lbl_progress.setText(
                f"처리 중 ({self._done}/{self._total}) - {os.path.basename(path)} - "
                f"{int(pct)}% / ETA {eta_s} {speed:.1f} MB/s")

    def _on_finished(self, path: str, success: bool, msg: str):
        with QMutexLocker(self._mutex):
            self._done += 1
            if success: self._success += 1
            else: self._fail += 1
            done = self._done; total = self._total
            success_n = self._success; fail_n = self._fail
            all_done = done >= total
        self.pbar.setValue(done)
        self.lbl_progress.setText(f"처리 중 ({done}/{total}) - 성공 {success_n} / 실패 {fail_n}")
        fname = os.path.basename(path)
        self._log(f"[{done}/{total}] {'✓' if success else '✗'} {fname}: {msg}")

        if all_done:
            self._processing_active = False
            self.pbar.setVisible(False)
            self._toggle_buttons(False)
            self.btn_pause.setText("⏸ 일시정지")
            cancelled = self.controller.is_cancelled()
            self.controller.reset()
            summary = (f"처리 완료\n\n총 {total}개 / 성공 {success_n} / 실패 {fail_n}"
                       + ("\n(취소됨)" if cancelled else ""))
            if fail_n > 0 or cancelled:
                QMessageBox.warning(self, "완료", summary)
            else:
                QMessageBox.information(self, "완료", summary)
            self.lbl_progress.setText("준비됨")

    def _on_check_log(self, msg: str):
        if msg.startswith("__META__"):
            try:
                payload = json.loads(msg[len("__META__"):])
                p = payload["path"]
                m = payload["meta"]
                self._meta[p] = m
                it = self._find_item(p)
                if it:
                    it.setText(COL_SIZE,     self._fmt_size(m.get("size", 0)))
                    it.setText(COL_DURATION, self._fmt_dur(m.get("duration", 0)))
            except Exception:
                pass
            return
        self._log(msg)

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {msg}")
        logger.info(msg)

    def _apply_filter(self, text: str):
        text = (text or "").lower().strip()
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            visible = (not text) or text in it.text(COL_NAME).lower() \
                                  or text in it.text(COL_STATUS).lower() \
                                  or text in it.text(COL_PATH).lower()
            it.setHidden(not visible)

    def _show_context_menu(self, pos):
        item = self.tree.itemAt(pos)
        if not item: return
        m = QMenu(self)
        a_open = m.addAction("📁 폴더 열기")
        a_copy = m.addAction("📋 경로 복사")
        a_retry = m.addAction("🔁 재처리")
        a_remove = m.addAction("🗑 제거")
        act = m.exec_(self.tree.viewport().mapToGlobal(pos))
        if act == a_open:
            self._open_explorer(item.text(COL_PATH))
        elif act == a_copy:
            QApplication.clipboard().setText(item.text(COL_PATH))
        elif act == a_retry:
            if not self._processing_active:
                self.tree.clearSelection(); item.setSelected(True)
                self._start(True)
        elif act == a_remove:
            if not self._processing_active:
                self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(item))

    def _on_dbl(self, item, _col):
        status = item.text(COL_STATUS)
        if (TaskState.PENDING.value in status or TaskState.FAILED.value in status
                or TaskState.CANCELLED.value in status):
            if QMessageBox.question(self, "재처리",
                    f"'{item.text(COL_NAME)}' 파일을 처리할까요?") == QMessageBox.Yes:
                self.tree.clearSelection(); item.setSelected(True)
                self._start(True)

    def _open_explorer(self, filepath: str):
        try:
            if sys.platform == "win32":
                # Tiny 버전: subprocess 사용 안함 → os.startfile 로 폴더 열기
                folder = os.path.dirname(filepath)
                if os.path.isdir(folder):
                    os.startfile(folder)
            elif sys.platform == "darwin":
                os.system(f'open -R "{filepath}"')
            else:
                os.system(f'xdg-open "{os.path.dirname(filepath)}"')
        except Exception as e:
            self._log(f"탐색기 열기 실패: {e}")

    @staticmethod
    def _fmt_size(b):
        b = int(b or 0)
        if b <= 0: return "-"
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if b < 1024:
                return f"{b:.1f} {unit}" if unit != "B" else f"{b} B"
            b /= 1024
        return f"{b:.1f} PB"

    @staticmethod
    def _fmt_dur(s):
        s = float(s or 0)
        if s <= 0: return "-"
        return str(timedelta(seconds=int(s)))

    def _toggle_buttons(self, processing: bool):
        for b in (self.btn_add, self.btn_add_dir, self.btn_proc_sel, self.btn_proc_all,
                  self.btn_settings, self.btn_remove, self.btn_clear, self.cb_mode):
            b.setEnabled(not processing)
        self.btn_pause.setEnabled(processing)
        self.btn_cancel.setEnabled(processing)

    def _cleanup_orphan(self):
        try:
            now = time.time()
            for fn in os.listdir(self.temp_dir):
                fp = os.path.join(self.temp_dir, fn)
                try:
                    if os.path.isfile(fp) and now - os.path.getmtime(fp) > 3600:
                        os.remove(fp)
                except Exception:
                    pass
        except Exception:
            pass

    def closeEvent(self, e):
        if self._processing_active:
            if QMessageBox.question(self, "종료", "진행 중인 작업을 취소하고 종료할까요?") != QMessageBox.Yes:
                e.ignore(); return
            self.controller.cancel()
            t0 = time.monotonic()
            while self._processing_active and time.monotonic() - t0 < 5:
                QApplication.processEvents()
                time.sleep(0.05)
        self.settings.save()
        e.accept()


# =====================================================================
def _load_app_icon() -> Optional[QIcon]:
    """app_icon.ico 를 다중 사이즈로 로드. 못 찾으면 None."""
    candidates = [
        os.path.join(app_base_dir(), "app_icon.ico"),
        os.path.join(os.getcwd(), "app_icon.ico"),
    ]
    icon_path = None
    for c in candidates:
        if os.path.isfile(c):
            icon_path = c
            break
    if not icon_path:
        return None
    icon = QIcon()
    # ICO 파일에 들어 있는 모든 사이즈를 명시적으로 추가
    for sz in (16, 20, 24, 32, 40, 48, 64, 96, 128, 256):
        icon.addFile(icon_path, QSize(sz, sz))
    if icon.isNull():
        return QIcon(icon_path)
    return icon


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # Windows 에서 작업표시줄/타이틀바 아이콘이 Python.exe 의 것으로 보이는 문제 방지
    if sys.platform == "win32":
        try:
            import ctypes
            myappid = f"{APP_ORG}.{APP_NAME}.{APP_VERSION}"
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName(APP_ORG)

    app_icon = _load_app_icon()
    if app_icon is not None:
        app.setWindowIcon(app_icon)

    sm = SettingsManager()
    if sm.cfg.theme == "dark":
        Theme.apply_dark(app)
    else:
        Theme.apply_light(app)

    win = MainWindow()
    if app_icon is not None:
        win.setWindowIcon(app_icon)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Fatal error")
        raise
