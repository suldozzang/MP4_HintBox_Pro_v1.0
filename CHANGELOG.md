# MP4 HintBox Pro - 변경사항 통합 문서

> 최종 갱신 : 2026-05-02
> 제작 : EV7lab ([https://www.mp4hintbox.co.kr](https://www.mp4hintbox.co.kr))

---

## 0. 빌드 산출물 한눈에

| 산출물 | 소스 | 외부 도구 동봉 | EXE 크기 (대략) | 의존성 |
|---|---|---|---|---|
| `MP4_HintBox_Pro_v2.0.5.exe` | `mp4hint_2.0.0.py` | ffmpeg + mp4box + DLL 9종 | **~100 MB** | PyQt5 + psutil |
| `MP4_HintBox_Pro_Tiny_v2.0.5.exe` | `mp4hint_2.0.0_tiny.py` | 없음 (순수 Python) | **~25 MB** | PyQt5 |

빌드 명령은 본 문서 끝의 [부록](#부록-빌드-명령어) 참조.

---

## 1. v1.0.5 → v2.0.5 (일반 버전) 주요 차이

### 1-1. 처리 속도

| 항목 | v1.0.5 | v2.0.5 |
|---|---|---|
| FastStart 처리 방식 | FFmpeg subprocess 만 | **NativeFastStart** (FFmpeg 우회, qtfaststart 알고리즘) - 기본 ON |
| 처리 속도 | demux/remux 오버헤드 그대로 | **2~5 배 빠름** (디스크 I/O 한도까지) |
| Atom 파싱 | 첫 8 KB / 64 KB 만 (정확도 낮음) | `mmap` 기반 전체 즉시 스캔 |
| 청크 사이즈 | FFmpeg 기본 | 8 MB 시퀀셜 청크 |
| 안전성 검사 | 없음 | multi-mdat / moof / iloc 등 위험 구조는 자동 FFmpeg 폴백 |
| 32-bit 오버플로 / 검증 실패 시 | 처리 실패 | 자동 FFmpeg 폴백 |

### 1-2. 실시간 진행률·속도·ETA

| 항목 | v1.0.5 | v2.0.5 |
|---|---|---|
| 진행률 | 파일 단위 (n/총개수) | **파일별 % 진행률바** (오렌지 그라데이션) |
| ETA / 속도 | 없음 | 실시간 (예: `45% / ETA 0:01:23 / 2.30x`) |
| FFmpeg 진행률 | 파싱 안 함 | `-progress pipe:1` 의 `out_time_us` 파싱 |
| MP4Box 진행률 | 파싱 안 함 | stderr `XX %` 정규식 파싱 |
| Native 진행률 | - | 바이트 기반 실시간 (MB/s 표기) |

### 1-3. Pause / Resume / 강제 취소

| 항목 | v1.0.5 | v2.0.5 |
|---|---|---|
| 일시정지 / 재개 | **없음** | psutil `suspend()`/`resume()` (서브프로세스 트리 전체) + threading.Event (Native) |
| 취소 응답성 | subprocess.run 종료 대기 | **즉시** 자식 프로세스 트리 kill |
| 일시정지 단축키 | 없음 | `Space` |
| 취소 단축키 | 없음 | `Esc` |

### 1-4. 무결성·안전성

| 항목 | v1.0.5 | v2.0.5 |
|---|---|---|
| 출력 검증 | 없음 | ffprobe duration 5% 허용 검증 |
| 파일 교체 | `shutil.move` | `os.replace()` 원자성 보장, fallback `shutil.move` |
| 디스크 여유 확인 | 없음 | 처리 전 `원본 ×1.2` 또는 64 MB 점검 |
| 백업 | 옵션 (수동) | 옵션 + 처리 실패 시 **자동 복원** |
| 임시 파일 | 종종 잔류 | 시작 시 1 시간 이상 잔여 자동 정리 |
| 로그 | 단일 누적 | RotatingFileHandler (2 MB × 5) |

### 1-5. 음성 싱크 보존 (v2.0.3+)

| 엔진 | 추가된 옵션 |
|---|---|
| FFmpeg 카피 | `-fflags +genpts -copyts -avoid_negative_ts make_zero -map 0 -map_metadata 0` |
| FFmpeg GPU 트랜스코드 | `-vsync passthrough` |
| MP4Box 힌트 | `-inter 500 -mtu 1500 -rate 90000` |
| Native FastStart | **오디오 싱크 보정 옵션** (audio elst.media_time 을 +N초 in-place 수정) - 기본 0.150 초 |

### 1-6. 기능 (Features)

| 기능 | v1.0.5 | v2.0.5 |
|---|---|---|
| 출력 모드 | 원본 덮어쓰기만 | **덮어쓰기 / 별도 폴더 / 같은 폴더 + 접미사** 3 모드 |
| 폴더 추가 | 미지원 | **재귀 스캔** (드롭 시 하위 MP4 모두) |
| 검색 / 필터 | 미지원 | 파일명·상태·경로 즉시 필터 |
| 처리 로그 표시 | 항상 표시 | 기본 숨김, `Ctrl+L` 토글 (영구 저장) |
| 재처리 | 더블클릭만 | 더블클릭 + 우클릭 메뉴 + 선택 처리 |
| 컨텍스트 메뉴 | 없음 | 폴더 열기 / 경로 복사 / 재처리 / 제거 |
| 단축키 | 없음 | `Ctrl+O` `Ctrl+Shift+O` `F5` `Ctrl+R` `Space` `Esc` `Delete` `Ctrl+A` `Ctrl+L` `Ctrl+,` |
| 메타정보 표시 | 상태 텍스트만 | 크기 / 길이 / 코덱 / 해상도 (ffprobe) |
| 정렬 | 미지원 | 컬럼 클릭 정렬 |

### 1-7. UI / UX

| 항목 | v1.0.5 | v2.0.5 |
|---|---|---|
| 위젯 | QListWidget (단순 텍스트) | **QTreeWidget** + 6 컬럼 |
| 진행률 표시 | 텍스트만 | 셀 내 **오렌지 그라데이션 진행률바** (자동 폰트 축소) |
| 테마 | 시스템 기본 | **모던 다크** (Fusion + 커스텀 팔레트 + 스타일시트) / 라이트 전환 가능 |
| 드래그&드롭 | 점선 단색 | 오렌지 점선 |
| 메뉴바 | 없음 | 파일 / 처리 / 보기 / 설정 / 도움말 |
| 상태바 | 없음 | GPU/CPU 정보 + Official Website 링크 |
| 설정 다이얼로그 | 없음 | 출력 / 처리 / 외관 통합 다이얼로그 (Native FastStart 토글, 오디오 싱크 보정 SpinBox 포함) |

### 1-8. 설정 영구 저장 (QSettings)

v2.0.5 는 다음을 모두 자동 복원:

- 처리 방식 (직렬 / 병렬), 최대 동시 작업 수
- 엔진 (FFmpeg / MP4Box)
- 하드웨어 가속 사용 여부
- **Native FastStart 사용 여부**
- **오디오 싱크 보정 값** (초)
- 백업 사용 여부
- 출력 모드 / 폴더 / 접미사
- 타임아웃, 무결성 검증
- 테마 (다크 / 라이트)
- **처리 로그 표시 여부**
- 최근 사용 폴더 10 개

### 1-9. 아키텍처 / 코드 품질

| 항목 | v1.0.5 | v2.0.5 |
|---|---|---|
| 워커 | `ProcessingWorker(QThread)` + `ProcessWorker(QRunnable)` 두 종류 (중복) | **`FileProcessor(QRunnable)` 단일** + ThreadPool max=1 로 직렬 통일 |
| 일시정지 / 취소 | 워커별 플래그 | **`ProcessController`** 중앙화 |
| 엔진 호출 | 워커마다 cmd 빌드 | **`EngineRunner`** 통합 래퍼 (Native / FFmpeg / MP4Box) |
| 메타데이터 | 미저장 | `MediaInfo` dataclass + 캐시 |
| 설정 모델 | 클래스 변수 | `AppConfig` dataclass + `SettingsManager` |
| 상태값 | 문자열 하드코딩 | `Enum` (`Engine`, `OutputMode`, `TaskState`) |
| 경로 처리 | `str` 위주 | `pathlib.Path` + `os.path.normcase` 정규화 (대소문자 dedup) |
| 로깅 | basicConfig | RotatingFileHandler + 콘솔 |
| 타입 힌트 | 부분 | 전반 (Optional / Tuple / List / Dict / Callable) |

### 1-10. 빌드 / 패키징

| 항목 | v1.0.5 | v2.0.5 |
|---|---|---|
| .spec | 외부 도구 동봉 안 함 | ffmpeg / ffprobe / mp4box / DLL 모두 자동 동봉 |
| 제외 모듈 | tkinter / matplotlib / numpy / pandas / scipy | 위 + PyQt5 의 WebEngine / Multimedia / Qml / Quick |
| optimize | 0 | 2 |
| UPX 제외 | - | ffmpeg / mp4box / DLL UPX 제외 (실행 안정성) |
| hiddenimports | wmi / pythoncom / pywintypes | psutil / PyQt5.sip |

### 1-11. 브랜딩

| 항목 | v1.0.5 | v2.0.5 |
|---|---|---|
| 제작 표기 | suldo.com | **EV7lab** |
| 공식 웹사이트 | suldo.com | **www.mp4hintbox.co.kr** |

---

## 2. Tiny v2.0.5 (`mp4hint_2.0.0_tiny.py`) 와의 차이점

Tiny 버전은 **NativeFastStart 단독** 으로 동작하는 경량 빌드입니다.

### 2-1. 일반 버전 vs Tiny 버전 비교표

| 항목 | 일반 v2.0.5 | Tiny v2.0.5 |
|---|---|---|
| **EXE 크기** | ~100 MB | **~30 MB** |
| **외부 의존성** | ffmpeg.exe / mp4box.exe / DLL 9종 동봉 | **없음** |
| **Python 의존성** | PyQt5 + psutil | **PyQt5 만** |
| **FastStart 엔진** | NativeFastStart (기본) + FFmpeg 폴백 | **NativeFastStart 단독** |
| **Hint Track** | MP4Box 엔진 지원 | **미지원** (faststart 전용) |
| **GPU 트랜스코드** | NVIDIA / Intel / AMD 자동 감지·사용 | **미지원** (코덱 변경 불가) |
| **오디오 싱크 보정** | ✓ (audio elst.media_time in-place) | ✓ (동일 알고리즘) |
| **처리 속도** | Native 사용 시 동일, FFmpeg 폴백 시 절반 | 항상 Native (가장 빠름) |
| **위험 구조 처리** | multi-mdat / moof / iloc → FFmpeg 자동 폴백 | multi-mdat / moof / iloc → **처리 거부** (실패 보고) |
| **무결성 검증** | ffprobe (duration ±5%) | **자체 atom 검사** (moov 가 mdat 보다 앞에 있는지) |
| **메타 정보** | 코덱 / 해상도 / 길이 (ffprobe) | 길이만 (mvhd 직접 파싱) |
| **Pause / Resume** | psutil suspend/resume + threading.Event | **threading.Event 만** (chunk 단위 일시정지) |
| **로그 / 테마 / 출력 모드 / 검색 / 폴더 재귀** | 동일 |  |
| **단축키** | 동일 | 동일 |
| **시작 시간** | 1~3 초 (PyInstaller unpack) | **0.5~1 초** (작은 EXE) |

### 2-2. Tiny 버전이 처리할 수 없는 케이스

다음은 **처리 거부**되며 사용자에게 실패로 보고됩니다 (자동 폴백 없음):

- 다중 mdat (`mdat × 2 이상`) - 카메라 일부 / 일부 라이브 녹화 파일
- 다중 moov (`moov × 2 이상`) - 손상된 파일
- Fragmented MP4 (`moof` 존재) - DASH / HLS 세그먼트 / YouTube 다운로드 일부
- 최상위 `meta + iloc` - HEIC / HEIF / 일부 카메라 파일
- 코덱 변환이 필요한 케이스 - 일반 버전의 GPU 트랜스코드 모드를 사용해야 함

위 케이스가 자주 발생한다면 **일반 버전**을 사용하세요.

### 2-3. Tiny 버전 권장 사용처

- 일반적인 카메라 / 휴대폰 녹화 MP4 (단일 mdat, 비-fragmented)
- 단순 faststart 처리만 필요 (스트리밍 서버 업로드 전 처리 등)
- 외부 도구 설치 / EXE 크기 / 시작 속도가 중요한 환경
- USB 휴대용 / 한 폴더에 같이 두고 쓰는 경우

### 2-4. 기능 매트릭스 요약

| 기능 분류 | 일반 v2.0.5 | Tiny v2.0.5 |
|---|---|---|
| FastStart 처리 | ✓ | ✓ |
| Hint Track 처리 | ✓ | ✗ |
| GPU 트랜스코드 | ✓ | ✗ |
| 오디오 싱크 보정 | ✓ | ✓ |
| Native 안전성 검사 | ✓ (실패 시 폴백) | ✓ (실패 시 거부) |
| 모던 다크 UI | ✓ | ✓ |
| 출력 모드 3종 | ✓ | ✓ |
| 폴더 재귀 / 검색 / 필터 | ✓ | ✓ |
| Pause / Resume / 취소 | ✓ | ✓ |
| 진행률 / ETA / 속도 | ✓ | ✓ |
| 무결성 검증 | ffprobe | atom 검사 |
| 백업·복원 | ✓ | ✗ (단순화) |
| 컨텍스트 메뉴 | ✓ | ✓ |
| 단축키 풀세트 | ✓ | ✓ |
| 설정 영구 저장 | ✓ | ✓ |

---

## 부록: 빌드 명령어

### 일반 버전 (v2.0.5)

```cmd
pyinstaller --noconsole --onefile ^
  --name="MP4_HintBox_Pro_v2.0.5" ^
  --icon="app_icon.ico" ^
  --add-binary "ffmpeg.exe;." ^
  --add-binary "ffprobe.exe;." ^
  --add-binary "mp4box.exe;." ^
  --add-binary "libgpac.dll;." ^
  --add-binary "MediaInfo.dll;." ^
  --add-binary "libeay32.dll;." ^
  --add-binary "ssleay32.dll;." ^
  --add-binary "js.dll;." ^
  --add-binary "js32.dll;." ^
  --add-data "app_icon.ico;." ^
  --add-data "license.txt;." ^
  --add-data "README.txt;." ^
  --exclude-module=tkinter ^
  --exclude-module=matplotlib ^
  --exclude-module=numpy ^
  --exclude-module=pandas ^
  --exclude-module=scipy ^
  --exclude-module=PyQt5.QtWebEngineCore ^
  --exclude-module=PyQt5.QtWebEngineWidgets ^
  --exclude-module=PyQt5.QtMultimedia ^
  --exclude-module=PyQt5.QtMultimediaWidgets ^
  --exclude-module=PyQt5.QtQml ^
  --exclude-module=PyQt5.QtQuick ^
  --hidden-import=psutil ^
  --hidden-import=PyQt5.sip ^
  mp4hint_2.0.0.py
```

산출물 : `dist\MP4_HintBox_Pro_v2.0.5.exe`

### Tiny 버전 (v2.0.0)

```cmd
pyinstaller --noconsole --onefile ^
  --name="MP4_HintBox_Pro_Tiny_v2.0.5" ^
  --icon="app_icon.ico" ^
  --exclude-module=tkinter ^
  --exclude-module=matplotlib ^
  --exclude-module=numpy ^
  --exclude-module=pandas ^
  --exclude-module=scipy ^
  --exclude-module=PyQt5.QtWebEngineCore ^
  --exclude-module=PyQt5.QtWebEngineWidgets ^
  --exclude-module=PyQt5.QtMultimedia ^
  --exclude-module=PyQt5.QtMultimediaWidgets ^
  --exclude-module=PyQt5.QtQml ^
  --exclude-module=PyQt5.QtQuick ^
  --hidden-import=PyQt5.sip ^
  mp4hint_2.0.0_tiny.py
```

산출물 : `dist\MP4_HintBox_Pro_Tiny_v2.0.5.exe`

### 빌드 전 의존성 설치

```cmd
py -m pip install --upgrade pip
py -m pip install pyqt5 psutil pyinstaller
```

> Tiny 버전만 빌드한다면 `psutil` 은 생략해도 됩니다.

### 한 줄 명령어 (cmd 줄바꿈 없는 형태)

**일반 버전:**

```
pyinstaller --noconsole --onefile --name="MP4_HintBox_Pro_v2.0.5" --icon="app_icon.ico" --add-binary "ffmpeg.exe;." --add-binary "ffprobe.exe;." --add-binary "mp4box.exe;." --add-binary "libgpac.dll;." --add-binary "MediaInfo.dll;." --add-binary "libeay32.dll;." --add-binary "ssleay32.dll;." --add-binary "js.dll;." --add-binary "js32.dll;." --add-data "app_icon.ico;." --add-data "license.txt;." --add-data "README.txt;." --exclude-module=tkinter --exclude-module=matplotlib --exclude-module=numpy --exclude-module=pandas --exclude-module=scipy --exclude-module=PyQt5.QtWebEngineCore --exclude-module=PyQt5.QtWebEngineWidgets --exclude-module=PyQt5.QtMultimedia --exclude-module=PyQt5.QtMultimediaWidgets --exclude-module=PyQt5.QtQml --exclude-module=PyQt5.QtQuick --hidden-import=psutil --hidden-import=PyQt5.sip mp4hint_2.0.0.py
```

**Tiny 버전:**

```
pyinstaller --noconsole --onefile --name="MP4_HintBox_Pro_Tiny_v2.0.5" --icon="app_icon.ico" --exclude-module=tkinter --exclude-module=matplotlib --exclude-module=numpy --exclude-module=pandas --exclude-module=scipy --exclude-module=PyQt5.QtWebEngineCore --exclude-module=PyQt5.QtWebEngineWidgets --exclude-module=PyQt5.QtMultimedia --exclude-module=PyQt5.QtMultimediaWidgets --exclude-module=PyQt5.QtQml --exclude-module=PyQt5.QtQuick --hidden-import=PyQt5.sip mp4hint_2.0.0_tiny.py
```

---

## 한 줄 요약

> **v1.0.5** : 동작하는 GUI 변환 도구
> **v2.0.5 (일반)** : Native + FFmpeg + MP4Box 통합, 진행률 / Pause / 무결성 / 출력 모드 / 검색 / 다크 / 오디오 싱크 보정까지 갖춘 **상용 배포용 빌드** (~100 MB)
> **v2.0.0 (Tiny)** : 동일 UX 에서 NativeFastStart 만 단독 동작하는 **경량 빌드** (~25 MB, 외부 도구 제로)
