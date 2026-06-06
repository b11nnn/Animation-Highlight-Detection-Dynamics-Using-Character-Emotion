#!/usr/bin/env bash
# =============================================================================
# Inside Out 2 Web — 바탕화면 일괄 내보내기 스크립트
#
# 사용법 (터미널 또는 Cursor [Run Command]):
#   bash scripts/export_to_desktop.sh
#
# 또는 프로젝트 루트에서:
#   chmod +x scripts/export_to_desktop.sh && ./scripts/export_to_desktop.sh
# =============================================================================
set -euo pipefail

# ── 경로 설정 ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST="${HOME}/Desktop/insideout2-web"
TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Inside Out 2 Web → Desktop Export                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  소스 : ${SOURCE}"
echo "  대상 : ${DEST}"
echo "  시각 : ${TIMESTAMP}"
echo ""

# ── 대상 폴더 초기화 (덮어쓰기 방식 — macOS Desktop 보호 정책 대응) ─────────
mkdir -p "${DEST}"
mkdir -p "${DEST}/.streamlit"
mkdir -p "${DEST}/pages"
mkdir -p "${DEST}/data"
mkdir -p "${DEST}/insideout2/ui"
mkdir -p "${DEST}/scripts"
mkdir -p "${DEST}/insideout2_output"

echo "▸ 폴더 구조 준비 완료 (기존 파일은 덮어씀)"

# ── 파일 복사 함수 ───────────────────────────────────────────────────────────
copy_file() {
  local src="${1}"
  local rel="${2:-$(basename "${src}")}"
  local dst="${DEST}/${rel}"
  if [[ ! -f "${src}" ]]; then
    echo "  ⚠  누락 (건너뜀): ${src}"
    return 0
  fi
  mkdir -p "$(dirname "${dst}")"
  cp "${src}" "${dst}"
  echo "  ✓  ${rel}"
}

# ── 루트 파일 ────────────────────────────────────────────────────────────────
echo ""
echo "▸ 루트 파일 복사..."
copy_file "${SOURCE}/app.py"                          "app.py"
copy_file "${SOURCE}/requirements.txt"                "requirements.txt"
copy_file "${SOURCE}/Dockerfile"                      "Dockerfile"
copy_file "${SOURCE}/docker-compose.yml"              "docker-compose.yml"
copy_file "${SOURCE}/.dockerignore"                   ".dockerignore"

# ── Streamlit 설정 ───────────────────────────────────────────────────────────
echo ""
echo "▸ .streamlit/ 복사..."
copy_file "${SOURCE}/.streamlit/config.toml"          ".streamlit/config.toml"

# ── pages ────────────────────────────────────────────────────────────────────
echo ""
echo "▸ pages/ 복사..."
copy_file "${SOURCE}/pages/01_Analysis.py"              "pages/01_Analysis.py"
copy_file "${SOURCE}/pages/02_Analytics_Dashboard.py"   "pages/02_Analytics_Dashboard.py"

# ── data ─────────────────────────────────────────────────────────────────────
echo ""
echo "▸ data/ 복사..."
copy_file "${SOURCE}/data/character_queries.csv"      "data/character_queries.csv"
copy_file "${SOURCE}/data/emotion_prompts.csv"        "data/emotion_prompts.csv"

# ── insideout2 패키지 ─────────────────────────────────────────────────────────
echo ""
echo "▸ insideout2/ 복사..."
INSIDEOUT2_FILES=(
  "__init__.py"
  "config.py"
  "characters.py"
  "utils.py"
  "io_paths.py"
  "frames.py"
  "models.py"
  "detection.py"
  "annotation.py"
  "highlights.py"
  "highlight_metrics.py"
  "youtube.py"
  "visualize.py"
  "clips.py"
  "report.py"
  "pipeline.py"
)
for f in "${INSIDEOUT2_FILES[@]}"; do
  copy_file "${SOURCE}/insideout2/${f}" "insideout2/${f}"
done

echo ""
echo "▸ insideout2/ui/ 복사..."
UI_FILES=(
  "__init__.py"
  "components.py"
  "session.py"
  "results.py"
)
for f in "${UI_FILES[@]}"; do
  copy_file "${SOURCE}/insideout2/ui/${f}" "insideout2/ui/${f}"
done

# ── 내보내기 스크립트 자체도 포함 ───────────────────────────────────────────
echo ""
echo "▸ scripts/ 복사..."
copy_file "${SOURCE}/scripts/export_to_desktop.sh"    "scripts/export_to_desktop.sh"
chmod +x "${DEST}/scripts/export_to_desktop.sh"

# ── README 생성 ───────────────────────────────────────────────────────────────
cat > "${DEST}/README.md" << 'READMEEOF'
# Inside Out 2 감정 분석 웹사이트

OWL-ViT 캐릭터 탐지 + CLIP 감정 분류 + 하이라이트 점수 Streamlit 웹 서비스

## 폴더 구조

```
insideout2-web/
├── app.py                  # 홈 (랜딩 페이지)
├── pages/
│   ├── 01_분석.py          # 분석 실행
│   └── 02_결과_대시보드.py # 결과 조회
├── insideout2/             # 백엔드 파이프라인 패키지
│   ├── pipeline.py
│   ├── detection.py
│   ├── frames.py
│   └── ui/                 # 웹 UI 컴포넌트
├── data/                   # 캐릭터·감정 CSV
├── .streamlit/config.toml  # 테마·서버 설정
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## 로컬 실행

```bash
cd ~/Desktop/insideout2-web
pip install -r requirements.txt
streamlit run app.py
```

브라우저: http://localhost:8501

## Docker 실행

```bash
cd ~/Desktop/insideout2-web
docker compose up --build
```

## 재내보내기 (소스 프로젝트에서 업데이트 반영)

```bash
bash ~/insideout2-streamlit/scripts/export_to_desktop.sh
```
READMEEOF
echo "  ✓  README.md (생성)"

# ── .gitkeep for output dir ───────────────────────────────────────────────────
touch "${DEST}/insideout2_output/.gitkeep"
echo "  ✓  insideout2_output/.gitkeep"

# ── 완료 요약 ─────────────────────────────────────────────────────────────────
FILE_COUNT="$(find "${DEST}" -type f 2>/dev/null | wc -l | tr -d ' ' || echo "?")"
DIR_COUNT="$(find "${DEST}" -type d 2>/dev/null | wc -l | tr -d ' ' || echo "?")"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   ✅  내보내기 완료                                           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  📁  ${DEST}"
echo "  📄  파일 ${FILE_COUNT}개 · 폴더 ${DIR_COUNT}개"
echo ""
echo "  실행:"
echo "    cd ~/Desktop/insideout2-web"
echo "    pip install -r requirements.txt"
echo "    streamlit run app.py"
echo ""
echo "  폴더 구조 (Finder에서 ~/Desktop/insideout2-web 확인):"
echo "    insideout2-web/"
echo "    ├── app.py"
echo "    ├── pages/"
echo "    ├── insideout2/"
echo "    ├── data/"
echo "    ├── .streamlit/"
echo "    ├── scripts/export_to_desktop.sh"
echo "    └── README.md"
echo ""
