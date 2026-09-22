#!/usr/bin/env bash
# scripts/convert_phowhisper.sh
# ============================================================
# Chuyển đổi vinai/PhoWhisper-base sang định dạng CTranslate2 int8
# để dùng với faster-whisper.
#
# Spec: B1_asr_setup_phowhisper.md §3
#
# Cách dùng:
#   bash scripts/convert_phowhisper.sh
#   bash scripts/convert_phowhisper.sh --force   # ghi đè nếu đã tồn tại
#
# Yêu cầu:
#   pip install ctranslate2 transformers
#
# Idempotent: nếu thư mục đích đã đủ file cần thiết, bỏ qua (trừ khi --force).
# ============================================================
set -euo pipefail

# ── Cấu hình ──────────────────────────────────────────────────────────────
HF_MODEL="vinai/PhoWhisper-base"
OUTPUT_DIR="models/asr/phowhisper-base-ct2"
QUANTIZATION="int8"

# File bắt buộc phải có sau khi convert (kiểm tra idempotency)
REQUIRED_FILES=(
    "model.bin"
    "config.json"
    "vocabulary.json"
    "tokenizer.json"
    "preprocessor_config.json"
    "tokenizer_config.json"
)

FORCE=false
if [[ "${1:-}" == "--force" ]]; then
    FORCE=true
fi

# ── Đường dẫn repo gốc ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ABS_OUTPUT="${REPO_ROOT}/${OUTPUT_DIR}"

# ── Kiểm tra idempotency ──────────────────────────────────────────────────
if [[ "$FORCE" == false ]] && [[ -d "$ABS_OUTPUT" ]]; then
    all_present=true
    missing_files=()
    for f in "${REQUIRED_FILES[@]}"; do
        if [[ ! -f "${ABS_OUTPUT}/${f}" ]]; then
            all_present=false
            missing_files+=("$f")
        fi
    done

    if [[ "$all_present" == true ]]; then
        echo "[convert_phowhisper] ✅ Thư mục đích đã đủ file — bỏ qua convert."
        echo "   Đường dẫn: ${ABS_OUTPUT}"
        echo "   Dùng --force để ghi đè."
        exit 0
    else
        echo "[convert_phowhisper] ⚠️  Thư mục tồn tại nhưng thiếu: ${missing_files[*]}"
        echo "   Tiến hành convert lại..."
    fi
fi

# ── Kiểm tra ct2-transformers-converter ──────────────────────────────────
if ! command -v ct2-transformers-converter &> /dev/null; then
    echo "[convert_phowhisper] ❌ Không tìm thấy ct2-transformers-converter."
    echo "   Cài đặt bằng: pip install ctranslate2"
    exit 1
fi

# ── Tạo thư mục đích ─────────────────────────────────────────────────────
mkdir -p "${ABS_OUTPUT}"

# ── Chạy convert ──────────────────────────────────────────────────────────
echo "[convert_phowhisper] 🔄 Đang convert ${HF_MODEL} → ${ABS_OUTPUT}"
echo "   Quantization: ${QUANTIZATION}"
echo "   Có thể mất vài phút lần đầu (tải checkpoint từ HuggingFace)..."

ct2-transformers-converter \
    --model "${HF_MODEL}" \
    --output_dir "${ABS_OUTPUT}" \
    --quantization "${QUANTIZATION}" \
    --copy_files \
        tokenizer.json \
        preprocessor_config.json \
        tokenizer_config.json \
    --force

# ── Kiểm tra sau convert ─────────────────────────────────────────────────
echo ""
echo "[convert_phowhisper] 🔍 Kiểm tra file đầu ra..."
all_ok=true
for f in "${REQUIRED_FILES[@]}"; do
    if [[ -f "${ABS_OUTPUT}/${f}" ]]; then
        size=$(du -sh "${ABS_OUTPUT}/${f}" 2>/dev/null | cut -f1)
        echo "   ✅ ${f}  (${size})"
    else
        echo "   ❌ THIẾU: ${f}"
        all_ok=false
    fi
done

echo ""
if [[ "$all_ok" == true ]]; then
    total_size=$(du -sh "${ABS_OUTPUT}" 2>/dev/null | cut -f1)
    echo "[convert_phowhisper] ✅ Convert thành công!"
    echo "   Thư mục: ${ABS_OUTPUT}"
    echo "   Tổng dung lượng: ${total_size}"
    echo ""
    echo "💡 Nếu > 100 MB, thêm vào .gitignore hoặc Git LFS:"
    echo "   echo 'models/asr/' >> .gitignore"
    echo "   # hoặc: git lfs track 'models/asr/**/*.bin'"
else
    echo "[convert_phowhisper] ❌ Một số file bị thiếu — kiểm tra lại lỗi bên trên."
    exit 1
fi
