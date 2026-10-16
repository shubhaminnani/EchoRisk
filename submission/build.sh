#!/usr/bin/env bash
# build.sh — stage weights + checkpoints into the build context, then build
# one task image. Docker COPY can only read from the build context, so the
# large artifacts (kept outside submission/ in the repo) are staged here.
#
# Usage:  ./build.sh <task_id: 1|2|3> [tag]
#   ./build.sh 1              # -> echorisk-echoprime-task1:dev
#   ./build.sh 2 final        # -> echorisk-echoprime-task2:final
set -euo pipefail

TASK_ID="${1:?Usage: ./build.sh <task_id: 1|2|3> [tag]}"
TAG="${2:-dev}"
if [[ ! "${TASK_ID}" =~ ^[123]$ ]]; then
  echo "error: task_id must be 1, 2, or 3; got '${TASK_ID}'." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE="echorisk-echoprime-task${TASK_ID}:${TAG}"

ENCODER_SRC="${REPO_ROOT}/archive/EchoPrime/model_data/weights/echo_prime_encoder.pt"
CKPT_SRC="${REPO_ROOT}/checkpoints_echoprime"

echo "Staging weights + checkpoints into build context..."
mkdir -p "${SCRIPT_DIR}/weights"
cp -f "${ENCODER_SRC}" "${SCRIPT_DIR}/weights/echo_prime_encoder.pt"

# Only the checkpoints (best.pth per task); skip logs/last.pth to keep image lean.
for t in 1 2 3; do
  mkdir -p "${SCRIPT_DIR}/checkpoints/task${t}"
  cp -f "${CKPT_SRC}/task${t}/best.pth" "${SCRIPT_DIR}/checkpoints/task${t}/best.pth"
done

echo "Building ${IMAGE} (TASK_ID=${TASK_ID})..."
docker build --build-arg TASK_ID="${TASK_ID}" -t "${IMAGE}" "${SCRIPT_DIR}"
echo "Built ${IMAGE}."
