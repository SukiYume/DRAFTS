#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PIPELINE_DIR="${XZ_SEARCH_PIPELINE_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"

ROOT="${XZ_SEARCH_ROOT:-}"
MANIFEST="${XZ_SEARCH_MANIFEST:-}"
OUTPUT_BASE="${XZ_SEARCH_OUTPUT_BASE:-}"
OUTPUT_GROUP="${XZ_SEARCH_OUTPUT_GROUP:-}"
STAGING_GROUP="${XZ_SEARCH_STAGING_GROUP:-data}"
DM="${XZ_SEARCH_DM:-}"
PROBABILITY="${XZ_SEARCH_PROBABILITY:-}"
DOWNSAMPLE_RATES="${XZ_SEARCH_DOWNSAMPLE_RATES:-}"
MODEL_TYPE="${XZ_SEARCH_MODEL_TYPE:-}"
MODEL_NAME="${XZ_SEARCH_MODEL_NAME:-}"

EXTRACT_SCRIPT="${XZ_SEARCH_EXTRACT_SCRIPT:-$PIPELINE_DIR/extract_xz.sh}"
SEARCH_SCRIPT="${XZ_SEARCH_SCRIPT:-$PIPELINE_DIR/d-dm-time.py}"
CLASSIFIER="${XZ_SEARCH_CLASSIFIER:-}"
PYTHON_BIN="${XZ_SEARCH_PYTHON:-python3}"
QSUB_BIN="${XZ_SEARCH_QSUB:-qsub}"
QSTAT_BIN="${XZ_SEARCH_QSTAT:-qstat}"
SSH_BIN="${XZ_SEARCH_SSH:-ssh}"
PBS_SUBMIT_HOST="${XZ_SEARCH_PBS_SUBMIT_HOST:-}"
PBS_QUEUE="${XZ_SEARCH_PBS_QUEUE:-gpu}"
PBS_NODE="${XZ_SEARCH_PBS_NODE:-}"
PBS_GRES="${XZ_SEARCH_PBS_GRES:-gpu@1}"
JOB_PREFIX="${XZ_SEARCH_JOB_PREFIX:-fdm}"
EXTRACT_CONCURRENCY="${XZ_SEARCH_EXTRACT_CONCURRENCY:-2}"
CHECK_GPU_ECC="${XZ_SEARCH_CHECK_GPU_ECC:-1}"
JOB_BASHRC="${XZ_SEARCH_JOB_BASHRC:-}"
CONDA_ENV="${XZ_SEARCH_CONDA_ENV:-}"
RUN_NOMPS="${XZ_SEARCH_RUN_NOMPS:-0}"

STATE_ROOT=
STAGE_ROOT=
LOG_ROOT=
PBS_ROOT=

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage:
  run_fixed_dm_xz.sh list
  run_fixed_dm_xz.sh start <unit_id>
  run_fixed_dm_xz.sh status <unit_id>
  run_fixed_dm_xz.sh discard-empty <unit_id>

The controller extracts one manifest unit, submits its fixed-DM search sections,
and stops at WAIT_TOA. It does not cut or calibrate candidates. Configuration is
provided through XZ_SEARCH_* variables; use an FRB-specific wrapper for routine
runs. See README.md for the required variables and manifest format.
EOF
}

require_command() {
    local command_name="$1"
    if [[ "$command_name" == */* ]]; then
        [[ -x "$command_name" ]] || die "Missing executable: $command_name"
    else
        command -v "$command_name" >/dev/null 2>&1 || die "Missing command: $command_name"
    fi
}

require_manifest() {
    local expected_header actual_header
    [[ -n "$ROOT" ]] || die "XZ_SEARCH_ROOT is required"
    [[ "$ROOT" != / ]] || die "XZ_SEARCH_ROOT must not be /"
    [[ -n "$MANIFEST" ]] || die "XZ_SEARCH_MANIFEST is required"
    [[ -f "$MANIFEST" ]] || die "Missing manifest: $MANIFEST"

    STATE_ROOT="$ROOT/state"
    STAGE_ROOT="$ROOT/staging"
    LOG_ROOT="$ROOT/logs"
    PBS_ROOT="$ROOT/pbs"

    expected_header=$'unit_id\tobs_id\tbeam\tsource_dir\tprefix\texpected_files\tsections'
    IFS= read -r actual_header < "$MANIFEST" || die "Cannot read manifest: $MANIFEST"
    actual_header=${actual_header%$'\r'}
    [[ "$actual_header" == "$expected_header" ]] || \
        die "Unexpected manifest header: $actual_header"
}

require_output_config() {
    require_manifest
    [[ -n "$OUTPUT_BASE" ]] || die "XZ_SEARCH_OUTPUT_BASE is required"
    [[ "$OUTPUT_GROUP" =~ ^[A-Za-z0-9._-]+$ ]] || \
        die "XZ_SEARCH_OUTPUT_GROUP must be one safe path component"
    [[ "$OUTPUT_GROUP" != . && "$OUTPUT_GROUP" != .. ]] || \
        die "Invalid XZ_SEARCH_OUTPUT_GROUP: $OUTPUT_GROUP"
    [[ "$STAGING_GROUP" =~ ^[A-Za-z0-9._-]+$ ]] || \
        die "XZ_SEARCH_STAGING_GROUP must be one safe path component"
    [[ "$STAGING_GROUP" != . && "$STAGING_GROUP" != .. ]] || \
        die "Invalid XZ_SEARCH_STAGING_GROUP: $STAGING_GROUP"
}

require_runtime() {
    require_output_config
    [[ "$DM" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
        die "XZ_SEARCH_DM must be a non-negative number"
    [[ "$PROBABILITY" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]] || \
        die "XZ_SEARCH_PROBABILITY must be in [0, 1]"
    [[ "$DOWNSAMPLE_RATES" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || \
        die "XZ_SEARCH_DOWNSAMPLE_RATES must be comma-separated positive integers"
    [[ -n "$MODEL_TYPE" ]] || die "XZ_SEARCH_MODEL_TYPE is required"
    [[ -n "$MODEL_NAME" ]] || die "XZ_SEARCH_MODEL_NAME is required"
    [[ -n "$PBS_NODE" ]] || die "XZ_SEARCH_PBS_NODE is required"
    [[ "$EXTRACT_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || \
        die "XZ_SEARCH_EXTRACT_CONCURRENCY must be positive"
    [[ "$CHECK_GPU_ECC" == 0 || "$CHECK_GPU_ECC" == 1 ]] || \
        die "XZ_SEARCH_CHECK_GPU_ECC must be 0 or 1"
    [[ "$RUN_NOMPS" == 0 || "$RUN_NOMPS" == 1 ]] || \
        die "XZ_SEARCH_RUN_NOMPS must be 0 or 1"

    [[ -f "$EXTRACT_SCRIPT" ]] || die "Missing extraction script: $EXTRACT_SCRIPT"
    [[ -f "$SEARCH_SCRIPT" ]] || die "Missing search implementation: $SEARCH_SCRIPT"
    [[ -f "$SCRIPT_DIR/fixed_dm_runner.py" ]] || die "Missing fixed_dm_runner.py"
    [[ -f "$CLASSIFIER" ]] || die "Missing classifier: $CLASSIFIER"
    require_command "$PYTHON_BIN"
    require_command xz
    require_command flock
    if [[ -n "$PBS_SUBMIT_HOST" ]]; then
        require_command "$SSH_BIN"
    else
        require_command "$QSUB_BIN"
        require_command "$QSTAT_BIN"
    fi
}

load_unit() {
    local wanted="$1" row
    require_manifest
    row=$(awk -F '\t' -v wanted="$wanted" '
        NR > 1 && $1 == wanted {print; matches++}
        END {if (matches != 1) exit 2}
    ' "$MANIFEST") || die "Unit not found or duplicated: $wanted"
    IFS=$'\t' read -r unit_id obs_id beam source_dir prefix expected_files section_num <<<"$row"
    section_num=${section_num%$'\r'}

    [[ "$unit_id" =~ ^[A-Za-z0-9._-]+$ ]] || die "Unsafe unit_id: $unit_id"
    [[ "$obs_id" =~ ^[A-Za-z0-9._-]+$ ]] || die "Unsafe obs_id: $obs_id"
    [[ "$beam" =~ ^M[0-9]{2}$ ]] || die "Invalid beam: $beam"
    [[ -n "$source_dir" && -n "$prefix" ]] || die "Empty source_dir or prefix for $unit_id"
    [[ "$expected_files" =~ ^[1-9][0-9]*$ ]] || die "Invalid expected_files for $unit_id"
    [[ "$section_num" =~ ^[1-9][0-9]*$ ]] || die "Invalid sections for $unit_id"
    ((section_num <= expected_files)) || \
        die "$unit_id has more sections ($section_num) than files ($expected_files)"
}

state_name() {
    local dir="$1"
    if [[ -e "$dir/CLEANED" ]]; then echo CLEANED
    elif [[ -e "$dir/WAIT_TOA" ]]; then echo WAIT_TOA
    elif [[ -e "$dir/SUBMISSION_PARTIAL" ]]; then echo SUBMISSION_PARTIAL
    elif [[ -e "$dir/SEARCH_FAILED" ]]; then echo SEARCH_FAILED
    elif [[ -e "$dir/SEARCH_SUBMITTED" ]]; then echo SEARCH_SUBMITTED
    elif [[ -e "$dir/EXTRACT_FAILED" ]]; then echo EXTRACT_FAILED
    elif [[ -e "$dir/EXTRACTED" ]]; then echo EXTRACTED
    elif [[ -e "$dir/EXTRACTING" ]]; then echo EXTRACTING
    else echo PENDING
    fi
}

assert_no_other_active_unit() {
    local current="$1" marker dir
    shopt -s nullglob
    for marker in "$STATE_ROOT"/*/EXTRACTING "$STATE_ROOT"/*/EXTRACTED \
                  "$STATE_ROOT"/*/SEARCH_SUBMITTED "$STATE_ROOT"/*/WAIT_TOA; do
        dir=$(dirname "$marker")
        if [[ "$(basename "$dir")" != "$current" ]]; then
            die "Another unit is active at $(basename "$dir"): $(state_name "$dir")"
        fi
    done
    shopt -u nullglob
}

pbs_submit() {
    local pbs_file="$1"
    if [[ -n "$PBS_SUBMIT_HOST" ]]; then
        "$SSH_BIN" -o BatchMode=yes "$PBS_SUBMIT_HOST" "$QSUB_BIN" "$pbs_file"
    else
        "$QSUB_BIN" "$pbs_file"
    fi
}

pbs_status() {
    local job_id="$1"
    if [[ -n "$PBS_SUBMIT_HOST" ]]; then
        "$SSH_BIN" -o BatchMode=yes "$PBS_SUBMIT_HOST" "$QSTAT_BIN" "$job_id"
    else
        "$QSTAT_BIN" "$job_id"
    fi
}

list_units() {
    local u o b src pref count sections
    require_manifest
    printf '%-25s %-18s %-4s %-6s %-8s %s\n' UNIT OBS_ID BEAM FILES SECTIONS STATE
    while IFS=$'\t' read -r u o b src pref count sections; do
        [[ "$u" == unit_id ]] && continue
        sections=${sections%$'\r'}
        printf '%-25s %-18s %-4s %-6s %-8s %s\n' \
            "$u" "$o" "$b" "$count" "$sections" "$(state_name "$STATE_ROOT/$u")"
    done < "$MANIFEST"
}

prepare_unit() {
    local state_dir="$1" unit_root="$2" data_dir="$3" list_file="$4"
    local found_count extracted_count temp_count existing_count
    local expected_bytes available_bytes required_bytes
    local size_report expected_sizes size_rows file_name expected_size actual_size bad_sizes=0
    local -a xz_files

    [[ -d "$source_dir" ]] || die "Missing source directory: $source_dir"
    mkdir -p "$state_dir" "$unit_root" "$data_dir" "$(dirname "$list_file")" "$LOG_ROOT/$unit_id"
    if [[ -e "$state_dir/EXTRACTED" ]]; then
        echo "[Extract] reusing verified staging data for $unit_id"
        return 0
    fi

    mapfile -t xz_files < <(
        find "$source_dir" -maxdepth 1 -type f -name "${prefix}*.fits.xz" -print | sort
    )
    found_count=${#xz_files[@]}
    [[ "$found_count" -eq "$expected_files" ]] || \
        die "$unit_id expected $expected_files compressed files, found $found_count"
    printf '%s\n' "${xz_files[@]}" > "$list_file"

    size_report="$state_dir/xz_sizes.robot.txt"
    expected_sizes="$state_dir/expected_fits_sizes.tsv"
    xz --robot --list -- "${xz_files[@]}" > "$size_report"
    awk -F '\t' '
        $1 == "name" {path = $2}
        $1 == "file" {
            sub(/\.xz$/, "", path)
            count = split(path, parts, "/")
            print parts[count] "\t" $5
        }
    ' "$size_report" > "$expected_sizes"
    size_rows=$(wc -l < "$expected_sizes")
    [[ "$size_rows" -eq "$expected_files" ]] || die "Could not read all xz sizes"
    expected_bytes=$(awk -F '\t' '{sum += $2} END {printf "%.0f\n", sum}' "$expected_sizes")

    existing_count=$(find "$data_dir" -maxdepth 1 -type f -name '*.fits' | wc -l)
    if ((existing_count > 0)) && [[ ! -e "$state_dir/EXTRACTING" && ! -e "$state_dir/EXTRACT_FAILED" ]]; then
        die "Unverified FITS already exist in $data_dir"
    fi
    available_bytes=$(df -PB1 "$unit_root" | awk 'NR == 2 {print $4}')
    required_bytes=$((expected_bytes + expected_bytes / 5))
    ((available_bytes >= required_bytes)) || \
        die "Insufficient space: need at least $required_bytes bytes, have $available_bytes"

    rm -f "$state_dir/EXTRACT_FAILED"
    touch "$state_dir/EXTRACTING"
    echo "[Extract] unit=$unit_id files=$expected_files expected_bytes=$expected_bytes"
    if ! CONCURRENT_TASKS="$EXTRACT_CONCURRENCY" \
        bash "$EXTRACT_SCRIPT" "$list_file" "$data_dir"; then
        touch "$state_dir/EXTRACT_FAILED"
        die "Extraction failed for $unit_id"
    fi

    extracted_count=$(find "$data_dir" -maxdepth 1 -type f -name '*.fits' | wc -l)
    temp_count=$(find "$data_dir" -maxdepth 1 -type f \
        \( -name '*.temp' -o -name '*.partial' \) | wc -l)
    [[ "$extracted_count" -eq "$expected_files" ]] || {
        touch "$state_dir/EXTRACT_FAILED"
        die "Extracted count mismatch: expected=$expected_files actual=$extracted_count"
    }
    [[ "$temp_count" -eq 0 ]] || {
        touch "$state_dir/EXTRACT_FAILED"
        die "Temporary extraction markers remain"
    }
    while IFS=$'\t' read -r file_name expected_size; do
        if [[ ! -f "$data_dir/$file_name" ]]; then
            echo "[Validate] missing $file_name" >&2
            bad_sizes=$((bad_sizes + 1))
            continue
        fi
        actual_size=$(stat -c %s "$data_dir/$file_name")
        if [[ "$actual_size" -ne "$expected_size" ]]; then
            echo "[Validate] $file_name expected=$expected_size actual=$actual_size" >&2
            bad_sizes=$((bad_sizes + 1))
        fi
    done < "$expected_sizes"
    if ((bad_sizes > 0)); then
        touch "$state_dir/EXTRACT_FAILED"
        die "$bad_sizes FITS file(s) failed xz-size validation"
    fi

    rm -f "$state_dir/EXTRACTING" "$state_dir/EXTRACT_FAILED"
    touch "$state_dir/EXTRACTED"
    echo "[Extract] verified $extracted_count FITS in $data_dir"
}

write_pbs_script() {
    local pbs_file="$1" state_dir="$2" data_dir="$3" section="$4"
    local section_tag job_name section_log pbs_out pbs_err
    section_tag=$(printf '%02d' "$section")
    job_name="${JOB_PREFIX}-${section_tag}"
    section_log="$LOG_ROOT/$unit_id/section${section_tag}.log"
    pbs_out="$LOG_ROOT/$unit_id/section${section_tag}.pbs.out"
    pbs_err="$LOG_ROOT/$unit_id/section${section_tag}.pbs.err"

    cat > "$pbs_file" <<EOF
#!/usr/bin/env bash
#PBS -N $job_name
#PBS -o $pbs_out
#PBS -e $pbs_err
#PBS -q $PBS_QUEUE
#PBS -l nodes=$PBS_NODE
#PBS -W x=GRES:$PBS_GRES
if [[ -n "$JOB_BASHRC" ]]; then
    source "$JOB_BASHRC"
fi
if [[ -n "$CONDA_ENV" ]]; then
    conda activate "$CONDA_ENV"
fi
if [[ "$RUN_NOMPS" == 1 ]] && command -v nomps >/dev/null 2>&1; then
    nomps || true
fi
set -euo pipefail
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
cd "$PIPELINE_DIR"
if [[ "$CHECK_GPU_ECC" == 1 ]] && command -v nvidia-smi >/dev/null 2>&1; then
    ecc=\$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [[ "\$ecc" =~ ^[0-9]+$ ]] && ((ecc > 0)); then
        echo "Refusing assigned GPU with volatile uncorrectable ECC=\$ecc" > "$section_log"
        touch "$state_dir/section${section_tag}.fail" "$state_dir/SEARCH_FAILED"
        exit 70
    fi
fi
if "$PYTHON_BIN" "$SCRIPT_DIR/fixed_dm_runner.py" \
    --section "$section" \
    --section-num "$section_num" \
    --data-path "$data_dir" \
    --output-root "$OUTPUT_BASE/$OUTPUT_GROUP" \
    --beam "$beam" \
    --dm "$DM" \
    --probability "$PROBABILITY" \
    --downsample-rates "$DOWNSAMPLE_RATES" \
    --search-script "$SEARCH_SCRIPT" \
    --classifier "$CLASSIFIER" \
    --model-type "$MODEL_TYPE" \
    --model-name "$MODEL_NAME" \
    > "$section_log" 2>&1; then
    touch "$state_dir/section${section_tag}.ok"
    (
        flock -x 9
        ok_count=\$(find "$state_dir" -maxdepth 1 -type f -name 'section*.ok' | wc -l)
        if [[ "\$ok_count" -eq "$section_num" && ! -e "$state_dir/SEARCH_FAILED" ]]; then
            touch "$state_dir/WAIT_TOA"
        fi
    ) 9>"$state_dir/finalize.lock"
else
    rc=\$?
    touch "$state_dir/section${section_tag}.fail" "$state_dir/SEARCH_FAILED"
    exit "\$rc"
fi
EOF
    chmod 700 "$pbs_file"
}

submit_search() {
    local state_dir="$1" data_dir="$2"
    local section pbs_file job_id submitted=0 section_tag
    local output_dir="$OUTPUT_BASE/$OUTPUT_GROUP/CalData/$obs_id/$beam"

    [[ -e "$state_dir/EXTRACTED" ]] || die "Unit is not extracted"
    [[ ! -e "$state_dir/SEARCH_SUBMITTED" ]] || die "Search already submitted"
    [[ ! -s "$state_dir/job_ids.txt" ]] || \
        die "Existing job IDs require manual review: $state_dir/job_ids.txt"
    if [[ -d "$output_dir" ]] && find "$output_dir" -type f -print -quit | grep -q .; then
        die "Search output already contains files: $output_dir"
    fi
    mkdir -p "$PBS_ROOT/$unit_id" "$LOG_ROOT/$unit_id"
    : > "$state_dir/job_ids.txt"
    rm -f "$state_dir/SEARCH_FAILED" "$state_dir/WAIT_TOA" \
          "$state_dir/SUBMISSION_PARTIAL" "$state_dir"/section*.ok \
          "$state_dir"/section*.fail

    for ((section = 0; section < section_num; section++)); do
        section_tag=$(printf '%02d' "$section")
        pbs_file="$PBS_ROOT/$unit_id/section${section_tag}.pbs"
        write_pbs_script "$pbs_file" "$state_dir" "$data_dir" "$section"
        if job_id=$(pbs_submit "$pbs_file"); then
            printf '%s\t%s\n' "$section_tag" "$job_id" >> "$state_dir/job_ids.txt"
            submitted=$((submitted + 1))
            echo "[Submit] unit=$unit_id section=$section_tag job=$job_id"
        else
            touch "$state_dir/SUBMISSION_PARTIAL" "$state_dir/SEARCH_FAILED"
            die "qsub failed after $submitted/$section_num submissions"
        fi
    done
    touch "$state_dir/SEARCH_SUBMITTED"
    echo "[Submit] submitted $submitted section job(s); search stops at WAIT_TOA"
}

start_unit() {
    local wanted="$1" state_dir unit_root data_dir list_file lock_file
    require_runtime
    load_unit "$wanted"
    mkdir -p "$STATE_ROOT" "$STAGE_ROOT" "$LOG_ROOT" "$PBS_ROOT"
    lock_file="$ROOT/controller.lock"
    exec 9>"$lock_file"
    flock -n 9 || die "Another controller process holds $lock_file"
    assert_no_other_active_unit "$unit_id"

    state_dir="$STATE_ROOT/$unit_id"
    unit_root="$STAGE_ROOT/$unit_id"
    data_dir="$unit_root/$STAGING_GROUP/$obs_id"
    list_file="$state_dir/xz_files.txt"
    mkdir -p "$state_dir"
    printf '%s\n' \
        "unit_id=$unit_id" "obs_id=$obs_id" "beam=$beam" \
        "source_dir=$source_dir" "prefix=$prefix" "expected_files=$expected_files" \
        "sections=$section_num" "dm=$DM" "probability=$PROBABILITY" \
        "downsample_rates=$DOWNSAMPLE_RATES" "output_group=$OUTPUT_GROUP" \
        "staging_group=$STAGING_GROUP" \
        > "$state_dir/unit.env"

    prepare_unit "$state_dir" "$unit_root" "$data_dir" "$list_file"
    submit_search "$state_dir" "$data_dir"
}

status_unit() {
    local wanted="$1" state_dir section section_tag log_file status job output_dir
    local jpg_count npy_count
    require_output_config
    load_unit "$wanted"
    state_dir="$STATE_ROOT/$unit_id"
    echo "unit=$unit_id state=$(state_name "$state_dir") obs_id=$obs_id beam=$beam"
    echo "expected_files=$expected_files sections=$section_num dm=${DM:-unset} rates=${DOWNSAMPLE_RATES:-unset}"
    if [[ -f "$state_dir/job_ids.txt" ]]; then
        echo "jobs:"
        while IFS=$'\t' read -r section_tag job; do
            if pbs_status "$job" >/dev/null 2>&1; then
                status=QUEUED_OR_RUNNING
            else
                status=LEFT_QUEUE
            fi
            printf '  section=%s job=%s %s\n' "$section_tag" "$job" "$status"
        done < "$state_dir/job_ids.txt"
    fi
    for ((section = 0; section < section_num; section++)); do
        section_tag=$(printf '%02d' "$section")
        log_file="$LOG_ROOT/$unit_id/section${section_tag}.log"
        if [[ -e "$state_dir/section${section_tag}.ok" ]]; then status=SUCCESS
        elif [[ -e "$state_dir/section${section_tag}.fail" ]]; then status=FAILED
        elif [[ -f "$log_file" ]]; then status=RUNNING_OR_BUFFERED
        else status=NOT_STARTED
        fi
        printf 'section=%s status=%s log=%s\n' "$section_tag" "$status" "$log_file"
        [[ -f "$log_file" ]] && tail -3 "$log_file"
    done
    output_dir="$OUTPUT_BASE/$OUTPUT_GROUP/CalData/$obs_id/$beam"
    if [[ -d "$output_dir" ]]; then
        jpg_count=$(find "$output_dir" -type f -name '*.jpg' | wc -l)
        npy_count=$(find "$output_dir" -type f -name '*.npy' | wc -l)
        echo "output=$output_dir jpg=$jpg_count npy=$npy_count"
    else
        echo "output=$output_dir (not created)"
    fi
}

discard_empty_unit() {
    local wanted="$1" state_dir unit_root data_dir output_dir
    local ok_count jpg_count=0 npy_count=0
    local stage_root_real unit_root_real deleted_bytes
    require_output_config
    load_unit "$wanted"
    state_dir="$STATE_ROOT/$unit_id"
    unit_root="$STAGE_ROOT/$unit_id"
    data_dir="$unit_root/$STAGING_GROUP/$obs_id"
    output_dir="$OUTPUT_BASE/$OUTPUT_GROUP/CalData/$obs_id/$beam"

    [[ -e "$state_dir/WAIT_TOA" ]] || die "$unit_id is not at WAIT_TOA"
    [[ -d "$unit_root" ]] || die "Missing staging unit: $unit_root"
    [[ ! -L "$unit_root" ]] || die "Refusing symlink staging unit: $unit_root"
    ok_count=$(find "$state_dir" -maxdepth 1 -type f -name 'section*.ok' | wc -l)
    [[ "$ok_count" -eq "$section_num" ]] || \
        die "Only $ok_count/$section_num sections succeeded"
    if [[ -d "$output_dir" ]]; then
        jpg_count=$(find "$output_dir" -type f -name '*.jpg' | wc -l)
        npy_count=$(find "$output_dir" -type f -name '*.npy' | wc -l)
    fi
    [[ "$jpg_count" -eq 0 && "$npy_count" -eq 0 ]] || \
        die "Refusing empty discard: jpg=$jpg_count npy=$npy_count"
    require_command pgrep
    if pgrep -af fixed_dm_runner.py | grep -F -- "--data-path $data_dir" >/dev/null; then
        die "An active search process still references $data_dir"
    fi

    stage_root_real=$(realpath -e "$STAGE_ROOT")
    unit_root_real=$(realpath -e "$unit_root")
    [[ "$unit_root_real" == "$stage_root_real/"* ]] || \
        die "Resolved staging unit escapes staging root: $unit_root_real"
    [[ "$unit_root_real" != "$stage_root_real" ]] || die "Refusing to remove staging root"
    deleted_bytes=$(du -sb "$unit_root_real" | awk '{print $1}')
    rm -rf -- "$unit_root_real"
    [[ ! -e "$unit_root_real" ]] || die "Staging removal did not complete"

    printf 'unit_id=%s\ndeleted_bytes=%s\ndeleted_at=%s\noriginal_xz_retained=true\n' \
        "$unit_id" "$deleted_bytes" "$(date -Is)" > "$state_dir/cleanup.env"
    touch "$state_dir/NO_CANDIDATES" "$state_dir/CLEANED"
    rm -f "$state_dir/EXTRACTING" "$state_dir/EXTRACTED" \
          "$state_dir/SEARCH_SUBMITTED" "$state_dir/WAIT_TOA" \
          "$state_dir/SEARCH_FAILED" "$state_dir/EXTRACT_FAILED" \
          "$state_dir/SUBMISSION_PARTIAL"
    echo "[Cleanup] removed $unit_root_real bytes=$deleted_bytes; original xz retained"
}

command_name="${1:-}"
case "$command_name" in
    list)
        [[ "$#" -eq 1 ]] || { usage; exit 2; }
        list_units
        ;;
    start)
        [[ "$#" -eq 2 ]] || { usage; exit 2; }
        start_unit "$2"
        ;;
    status)
        [[ "$#" -eq 2 ]] || { usage; exit 2; }
        status_unit "$2"
        ;;
    discard-empty)
        [[ "$#" -eq 2 ]] || { usage; exit 2; }
        discard_empty_unit "$2"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage
        exit 2
        ;;
esac
