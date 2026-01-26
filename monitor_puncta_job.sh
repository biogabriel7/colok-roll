#!/bin/bash
# Monitor SLURM job progress for puncta pipeline

JOB_ID="${1:-3272245}"  # Use provided job ID or default to 3272245
SLEEP_INTERVAL=2  # Sleep for 2 seconds between checks

OUTPUT_FILE="puncta_converted_${JOB_ID}.out"
ERROR_FILE="puncta_converted_${JOB_ID}.err"

echo "=============================================="
echo "Monitoring Job: ${JOB_ID}"
echo "Output: ${OUTPUT_FILE}"
echo "Error:  ${ERROR_FILE}"
echo "=============================================="
echo ""

# Function to check if job is still running
is_job_running() {
    squeue -j "${JOB_ID}" 2>/dev/null | grep -q "${JOB_ID}"
    return $?
}

# Function to get job status
get_job_status() {
    squeue -j "${JOB_ID}" -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R" 2>/dev/null || echo "Job not found in queue"
}

# Function to show last N lines of a file
show_tail() {
    local file="$1"
    local lines="${2:-10}"
    if [ -f "$file" ]; then
        echo "--- Last ${lines} lines of $(basename $file) ---"
        tail -n "${lines}" "$file"
        echo ""
    fi
}

# Initial check
echo "Checking job status..."
get_job_status
echo ""

# Monitor loop
ITERATION=0
while is_job_running; do
    ITERATION=$((ITERATION + 1))
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "=============================================="
    echo "[${TIMESTAMP}] Check #${ITERATION}"
    echo "=============================================="
    
    # Show job status
    STATUS=$(get_job_status)
    echo "Status: ${STATUS}"
    echo ""
    
    # Show recent output
    if [ -f "${OUTPUT_FILE}" ]; then
        show_tail "${OUTPUT_FILE}" 15
    else
        echo "Output file not found yet..."
        echo ""
    fi
    
    # Show recent errors (if any)
    if [ -f "${ERROR_FILE}" ]; then
        ERROR_LINES=$(wc -l < "${ERROR_FILE}" 2>/dev/null || echo "0")
        if [ "${ERROR_LINES}" -gt 0 ]; then
            echo "--- Recent errors (if any) ---"
            tail -n 5 "${ERROR_FILE}"
            echo ""
        fi
    fi
    
    # Check for common error patterns
    if [ -f "${ERROR_FILE}" ]; then
        if grep -qi "error\|exception\|traceback\|failed" "${ERROR_FILE}" 2>/dev/null | tail -n 3; then
            echo "⚠️  WARNING: Potential errors detected in error file!"
            echo ""
        fi
    fi
    
    # Sleep before next check
    sleep "${SLEEP_INTERVAL}"
done

# Job has finished
echo "=============================================="
echo "Job ${JOB_ID} has finished"
echo "=============================================="
echo ""

# Final status check
FINAL_STATUS=$(sacct -j "${JOB_ID}" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS --noheader 2>/dev/null | head -n 1)
if [ -n "${FINAL_STATUS}" ]; then
    echo "Final Status:"
    echo "${FINAL_STATUS}"
    echo ""
fi

# Show final output
if [ -f "${OUTPUT_FILE}" ]; then
    echo "=============================================="
    echo "Final Output (last 30 lines):"
    echo "=============================================="
    tail -n 30 "${OUTPUT_FILE}"
    echo ""
fi

# Show final errors
if [ -f "${ERROR_FILE}" ]; then
    ERROR_SIZE=$(wc -l < "${ERROR_FILE}" 2>/dev/null || echo "0")
    if [ "${ERROR_SIZE}" -gt 0 ]; then
        echo "=============================================="
        echo "Error File (last 20 lines):"
        echo "=============================================="
        tail -n 20 "${ERROR_FILE}"
        echo ""
    else
        echo "No errors found in error file."
        echo ""
    fi
fi

# Check exit code from output file
if [ -f "${OUTPUT_FILE}" ]; then
    EXIT_CODE=$(grep "Exit code:" "${OUTPUT_FILE}" | tail -n 1 | awk '{print $NF}')
    if [ -n "${EXIT_CODE}" ]; then
        if [ "${EXIT_CODE}" = "0" ]; then
            echo "✅ Job completed successfully (exit code: ${EXIT_CODE})"
        else
            echo "❌ Job completed with errors (exit code: ${EXIT_CODE})"
        fi
    fi
fi

echo ""
echo "Monitoring complete."
