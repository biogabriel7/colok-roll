#!/bin/bash
for i in {1..20}; do
  echo "=== Check $i at $(date +%H:%M:%S) ==="
  status=$(squeue -u $USER -j 3264225 -h -o '%T %M' 2>/dev/null)
  if [ -z "$status" ]; then
    echo "Job completed!"
    echo -e "\n=== Final Summary ===" 
    tail -50 puncta_batch_3264225.err | grep -E "(SUMMARY|completed|Success|Error)"
    break
  fi
  echo "Status: $status"
  echo "Progress: $(tail -3 puncta_batch_3264225.err)"
  echo ""
  sleep 180  # 3 minutes
done
