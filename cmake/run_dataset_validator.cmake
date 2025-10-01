set(DATA_FILE "${PROJECT_SOURCE_DIR}/tests/data/dbscan_static_data.bin")
set(TRUTH_FILE "${PROJECT_SOURCE_DIR}/tests/data/dbscan_static_truth.bin")
set(VALIDATOR "${PROJECT_BINARY_DIR}/dbscan_dataset_validator")

if(NOT EXISTS "${DATA_FILE}")
  message(FATAL_ERROR "Static dataset not found: ${DATA_FILE}")
endif()

if(NOT EXISTS "${TRUTH_FILE}")
  message(FATAL_ERROR "Truth labels file not found: ${TRUTH_FILE}")
endif()

if(NOT EXISTS "${VALIDATOR}")
  message(FATAL_ERROR "dbscan_dataset_validator executable not found: ${VALIDATOR}")
endif()

execute_process(
  COMMAND "${VALIDATOR}"
    "--data" "${DATA_FILE}"
    "--truth" "${TRUTH_FILE}"
    "--eps" "10"
    "--min-samples" "3"
    "--impl" "both"
  WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
  RESULT_VARIABLE VALIDATOR_RESULT
  COMMAND_ECHO STDOUT
)

if(NOT VALIDATOR_RESULT EQUAL 0)
  message(FATAL_ERROR "dbscan_dataset_validator reported a mismatch (exit code ${VALIDATOR_RESULT})")
endif()
