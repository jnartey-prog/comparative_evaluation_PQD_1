# Data dictionary

## Experimental manifests

- `development_conditions.csv`: development conditions used for property-response and nuisance assessment.
- `formal_repeatability_conditions.csv`: formal repeated-realisation conditions.
- `sealed_confirmation_conditions.csv`: disjoint confirmation conditions and deterministic seeds.
- `holdout_analysis_manifest.json`: expected confirmation row counts, source hashes and frozen analysis rules.

## Feature databases

- `outputs/development_features.sqlite`: development-stage extracted descriptors.
- `outputs/formal_features.sqlite`: formal repeated-condition descriptor records.
- `outputs/confirmation_features.sqlite`: complete frozen confirmation descriptor records for all four representations.

Database records contain condition or realisation identifiers, disturbance strata, representation identifiers, controlled reference properties, nuisance and acquisition variables, extracted descriptors, structural-estimability indicators and failure status fields as applicable.

## Development outputs

- `outputs/formal_analysis/record_level_metrics.csv`: record-level calibrated estimates and error quantities used in development summaries.
- `stratum_method_summary.csv`: disturbance-representation summary metrics.
- `calibration_cluster_bootstrap.csv`: bootstrap intervals for calibration quantities.
- `completed_development_gates.csv`: gate-level results for the 32 primary combinations.
- `criterion_level_diagnostics.csv`: criterion-specific diagnostic results.
- `physical_response_diagnostics.csv`: target-response, nuisance-response and margin evidence.
- `snr_tolerance_sensitivity.csv`: tolerance-coverage summaries by signal-to-noise ratio.
- `formal_development_summary.json`: complete structured development summary.

## Confirmation outputs

- `holdout_stratum_method_summary.json`: confirmation results obtained with frozen development relationships.
- `holdout_analysis_audit.json`: integrity checks for the confirmation analysis.
- `confirmation_extraction_audit.json`: record completeness and extraction-failure audit.
- `confirmation_generation_audit.json`: synthetic confirmation generation audit.
- `confirmation_generation_audit.json`: confirmation-generation completeness and integrity checks.

## Qualification outputs

- `primary_qualification_matrix.csv`: six-gate development status for each primary combination.
- `qualification_matrix_audit.json`: completeness and integrity checks for the matrix.
- `compound_descriptive_status.json`: status of the descriptive compound-event analyses.

Structurally undefined values are not coded as zero. They reduce availability and count as failures in all-record tolerance coverage, while numerical bias and RMSE summaries describe finite estimates only.
