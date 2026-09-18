# WORKSPACE MIGRATION AUDIT: `C:\legal-agentic-rag-m50-o2-clean`

- **Total files scanned:** 18843
- **Total size:** 152690.13 MB

## Category Breakdown

| Category | File Count | Size (MB) |
|---|---:|---:|
| `A. FINAL_PRODUCTION_SOURCE` | 93 | 0.86 MB |
| `B. FINAL_PRODUCTION_ARTIFACT` | 44 | 1334.67 MB |
| `C. FINAL_REPRODUCTION_SCRIPT` | 0 | 0.00 MB |
| `D. FINAL_CONFIG` | 16 | 0.04 MB |
| `E. TEST` | 4268 | 72.58 MB |
| `F. DOCUMENTATION_SOURCE` | 30 | 0.52 MB |
| `G. SUCCESSFUL_EXPERIMENT_EVIDENCE` | 5602 | 27130.96 MB |
| `H. FAILED/ABANDONED_EXPERIMENT` | 399 | 378.84 MB |
| `I. TEMPORARY/HOTFIX/DEBUG` | 7174 | 39394.46 MB |
| `J. LARGE_EXTERNAL_MODEL/DATA` | 496 | 81902.76 MB |
| `K. SECRET/SENSITIVE` | 287 | 89.54 MB |
| `L. UNKNOWN_REQUIRES_REVIEW` | 434 | 2384.89 MB |

## A. Key Final Production Sources

- `scratch/p63_clean_extraction_test_r3/P63_DEPLOYMENT_INFERENCE.py` (9599 bytes)
- `scratch/p63_clean_extraction_test_r3/P63_UNSEEN_MATERIALIZER.py` (11850 bytes)
- `scratch/p63_clean_extraction_test_r3/P63_UPSTREAM_RUNTIME_ADAPTER.py` (17771 bytes)
- `scratch/p63_r4_extract_validation_1789484482/P63_DEPLOYMENT_INFERENCE.py` (11735 bytes)
- `scratch/p63_r4_extract_validation_1789484482/P63_UNSEEN_MATERIALIZER.py` (11850 bytes)
- `scratch/p63_r4_extract_validation_1789484482/P63_UPSTREAM_RUNTIME_ADAPTER.py` (17771 bytes)
- `scratch/p63_unseen_query_deployment_staging/P63_DEPLOYMENT_INFERENCE.py` (2696 bytes)
- `scratch/p63_unseen_query_deployment_staging/P63_UNSEEN_MATERIALIZER.py` (11458 bytes)
- `scratch/p63_unseen_query_deployment_staging_r2/P63_DEPLOYMENT_INFERENCE.py` (4183 bytes)
- `scratch/p63_unseen_query_deployment_staging_r2/P63_UNSEEN_MATERIALIZER.py` (11850 bytes)
- `scratch/p63_unseen_query_deployment_staging_r2/P63_UPSTREAM_RUNTIME_ADAPTER.py` (9569 bytes)
- `scratch/p63_unseen_query_deployment_staging_r3/P63_DEPLOYMENT_INFERENCE.py` (9599 bytes)
- `scratch/p63_unseen_query_deployment_staging_r3/P63_UNSEEN_MATERIALIZER.py` (11850 bytes)
- `scratch/p63_unseen_query_deployment_staging_r3/P63_UPSTREAM_RUNTIME_ADAPTER.py` (17771 bytes)
- `scratch/p63_unseen_query_deployment_staging_r4/P63_DEPLOYMENT_INFERENCE.py` (11735 bytes)
- `scratch/p63_unseen_query_deployment_staging_r4/P63_UNSEEN_MATERIALIZER.py` (11850 bytes)
- `scratch/p63_unseen_query_deployment_staging_r4/P63_UPSTREAM_RUNTIME_ADAPTER.py` (17771 bytes)
- `scratch/p64_p63_production_integration_prep_r1/src/legal_agentic_rag/selection/p63_production_integration.py` (11012 bytes)
- `scratch/p64_p63_production_integration_prep_r2/src/legal_agentic_rag/selection/p63_production_integration.py` (15661 bytes)
- `scratch/p64_r1_extract_validation_1789486105/src/legal_agentic_rag/selection/p63_production_integration.py` (11012 bytes)
- `scratch/p64_r2_extract_validation/src/legal_agentic_rag/selection/p63_production_integration.py` (15661 bytes)
- `scratch/p65_p63_production_promotion_r1/src/legal_agentic_rag/selection/p63_production_integration.py` (15661 bytes)
- `scratch/p66_high_precision_extractive_fallback_r1/src/legal_agentic_rag/generation/p66_extractive_finalizer.py` (8704 bytes)
- `scratch/p66_one_shot_final_decision_r1/src/legal_agentic_rag/generation/p66_extractive_finalizer.py` (8704 bytes)
- `src/legal_agentic_rag/generation/article_authority.py` (12754 bytes)
- `src/legal_agentic_rag/generation/citation_verifier.py` (3350 bytes)
- `src/legal_agentic_rag/generation/claim_grounding.py` (12176 bytes)
- `src/legal_agentic_rag/generation/context_builder.py` (11390 bytes)
- `src/legal_agentic_rag/generation/context_grader.py` (6745 bytes)
- `src/legal_agentic_rag/generation/evidence_packer.py` (5815 bytes)

## B. Key Final Production Artifacts

- `p4_p2r_final/p2r_adapter/adapter_config.json` (1148 bytes)
- `p4_p2r_final/p2r_adapter/adapter_model.safetensors` (87322136 bytes)
- `p4_p2s_final/p2s_adapter/adapter_config.json` (1148 bytes)
- `p4_p2s_final/p2s_adapter/adapter_model.safetensors` (87322136 bytes)
- `scratch/master_one_shot_impl/adapter_config.json` (1148 bytes)
- `scratch/master_one_shot_impl/adapter_model.safetensors` (87322136 bytes)
- `scratch/p62_oof_action_eval50_prep_r1_staging/p2_adapter/adapter_config.json` (1148 bytes)
- `scratch/p62_oof_action_eval50_prep_r1_staging/p2_adapter/adapter_model.safetensors` (87322136 bytes)
- `scratch/p62_oof_action_eval50_prep_r2_staging/p2_adapter/adapter_config.json` (1148 bytes)
- `scratch/p62_oof_action_eval50_prep_r2_staging/p2_adapter/adapter_model.safetensors` (87322136 bytes)
- `scratch/p63_clean_extraction_test_r3/P63_DEPLOYMENT_MODEL.pkl` (387527 bytes)
- `scratch/p63_clean_extraction_test_r3/P63_DEPLOYMENT_MODEL_METADATA.json` (778 bytes)
- `scratch/p63_r4_extract_validation_1789484482/P63_DEPLOYMENT_MODEL.pkl` (387527 bytes)
- `scratch/p63_r4_extract_validation_1789484482/P63_DEPLOYMENT_MODEL_METADATA.json` (778 bytes)
- `scratch/p63_unseen_query_deployment_staging/P63_DEPLOYMENT_MODEL.pkl` (387527 bytes)
- `scratch/p63_unseen_query_deployment_staging/P63_DEPLOYMENT_MODEL_METADATA.json` (778 bytes)
- `scratch/p63_unseen_query_deployment_staging_r2/P63_DEPLOYMENT_MODEL.pkl` (387527 bytes)
- `scratch/p63_unseen_query_deployment_staging_r2/P63_DEPLOYMENT_MODEL_METADATA.json` (778 bytes)
- `scratch/p63_unseen_query_deployment_staging_r3/P63_DEPLOYMENT_MODEL.pkl` (387527 bytes)
- `scratch/p63_unseen_query_deployment_staging_r3/P63_DEPLOYMENT_MODEL_METADATA.json` (778 bytes)
- `scratch/p63_unseen_query_deployment_staging_r4/P63_DEPLOYMENT_MODEL.pkl` (387527 bytes)
- `scratch/p63_unseen_query_deployment_staging_r4/P63_DEPLOYMENT_MODEL_METADATA.json` (778 bytes)
- `scratch/p66_kaggle_one_shot_prep_r1_staging/p2_adapter/adapter_config.json` (1148 bytes)
- `scratch/p66_kaggle_one_shot_prep_r1_staging/p2_adapter/adapter_model.safetensors` (87322136 bytes)
- `scratch/p70_kaggle_t4x2_prep_r1_staging/p2_adapter/adapter_config.json` (1148 bytes)
- `scratch/p70_kaggle_t4x2_prep_r1_staging/p2_adapter/adapter_model.safetensors` (87322136 bytes)
- `scratch/p70_kaggle_t4x2_prep_r2_staging/p2_adapter/adapter_config.json` (1148 bytes)
- `scratch/p70_kaggle_t4x2_prep_r2_staging/p2_adapter/adapter_model.safetensors` (87322136 bytes)
- `scratch/p70_kaggle_t4x2_prep_r3_staging/p2_adapter/adapter_config.json` (1148 bytes)
- `scratch/p70_kaggle_t4x2_prep_r3_staging/p2_adapter/adapter_model.safetensors` (87322136 bytes)
- `scratch/p70_public1000_final_execution_r1_staging/p70_adapter/adapter_config.json` (1148 bytes)
- `scratch/p70_public1000_final_execution_r1_staging/p70_adapter/adapter_model.safetensors` (87322136 bytes)
- `scratch/p70_public1000_final_execution_r2_staging/p70_adapter/adapter_config.json` (1148 bytes)
- `scratch/p70_public1000_final_execution_r2_staging/p70_adapter/adapter_model.safetensors` (87322136 bytes)
- `scratch/p70_public1000_final_execution_r3_staging/p70_adapter/adapter_config.json` (1148 bytes)
- `scratch/p70_public1000_final_execution_r3_staging/p70_adapter/adapter_model.safetensors` (87322136 bytes)
- `scratch/p71_r2/adapter_config.json` (1148 bytes)
- `scratch/p71_r2/adapter_model.safetensors` (87322136 bytes)
- `scratch/p71_r2/p70_adapter/adapter_config.json` (1148 bytes)
- `scratch/p71_r2/p70_adapter/adapter_model.safetensors` (87322136 bytes)
- `scratch/sota_vnext_phase_generator_sft_p1e/checkpoint-2/adapter_config.json` (1148 bytes)
- `scratch/sota_vnext_phase_generator_sft_p1e/checkpoint-2/adapter_model.safetensors` (87322136 bytes)
- `t4_emergency/P2_ADAPTER_KAGGLE_UPLOAD/adapter_config.json` (1148 bytes)
- `t4_emergency/P2_ADAPTER_KAGGLE_UPLOAD/adapter_model.safetensors` (87322136 bytes)

## J. Excluded Large/Raw Competition Data (Strict Fail-Closed)

- `KAGGLE_SPRINT_R3_PREFIX40_PACKAGE_REPAIRED.bin` (6167164 bytes) — **EXCLUDED FROM GIT**
- `KAGGLE_SPRINT_R3_PREFIX40_PACKAGE_REPAIRED.zip` (6167164 bytes) — **EXCLUDED FROM GIT**
- `MASTER_ONE_SHOT_EXPERIMENT_KAGGLE.bin` (124975436 bytes) — **EXCLUDED FROM GIT**
- `MASTER_ONE_SHOT_EXPERIMENT_PREP.zip` (124922001 bytes) — **EXCLUDED FROM GIT**
- `METEOR60_COMBINED_RETRIEVAL_T4X2_OUTPUT.zip` (9578218 bytes) — **EXCLUDED FROM GIT**
- `P3_CANDIDATE_CONDITION_PRESERVING_STATIC_PREP.zip` (109626 bytes) — **EXCLUDED FROM GIT**
- `P3_CONDITION_PRESERVING_EVAL50_EXACT_SCREEN.zip` (47660 bytes) — **EXCLUDED FROM GIT**
- `P3_CONDITION_PRESERVING_EVAL50_H100_EXECUTION_PACKAGE.zip` (119759 bytes) — **EXCLUDED FROM GIT**
- `P3_CONDITION_PRESERVING_EVAL50_H100_GENERATION_OUTPUT.zip` (74881 bytes) — **EXCLUDED FROM GIT**
- `P3_E4_EVAL50_CONFIRM_AND_FAILURE_ANALYSIS.zip` (97545 bytes) — **EXCLUDED FROM GIT**
- `P3_E4_EVAL50_CONFIRM_AND_FAILURE_ANALYSIS_R1.zip` (208398 bytes) — **EXCLUDED FROM GIT**
- `P3_E4_EVAL50_CONFIRM_AND_FAILURE_ANALYSIS_R2.zip` (216207 bytes) — **EXCLUDED FROM GIT**
- `P3_E4_EVAL50_H100_EXECUTION_PACKAGE.zip` (17171 bytes) — **EXCLUDED FROM GIT**
- `P3_E4_EVAL50_H100_GENERATION_OUTPUT.zip` (62777 bytes) — **EXCLUDED FROM GIT**
- `P3_E4_EVAL50_INDEPENDENT_CONFIRM_STATIC_PREP.zip` (116772 bytes) — **EXCLUDED FROM GIT**
- `P3_FINAL_P3G2_KAGGLE_STAGE_B_DUALT4_SELFCONTAINED.bin` (568747 bytes) — **EXCLUDED FROM GIT**
- `P3_FINAL_P3G2_KAGGLE_STAGE_B_DUALT4_SELFCONTAINED.zip` (568747 bytes) — **EXCLUDED FROM GIT**
- `P3_FINAL_P3G2_LEAD_EXECUTION_HOTFIX.zip` (105003 bytes) — **EXCLUDED FROM GIT**
- `P3_FINAL_P3G2_REAL_EXECUTION_PACKAGE.zip` (95645 bytes) — **EXCLUDED FROM GIT**
- `P3_FINAL_P3G2_REAL_EXECUTION_PACKAGE_DUALT4_FINAL.zip` (95645 bytes) — **EXCLUDED FROM GIT**
