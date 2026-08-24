# API reference

All names below are importable directly from `gp3tools`.

## `add_gazepoint_aoi(data, x_col=None, y_col=None, aoi_geometry=None, output_col='aoi_current', outside_label='outside') -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Assign rectangular AOIs from a geometry table.

## `add_gazepoint_dynamic_aoi(data, aoi_data, time_col=None, x_col=None, y_col=None, aoi_time_col=None, output_col='aoi_current', tolerance=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Assign time-varying rectangular AOIs using nearest-time geometry rows.

## `add_gazepoint_polygon_aoi(data, polygons, x_col=None, y_col=None, output_col='aoi_current', outside_label='outside') -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `analyse_gazepoint_binocular_sensitivity(data, methods=('available_eye', 'linear_regression'), **kwargs) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `analyze_gazepoint_window(data, value_col=None, group_col=None, condition_col=None, **kwargs) -> 'dict[str, Any]'`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `as_gazepoint_master(data, copy: 'bool' = True) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `qc`

Coerce sample-level data to a standard Gazepoint master table.

## `audit_gazepoint_aoi_coding_matrix(data, aoi_col=None, group_cols=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `audit_gazepoint_aoi_geometry(aoi_geometry) -> 'dict[str, Any]'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `audit_gazepoint_aoi_margin_sensitivity(data, aoi_geometry, margins=(-0.02, 0, 0.02), x_col=None, y_col=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `audit_gazepoint_aoi_overlap(aoi_geometry) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `audit_gazepoint_aoi_screen_coverage(aoi_geometry, width=1.0, height=1.0) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `audit_gazepoint_aoi_window_denominators(data, success_col='success', total_col='total') -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `audit_gazepoint_binocular_reconstruction(data, observed_left=None, observed_right=None, reconstructed_left='left_pupil_reconstructed', reconstructed_right='right_pupil_reconstructed') -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `audit_gazepoint_condition_quality_imbalance(data, condition_col=None, **kwargs) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `qc`

Compatibility implementation for the frozen public API.

## `audit_gazepoint_design_balance(data, group_cols=('subject', 'condition')) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `qc`

Compatibility implementation for the frozen public API.

## `audit_gazepoint_dynamic_aoi_coverage(data, aoi_col='aoi_current') -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `audit_gazepoint_event_sync(gaze, events=None, **kwargs)`

**Status:** native  
**Module:** `face`

Compatibility implementation for the frozen public API.

## `audit_gazepoint_exclusion_flow(data, stages: 'list[str] | None' = None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `qc`

Compatibility implementation for the frozen public API.

## `audit_gazepoint_face_quality(data, confidence_col=None, threshold: 'float' = 0.8) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `face`

Compatibility implementation for the frozen public API.

## `audit_gazepoint_face_sync(gaze, face=None, **kwargs) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `face`

Compatibility implementation for the frozen public API.

## `audit_gazepoint_fixation_reliability(data, subject_col=None, duration_col='duration_ms') -> 'pd.DataFrame'`

**Status:** native  
**Module:** `events`

Compatibility implementation for the frozen public API.

## `audit_gazepoint_gaze_signal_quality(data, **kwargs) -> 'dict[str, pd.DataFrame]'`

**Status:** native  
**Module:** `qc`

Compatibility implementation for the frozen public API.

## `audit_gazepoint_master(data) -> 'dict[str, pd.DataFrame]'`

**Status:** native  
**Module:** `qc`

Create structural and signal-availability summaries for a master table.

## `audit_gazepoint_naming_consistency(names=None) -> 'dict[str, Any]'`

**Status:** native  
**Module:** `qc`

Compatibility implementation for the frozen public API.

## `audit_gazepoint_post_exclusion_balance(data, excluded_col: 'str' = 'excluded', group_cols=('condition',)) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `qc`

Compatibility implementation for the frozen public API.

## `audit_gazepoint_pupil_baseline(data, pupil_col=None, time_col=None, baseline=(-200, 0), group_cols=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `audit_gazepoint_pupil_drift(data, pupil_col=None, time_col=None, group_cols=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `audit_gazepoint_pupil_gaps(data, pupil_col=None, time_col=None, group_cols=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `audit_gazepoint_pupil_imbalance(data, pupil_col=None, condition_col=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `audit_gazepoint_pupil_overlap_risk(data, trial_duration_ms=3000, event_gap_ms=1000, trial_col=None, time_col=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `audit_gazepoint_pupil_reliability(data, pupil_col=None, subject_col=None, split_col=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `audit_gazepoint_screen_bounds(data, x_col=None, y_col=None, width: 'float' = 1.0, height: 'float' = 1.0, normalized: 'bool' = True) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `qc`

Compatibility implementation for the frozen public API.

## `audit_gazepoint_stimulus_luminance(data, luminance_col='luminance', pupil_col=None, group_cols=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `audit_gazepoint_timecourse_grid(data, time_col='time_bin', subject_col='subject', condition_col='condition', **kwargs) -> 'pd.DataFrame'`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `baseline_correct_gazepoint_pupil(data, pupil_col=None, time_col=None, baseline=(-200.0, 0.0), group_cols=None, output_col=None, mode='subtract') -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `benchmark_gazepoint_event_detectors(data, repeats: 'int' = 3, **kwargs) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `events`

Compatibility implementation for the frozen public API.

## `benchmark_gazepoint_export_performance(data, repeats=3, **kwargs) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `reporting`

Compatibility implementation for the frozen public API.

## `bootstrap_gazepoint_scanpath_clusters(data, n_boot=100, random_state=123, **kwargs) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `bootstrap_gazepoint_timecourse(data, value_col='value', time_col='time_bin', subject_col='subject', n_boot: 'int' = 500, random_state: 'int' = 123, **kwargs) -> 'pd.DataFrame'`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `check_gazepoint_bayesian_readiness(data, **kwargs) -> 'pd.DataFrame'`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `check_gazepoint_file_pairs(folder: 'str | Path') -> 'pd.DataFrame'`

**Status:** native  
**Module:** `qc`

Compatibility implementation for the frozen public API.

## `check_gazepoint_model_convergence(model) -> 'pd.DataFrame'`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `check_gazepoint_model_overdispersion(model) -> 'pd.DataFrame'`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `check_gazepoint_model_singularity(model, tolerance: 'float' = 1e-08) -> 'pd.DataFrame'`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `check_gazepoint_performance_regression(current, baseline, tolerance=0.2, metric='elapsed_seconds') -> 'pd.DataFrame'`

**Status:** native  
**Module:** `reporting`

Compatibility implementation for the frozen public API.

## `check_gazepoint_real_data_readiness(data) -> 'dict[str, Any]'`

**Status:** native  
**Module:** `qc`

Compatibility implementation for the frozen public API.

## `check_sampling_rate(data, time_col: 'str | None' = None, group_cols=None, expected_hz: 'float' = 60.0, tolerance_hz: 'float' = 5.0) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `qc`

Estimate effective sampling rate from timestamp differences.

## `classify_gazepoint_events_hmm(data, x_col=None, y_col=None, time_col=None, **kwargs) -> 'pd.DataFrame'`

**Status:** native-adapted  
**Module:** `events`

Classify samples into fixation/saccade states using a robust velocity mixture.

## `classify_gazepoint_export(path: 'str | Path') -> 'str'`

**Status:** native  
**Module:** `io`

Classify a Gazepoint export by filename and header content.

## `clean_gazepoint_by_trackloss(data, validity_col: 'str | None' = None, drop: 'bool' = True) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `qc`

Compatibility implementation for the frozen public API.

## `cluster_gazepoint_scanpaths(data, aoi_col=None, group_cols=None, time_col=None, n_clusters=3) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `collect_gazepoint_qc_summaries(data) -> 'dict[str, Any]'`

**Status:** native  
**Module:** `qc`

Compatibility implementation for the frozen public API.

## `combine_gazepoint_eyes(data, left_col=None, right_col=None, output_col='pupil_combined', policy='available_eye') -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `compare_gazepoint_event_detectors(data, **kwargs) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `events`

Compatibility implementation for the frozen public API.

## `compare_gazepoint_nested_models(models, labels=None) -> 'pd.DataFrame'`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `compute_gazepoint_aoi_entropy(data=None, sequence=None, aoi_col=None, group_cols=None, time_col=None, normalize=True) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `compute_gazepoint_aoi_sequence_metrics(data=None, sequence=None, **kwargs) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `compute_gazepoint_aoi_transition_matrix(data=None, sequence=None, aoi_col=None, group_cols=None, time_col=None, normalize=False, include_self=True)`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `compute_gazepoint_saccade_metrics(data, x_col=None, y_col=None, time_col=None, event_col='event_state', group_cols=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `events`

Summarise saccade amplitude, duration, and peak velocity.

## `compute_gazepoint_scanpath_geometry(data, x_col=None, y_col=None, time_col=None, group_cols=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `compute_gazepoint_scanpath_similarity(path_a, path_b, method='sequence') -> 'float'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `compute_gazepoint_sequence_complexity(data=None, sequence=None, aoi_col=None, group_cols=None, time_col=None, include_missing=False, missing_label='missing', collapse_repeats=False) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `compute_gazepoint_sequence_distance(sequence_a, sequence_b, method='levenshtein', normalize=True) -> 'float'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `compute_gazepoint_sequence_recurrence(sequence, lag=1) -> 'dict[str, float]'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `compute_gazepoint_time_varying_transition_matrix(data, aoi_col=None, time_col=None, bin_width=500, group_cols=None, normalize=False) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `compute_gazepoint_transition_network_metrics(matrix) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `compute_transition_matrix(sequence, normalize=False) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `construct_gazepoint_combined_pupil(data, left_col=None, right_col=None, output_col='pupil_combined', policy='available_eye') -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `create_gazepoint_analysis_decision_audit(**decisions) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `reporting`

Compatibility implementation for the frozen public API.

## `create_gazepoint_bayesian_sap(**kwargs) -> 'dict[str, Any]'`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `create_gazepoint_brms_template(formula=None, family='gaussian', priors=None, **kwargs) -> 'dict[str, Any]'`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `create_gazepoint_cross_package_report(result, output_file='cross_package_report.html', **kwargs)`

**Status:** native  
**Module:** `reporting`

Compatibility implementation for the frozen public API.

## `create_gazepoint_event_review_template(data, path: 'str | Path | None' = None, **kwargs) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `events`

Compatibility implementation for the frozen public API.

## `create_gazepoint_face_reporting_checklist(data=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `face`

Compatibility implementation for the frozen public API.

## `create_gazepoint_hddm_fit_script(path=None, model_name='HDDM', **kwargs) -> 'str'`

**Status:** native-adapter  
**Module:** `interop`

Compatibility implementation for the frozen public API.

## `create_gazepoint_markovchain_object(data=None, sequence=None, **kwargs)`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `create_gazepoint_master(data, **kwargs) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `qc`

Create a standardized sample-level master table.

## `create_gazepoint_preprocessing_multiverse(**grids) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `create_gazepoint_preprocessing_registry(blink_padding_pre_ms=100, blink_padding_post_ms=100, max_interpolation_gap_ms=150, smoothing_window_ms=50, baseline_start_ms=-200, baseline_end_ms=0, pupil_physiological_min=1, pupil_physiological_max=9, pupil_speed_mad_k=6, binocular_mad_k=6, baseline_missing_prop_threshold=0.3, baseline_interpolated_prop_threshold=0.3, baseline_artifact_prop_threshold=0.3, overlap_trial_duration_ms=3000, overlap_event_gap_ms=1000) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `create_gazepoint_report(results, output_file='gazepoint_report.html', title='gp3tools analysis report', metadata=None, **kwargs) -> 'Path'`

**Status:** native  
**Module:** `reporting`

Compatibility implementation for the frozen public API.

## `create_gazepoint_reporting_checklist(data=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `reporting`

Compatibility implementation for the frozen public API.

## `detect_gazepoint_blinks(data, pupil_col=None, time_col=None, min_duration_ms: 'float' = 50.0, max_duration_ms: 'float' = 800.0, output_col='blink') -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `detect_gazepoint_fixations_ivt(data, **kwargs) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `events`

Alias for I-VT fixation detection.

## `detect_gazepoint_fixations_velocity(data, x_col=None, y_col=None, time_col=None, velocity_threshold: 'float' = 0.08, min_duration_ms: 'float' = 100.0, group_cols=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `events`

Detect fixations with a transparent velocity-threshold algorithm.

## `diagnose_gazepoint_binocular_pupil(data, left_col=None, right_col=None, group_cols=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `diagnose_gazepoint_cluster_design(data, subject_col='subject', condition_col='condition', time_col='time_bin', **kwargs) -> 'pd.DataFrame'`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `diagnose_gazepoint_gamm(model) -> 'dict[str, pd.DataFrame]'`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `diagnose_gazepoint_glmm(model) -> 'dict[str, pd.DataFrame]'`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `downsample_gazepoint_pupil(data, time_col=None, pupil_col=None, target_hz: 'float' = 30.0, group_cols=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `estimate_gazepoint_cluster_offset(result, **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `estimate_gazepoint_cluster_onset(result, **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `estimate_gazepoint_divergence_point(data, value_col='value', time_col='time_bin', condition_col='condition', min_run: 'int' = 3, **kwargs) -> 'pd.DataFrame'`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `export_gazepoint_cluster_results(result, output_dir='cluster_results', prefix='cluster')`

**Status:** native  
**Module:** `reporting`

Compatibility implementation for the frozen public API.

## `export_gazepoint_heatmap_png(data, path='gazepoint_heatmap.png', dpi=150, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `export_gazepoint_master_audit(data, output_dir, prefix='master_audit', **kwargs)`

**Status:** native  
**Module:** `reporting`

Compatibility implementation for the frozen public API.

## `export_gazepoint_mne_cluster_input(data, path='mne_cluster_input.csv', **kwargs)`

**Status:** native-adapter  
**Module:** `interop`

Compatibility implementation for the frozen public API.

## `export_gazepoint_model_tables(models, output_dir, prefix='model', **kwargs)`

**Status:** native  
**Module:** `reporting`

Compatibility implementation for the frozen public API.

## `export_gazepoint_permuco_cluster_input(data, path='permuco_cluster_input.csv', **kwargs)`

**Status:** native-adapter  
**Module:** `interop`

Compatibility implementation for the frozen public API.

## `export_gazepoint_permutes_cluster_input(data, path='permutes_cluster_input.csv', **kwargs)`

**Status:** native-adapter  
**Module:** `interop`

Compatibility implementation for the frozen public API.

## `export_gazepoint_tables(tables, output_dir, prefix='gazepoint', index=False, **kwargs) -> 'dict[str, str]'`

**Status:** native  
**Module:** `reporting`

Compatibility implementation for the frozen public API.

## `export_gazepoint_to_bids(data, output_dir, subject_col=None, task='gazepoint', **kwargs) -> 'dict'`

**Status:** native-adapter  
**Module:** `interop`

Compatibility implementation for the frozen public API.

## `extract_gazepoint_representative_scanpaths(clustered) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `filter_gazepoint_cnn_uncertainty(data, uncertainty_col='uncertainty', threshold=0.5, keep_flag=True, **kwargs)`

**Status:** native  
**Module:** `misc`

Compatibility implementation for the frozen public API.

## `fit_gazepoint_aoi_brms(data, formula=None, **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `fit_gazepoint_aoi_gamm(data, **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `fit_gazepoint_aoi_model_sensitivity(data, formulas=None, **kwargs) -> 'dict[str, Any]'`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `fit_gazepoint_aoi_window_glmm(data, formula=None, **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `fit_gazepoint_binocular_calibration(data, left_col=None, right_col=None) -> 'dict[str, Any]'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `fit_gazepoint_brms_model(data, formula=None, family='gaussian', **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `fit_gazepoint_face_window_lmm(data, formula=None, subject_col=None, **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `fit_gazepoint_gca(data, formula=None, outcome_col=None, subject_col=None, order: 'int' = 2, **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `fit_gazepoint_multimodal_response_model(data, formula=None, subject_col=None, **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `fit_gazepoint_pupil_gamm(data, **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `fit_gazepoint_pupil_pfe_gamm(data, **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `fit_gazepoint_pupil_window_lmm(data, formula=None, subject_col=None, **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `fit_gazepoint_pupil_window_sensitivity(data, formulas=None, **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `fit_gazepoint_transition_count_nb_sensitivity(data, formula=None, **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `flag_gazepoint_pupil(data, pupil_col=None, physiological_min: 'float' = 1.0, physiological_max: 'float' = 9.0, output_col='pupil_flag') -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `flag_gazepoint_pupil_artifacts(data, pupil_col=None, time_col=None, physiological_min=1.0, physiological_max=9.0, pupil_speed_mad_k: 'float' = 6.0, blink_padding_pre_ms: 'float' = 100.0, blink_padding_post_ms: 'float' = 100.0, output_col='pupil_artifact') -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `flag_gazepoint_pupil_hampel(data, pupil_col=None, window: 'int' = 7, n_sigma: 'float' = 3.0, output_col='pupil_hampel_flag') -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `flag_gazepoint_sequence_anomalies(data=None, sequence=None, z_threshold=3.0, **kwargs) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `flag_tracking_quality(data, min_usable_prop: 'float' = 0.8, **kwargs) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `qc`

Compatibility implementation for the frozen public API.

## `gp3tools_naming_policy() -> 'dict[str, Any]'`

**Status:** native  
**Module:** `qc`

Compatibility implementation for the frozen public API.

## `gp3tools_performance_limits() -> 'pd.DataFrame'`

**Status:** native  
**Module:** `reporting`

Compatibility implementation for the frozen public API.

## `harmonize_gazepoint_screen_coordinates(data, x_col=None, y_col=None, width: 'float | None' = None, height: 'float | None' = None, output_x: 'str' = 'x_norm', output_y: 'str' = 'y_norm') -> 'pd.DataFrame'`

**Status:** native  
**Module:** `qc`

Compatibility implementation for the frozen public API.

## `impute_gazepoint_pupil_gp(data, pupil_col=None, time_col=None, output_col=None, max_points: 'int' = 2000, random_state: 'int' = 123) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `inspect_gazepoint_columns(data) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `io`

Return a compact audit of names, dtypes, missingness, and uniqueness.

## `interpolate_gazepoint_blinks(data, pupil_col=None, blink_col='blink', **kwargs) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `interpolate_gazepoint_pupil(data, pupil_col=None, output_col=None, method='linear', max_gap_ms: 'float | None' = 150.0, time_col=None, group_cols=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `interpolate_gazepoint_pupil_pchip(data, **kwargs) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `launch_gazepoint_qc_dashboard(data=None, **kwargs)`

**Status:** native-adapted  
**Module:** `reporting`

Create a lightweight Shiny-for-Python QC app when the optional dependency exists.

## `mean_gazepoint_pupil(data, left_col=None, right_col=None, output_col='pupil_mean', require_both=False) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_aoi_gamm(model_or_data, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_aoi_timeline(data, aoi_col=None, time_col=None, ax=None, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_aoi_transition_matrix(data, ax=None, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_aoi_verification(data, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_binocular_diagnostics(data, left_col=None, right_col=None, ax=None, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_cluster_null_distribution(result, ax=None, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_cluster_permutation(result, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_cluster_results(result, ax=None, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_event_detector_agreement(data, ax=None, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_event_detector_benchmark(data, ax=None, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_face_quality(data, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_gca(model_or_data, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_heatmap(data, x_col=None, y_col=None, bins=40, ax=None, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_heatmap_overlay(data, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_missingness_profile(data, ax=None, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_model_predictions(model, data=None, ax=None, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_model_residuals(model, ax=None, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_multiverse_results(data, ax=None, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_phase_timeline(data, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_pupil_preprocessing(data, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_pupil_status(data, status_col='pupil_flag', ax=None, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_pupil_timecourse(data, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_qc_overview(data, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_scanpath(data, x_col=None, y_col=None, ax=None, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_scanpath_cluster_stability(data, ax=None, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_scanpath_clusters(data, cluster_col='cluster', **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_scanpaths(data, group_col=None, ax=None, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_stimulus_layout_qc(data, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_time_series(data, x_col=None, y_col=None, group_col=None, ax=None, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_gazepoint_time_varying_effect(data, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_sampling_rate(data, ax=None, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_tracking_quality(data, ax=None, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `plot_transition_heatmap(data, **kwargs)`

**Status:** native  
**Module:** `plotting`

Compatibility implementation for the frozen public API.

## `prepare_gazepoint_aoi_gamm_data(data, **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `prepare_gazepoint_aoi_glmm_data(data, aoi_col=None, target_aoi=None, **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `prepare_gazepoint_aoi_sequences(data, aoi_col=None, group_cols=None, time_col=None, collapse_repeats=True, **kwargs) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `prepare_gazepoint_cluster_data(data, **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `prepare_gazepoint_eyetools_data(data, **kwargs)`

**Status:** native-adapter  
**Module:** `interop`

Compatibility implementation for the frozen public API.

## `prepare_gazepoint_eyetrackingr_data(data, participant_col=None, trial_col=None, time_col=None, x_col=None, y_col=None, **kwargs)`

**Status:** native-adapter  
**Module:** `interop`

Compatibility implementation for the frozen public API.

## `prepare_gazepoint_fixation_aligned_data(data, fixation_time_col=None, sample_time_col=None, window=(-0.5, 1.5), **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `prepare_gazepoint_gazer_data(data, **kwargs)`

**Status:** native-adapter  
**Module:** `interop`

Compatibility implementation for the frozen public API.

## `prepare_gazepoint_gca_data(data, time_col=None, outcome_col=None, group_cols=None, order: 'int' = 2, center_time: 'bool' = True, **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `prepare_gazepoint_gpbiometrics_bridge(data, participant_col=None, time_col=None, **kwargs)`

**Status:** native-adapter  
**Module:** `interop`

Compatibility implementation for the frozen public API.

## `prepare_gazepoint_hddm_export(data, response_col=None, rt_col=None, subject_col=None, condition_col=None, **kwargs)`

**Status:** native-adapter  
**Module:** `interop`

Compatibility implementation for the frozen public API.

## `prepare_gazepoint_heatmap_data(data, x_col=None, y_col=None, bins=40, **kwargs) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `misc`

Compatibility implementation for the frozen public API.

## `prepare_gazepoint_hmm_data(data, **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `prepare_gazepoint_multimodal_data(gaze, face=None, **kwargs)`

**Status:** native  
**Module:** `face`

Compatibility implementation for the frozen public API.

## `prepare_gazepoint_pupil_gamm_data(data, pupil_col=None, time_col=None, **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `prepare_gazepoint_pupil_window_model_data(data, pupil_col=None, time_col=None, windows=None, group_cols=None, **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `prepare_gazepoint_pupillometryr_data(data, participant_col=None, time_col=None, pupil_col=None, **kwargs)`

**Status:** native-adapter  
**Module:** `interop`

Compatibility implementation for the frozen public API.

## `prepare_gazepoint_semimarkov_data(data, **kwargs)`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `prepare_gazepoint_timecourse_test_data(data, time_col=None, value_col=None, subject_col=None, condition_col=None, **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `prepare_gazepoint_traminer_data(data, **kwargs)`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `preprocess_gazepoint_signals(data, pupil_col=None, physiological_min=1.0, physiological_max=9.0, interpolate=True, smooth=True, baseline=None, time_col=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `read_gazepoint(path: 'str | Path', standardise_names: 'bool' = True, drop_empty_cols: 'bool' = True, **read_csv_kwargs) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `io`

Read a Gazepoint all-gaze or fixation CSV export.

## `read_gazepoint_face_export(path: 'str | Path', **kwargs) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `io`

Read an external facial-behaviour CSV/TSV export.

## `read_gazepoint_folder(folder: 'str | Path', pattern: 'str' = '\\.csv$', source_col: 'str' = 'USER_FILE', recursive: 'bool' = False, **read_kwargs) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `io`

Read matching Gazepoint exports in a folder and row-bind them.

## `read_gazepoint_summary(path: 'str | Path') -> 'dict[str, Any]'`

**Status:** native  
**Module:** `io`

Parse an official Gazepoint Analysis summary export conservatively.

## `recalibrate_gazepoint_gaze(data, x_col=None, y_col=None, target_x=0.5, target_y=0.5, method='offset', output_x='x_recalibrated', output_y='y_recalibrated', **kwargs)`

**Status:** native  
**Module:** `misc`

Compatibility implementation for the frozen public API.

## `recommend_gazepoint_exclusions(data, participant_col=None, trial_col=None, validity_col=None, x_col=None, y_col=None, pupil_col=None, artifact_col=None, min_trial_samples: 'int' = 20, max_trial_missing_prop: 'float' = 0.5, max_trial_artifact_prop: 'float' = 0.5, min_participant_trials: 'int' = 1, min_participant_valid_trials: 'int' = 1, max_participant_missing_prop: 'float' = 0.5, max_participant_artifact_prop: 'float' = 0.5, require_both_gaze_coordinates: 'bool' = True, name: 'str' = 'gazepoint_exclusions', **kwargs) -> 'dict[str, Any]'`

**Status:** native  
**Module:** `qc`

Recommend exclusions without removing rows.

## `recommend_gazepoint_model_family(data, outcome_col=None, **kwargs) -> 'pd.DataFrame'`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `reconstruct_gazepoint_binocular_pupil(data, left_col=None, right_col=None, method='linear_regression', output_left='left_pupil_reconstructed', output_right='right_pupil_reconstructed', combined_col='pupil_combined') -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `regress_gazepoint_pupils(data, left_col=None, right_col=None, direction='right_from_left') -> 'dict[str, Any]'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `report_gazepoint_cluster_permutation(result) -> 'str'`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `report_gazepoint_face_qc(data) -> 'str'`

**Status:** native  
**Module:** `reporting`

Compatibility implementation for the frozen public API.

## `report_gazepoint_missingness(data) -> 'str'`

**Status:** native  
**Module:** `reporting`

Compatibility implementation for the frozen public API.

## `report_gazepoint_multiverse(data) -> 'str'`

**Status:** native  
**Module:** `reporting`

Compatibility implementation for the frozen public API.

## `report_gazepoint_phase_coverage(data) -> 'str'`

**Status:** native  
**Module:** `reporting`

Compatibility implementation for the frozen public API.

## `report_gazepoint_qc_overview(data) -> 'str'`

**Status:** native  
**Module:** `reporting`

Compatibility implementation for the frozen public API.

## `run_gazepoint_aoi_multiverse(data, specifications=None, **kwargs) -> 'pd.DataFrame'`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `run_gazepoint_cluster_permutation(data, value_col='value', time_col='time_bin', condition_col='condition', subject_col='subject', n_permutations: 'int' = 1000, alpha: 'float' = 0.05, random_state: 'int' = 123, **kwargs) -> 'dict[str, Any]'`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `run_gazepoint_cluster_permutation_anova(data, **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `run_gazepoint_cluster_permutation_covariate_adjusted(data, **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `run_gazepoint_cluster_permutation_lmer(data, **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `run_gazepoint_cluster_permutation_parallel(data, **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `run_gazepoint_cluster_threshold_sensitivity(data, thresholds=(1.0, 1.5, 2.0), **kwargs) -> 'pd.DataFrame'`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `run_gazepoint_eyetools_fixation_detection(data, **kwargs)`

**Status:** native-adapter  
**Module:** `interop`

Compatibility implementation for the frozen public API.

## `run_gazepoint_gazer_crosscheck(data, **kwargs)`

**Status:** native-adapter  
**Module:** `interop`

Compatibility implementation for the frozen public API.

## `run_gazepoint_gpbiometrics_workflow(data, **kwargs)`

**Status:** native-adapter  
**Module:** `interop`

Compatibility implementation for the frozen public API.

## `run_gazepoint_model_leave_one_out(data, fit_function=<function fit_gazepoint_pupil_window_lmm at 0x7fe8df2e4680>, subject_col=None, **kwargs) -> 'pd.DataFrame'`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `run_gazepoint_multidimensional_cluster_permutation(data, **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `run_gazepoint_pupil_multiverse(data, registry=None, **kwargs) -> 'pd.DataFrame'`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `run_gazepoint_tfce(data, **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `run_gazepoint_workflow(data=None, export_dir=None, output_dir=None, pattern='\\.csv$', create_report=True, save_plots=False, **kwargs) -> 'dict[str, Any]'`

**Status:** native  
**Module:** `reporting`

Compatibility implementation for the frozen public API.

## `save_gazepoint_plots(plots, output_dir, prefix='plot', dpi=150, **kwargs)`

**Status:** native  
**Module:** `reporting`

Compatibility implementation for the frozen public API.

## `segment_gazepoint_task_phases(data, time_col=None, boundaries=None, labels=None, output_col: 'str' = 'phase') -> 'pd.DataFrame'`

**Status:** native  
**Module:** `qc`

Compatibility implementation for the frozen public API.

## `select_gazepoint_adaptive_trial(data, score_col=None, strategy='highest_uncertainty', **kwargs)`

**Status:** native  
**Module:** `misc`

Compatibility implementation for the frozen public API.

## `select_gazepoint_scanpath_clusters(data, max_clusters=6, **kwargs) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `simulate_gazepoint_cluster_timecourse_data(n_subjects: 'int' = 16, n_time: 'int' = 80, effect_window=(30, 50), effect_size: 'float' = 0.35, random_state: 'int' = 123) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `simulation`

Compatibility implementation for the frozen public API.

## `simulate_gazepoint_data(n_subjects: 'int' = 6, n_trials: 'int' = 8, samples_per_trial: 'int' = 120, sampling_rate: 'float' = 60.0, random_state: 'int' = 123) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `simulation`

Compatibility implementation for the frozen public API.

## `simulate_gazepoint_fixations(n_fixations: 'int' = 30, samples_per_fixation: 'int' = 8, random_state: 'int' = 123) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `events`

Compatibility implementation for the frozen public API.

## `simulate_gazepoint_pupil_data(n_subjects: 'int' = 12, n_trials: 'int' = 12, samples_per_trial: 'int' = 180, random_state: 'int' = 123, **kwargs) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `simulation`

Compatibility implementation for the frozen public API.

## `smooth_gazepoint_coordinate(data, column=None, output_col=None, window: 'int' = 5, method='moving_average') -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `smooth_gazepoint_pupil(data, pupil_col=None, output_col=None, window: 'int' = 5, method='moving_average') -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `standardise_gazepoint_names(x)`

**Status:** native  
**Module:** `io`

Standardise Gazepoint column names or a sequence of names.

## `standardize_gazepoint_face_columns(data) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `face`

Compatibility implementation for the frozen public API.

## `stress_test_gazepoint_binocular_reconstruction(data, fractions=(0.05, 0.1, 0.2, 0.3), **kwargs) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `summarise_aoi_samples(data, aoi_col=None, group_cols=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `summarise_fixations(data, fixation_col='fixation', fixation_id_col='fixation_id', x_col=None, y_col=None, time_col=None, group_cols=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `events`

Collapse fixation-classified samples to one row per fixation.

## `summarise_gazepoint_aoi(data, aoi_col=None, group_cols=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `summarise_gazepoint_aoi_entries(data, aoi_col=None, group_cols=None, time_col=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `summarise_gazepoint_aoi_transitions(data, aoi_col=None, group_cols=None, time_col=None, include_self=False) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `summarise_gazepoint_aoi_trial_features(data, aoi_col=None, trial_col=None, group_cols=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `summarise_gazepoint_aoi_windows(data, aoi_col=None, time_col=None, windows=None, group_cols=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `summarise_gazepoint_binocular_reporting(data, **kwargs) -> 'dict[str, pd.DataFrame]'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `summarise_gazepoint_clusters(result) -> 'pd.DataFrame'`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `summarise_gazepoint_coordinate_coverage(data, x_col=None, y_col=None, group_cols=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `qc`

Compatibility implementation for the frozen public API.

## `summarise_gazepoint_emmeans(data, factor=None, outcome=None, group_cols=None, **kwargs) -> 'pd.DataFrame'`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `summarise_gazepoint_event_detector_agreement(data, **kwargs) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `events`

Compatibility implementation for the frozen public API.

## `summarise_gazepoint_event_detector_benchmark(data) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `events`

Compatibility implementation for the frozen public API.

## `summarise_gazepoint_face_quality(data, **kwargs)`

**Status:** native  
**Module:** `face`

Compatibility implementation for the frozen public API.

## `summarise_gazepoint_face_reactivity(data, **kwargs)`

**Status:** native  
**Module:** `face`

Compatibility implementation for the frozen public API.

## `summarise_gazepoint_face_windows(data, **kwargs)`

**Status:** native  
**Module:** `face`

Compatibility implementation for the frozen public API.

## `summarise_gazepoint_fixation_trials(data, trial_col=None, subject_col=None, duration_col='duration_ms') -> 'pd.DataFrame'`

**Status:** native  
**Module:** `events`

Compatibility implementation for the frozen public API.

## `summarise_gazepoint_fixed_effects(model, **kwargs)`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `summarise_gazepoint_markovchain(data=None, sequence=None, **kwargs) -> 'dict[str, Any]'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `summarise_gazepoint_missingness(data, group_cols=None, columns=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `qc`

Compatibility implementation for the frozen public API.

## `summarise_gazepoint_multiverse_results(data) -> 'pd.DataFrame'`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `summarise_gazepoint_phase_coverage(data, phase_col='phase', group_cols=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `qc`

Compatibility implementation for the frozen public API.

## `summarise_gazepoint_pupil(data, pupil_col=None, group_cols=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `summarise_gazepoint_pupil_response_features(data, **kwargs)`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `summarise_gazepoint_pupil_trial_features(data, pupil_col=None, trial_col=None, group_cols=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `summarise_gazepoint_pupil_windows(data, pupil_col=None, time_col=None, windows=None, group_cols=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `summarise_gazepoint_qc_status(qc) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `qc`

Compatibility implementation for the frozen public API.

## `summarise_gazepoint_scanpath_cluster_stability(data) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `summarise_gazepoint_semimarkov(data=None, sequence=None, **kwargs)`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `summarise_gazepoint_time_clusters(result) -> 'pd.DataFrame'`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `summarise_gazepoint_workflow(result) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `reporting`

Compatibility implementation for the frozen public API.

## `summarise_tracking_quality(data, validity_col: 'str | None' = None, group_cols=None, x_col: 'str | None' = None, y_col: 'str | None' = None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `qc`

Summarise usable gaze samples and tracking loss.

## `summarize_gazepoint_coordinate_coverage(data, x_col=None, y_col=None, group_cols=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `qc`

Compatibility implementation for the frozen public API.

## `summarize_gazepoint_face_quality(data, **kwargs)`

**Status:** native  
**Module:** `face`

Compatibility implementation for the frozen public API.

## `summarize_gazepoint_face_reactivity(data, baseline=None, group_cols=None, value_cols=None, **kwargs) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `face`

Compatibility implementation for the frozen public API.

## `summarize_gazepoint_face_windows(data, group_cols=None, value_cols=None, **kwargs) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `face`

Compatibility implementation for the frozen public API.

## `summarize_gazepoint_missingness(data, group_cols=None, columns=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `qc`

Compatibility implementation for the frozen public API.

## `summarize_gazepoint_phase_coverage(data, phase_col='phase', group_cols=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `qc`

Compatibility implementation for the frozen public API.

## `summarize_gazepoint_pupil_response_features(data, **kwargs)`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `summarize_gazepoint_qc_status(qc) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `qc`

Compatibility implementation for the frozen public API.

## `summarize_gazepoint_time_clusters(result) -> 'pd.DataFrame'`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `sync_gazepoint_face_data(gaze, face, gaze_time_col=None, face_time_col=None, tolerance_ms: 'float' = 50.0, by=None) -> 'pd.DataFrame'`

**Status:** native  
**Module:** `face`

Compatibility implementation for the frozen public API.

## `tidy_gazepoint_model_summary(model) -> 'pd.DataFrame'`

**Status:** native-adapted  
**Module:** `stats`

Compatibility implementation for the frozen public API.

## `transform_gazepoint_aoi_empirical_logit(data, success_col='success', total_col='total', adjustment=0.5, output_col='empirical_logit') -> 'pd.DataFrame'`

**Status:** native  
**Module:** `aoi`

Compatibility implementation for the frozen public API.

## `validate_gazepoint_binocular_reconstruction(data, left_col=None, right_col=None, fraction: 'float' = 0.1, random_state: 'int' = 123, method='linear_regression') -> 'pd.DataFrame'`

**Status:** native  
**Module:** `pupil`

Compatibility implementation for the frozen public API.

## `validate_gazepoint_master(data, required: 'tuple[str, ...]' = ('subject', 'time')) -> 'dict[str, Any]'`

**Status:** native  
**Module:** `qc`

Validate a master table and return an explicit pass/fail gate.

## `write_gazepoint_naming_audit(path, names=None) -> 'Path'`

**Status:** native  
**Module:** `qc`

Compatibility implementation for the frozen public API.

## `write_gazepoint_outputs(results, output_dir, prefix='gazepoint', **kwargs)`

**Status:** native  
**Module:** `reporting`

Compatibility implementation for the frozen public API.

## `write_gazepoint_performance_benchmark(data, path='performance_benchmark.csv') -> 'Path'`

**Status:** native  
**Module:** `reporting`

Compatibility implementation for the frozen public API.
