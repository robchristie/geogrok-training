export type RunSummary = {
	run_id: string;
	run_root: string;
	summary_path: string;
	run_kind: string;
	teacher_model: string | null;
	student_model: string | null;
	metrics: Record<string, number>;
};

export type ChipRecord = {
	chip_id: string;
	asset_id: string;
	capture_id: string;
	scene_id: string;
	split: string;
	city: string;
	modality: string;
	sensor: string;
	local_path: string;
	width: number;
	height: number;
	x0: number;
	y0: number;
};

export type ReviewArtifactRecord = {
	chip_id: string;
	artifact_kind: 'j2c' | 'png';
	codec_profile: string;
	media_type: string;
	content_path: string;
	content_url: string;
	fallback_png_url: string;
	width: number;
	height: number;
	channels: number;
	bits_per_sample: number;
	is_signed: boolean;
	generated_at: string;
	file_size_bytes: number;
	source_path: string;
	source_window: {
		x0: number;
		y0: number;
		width: number;
		height: number;
	};
};

export type PairReviewArtifactRecord = {
	pair_key: string;
	query: ReviewArtifactRecord;
	candidate: ReviewArtifactRecord;
	fallback_png_url: string;
};

export type PairRecord = {
	pair_key: string;
	query_chip_id: string;
	candidate_chip_id: string;
	pair_label: string;
	pair_group: string;
	query_split: string;
	candidate_split: string;
	city: string;
	modality: string;
	overlap_fraction: number;
	overlap_iou: number;
	time_delta_seconds: number;
	center_distance_m: number;
	annotation?: PairAnnotation | null;
};

export type ChipFacets = {
	cities: string[];
	splits: string[];
	modalities: string[];
	sensors: string[];
};

export type RunSelection = {
	selection_id: string;
	label: string;
	query_splits: string[];
	gallery_splits: string[];
};

export type PairAnnotation = {
	pair_key: string;
	query_chip_id: string;
	candidate_chip_id: string;
	status: string;
	bookmarked: boolean;
	tags: string[];
	note: string;
	created_at: string;
	updated_at: string;
};

export type AnnotationStatus =
	| 'all'
	| 'unreviewed'
	| 'reviewed'
	| 'confirmed'
	| 'incorrect_label'
	| 'interesting'
	| 'needs_followup';

export type PairFacets = {
	pair_labels: string[];
	cities: string[];
	splits: string[];
	annotation_statuses: AnnotationStatus[];
};

export type FailureRecord = PairRecord & {
	rank: number;
	similarity: number;
};

export type DisagreementRecord = PairRecord & {
	teacher_rank: number;
	student_rank: number;
	teacher_similarity: number;
	student_similarity: number;
	teacher_rank_advantage: number;
	student_rank_advantage: number;
	teacher_similarity_advantage: number;
	student_similarity_advantage: number;
};

export type RunDetail = {
	run_id: string;
	run_root: string;
	run_kind: string;
	teacher_model: string | null;
	student_model: string | null;
	metrics: Record<string, number>;
	available_selections: RunSelection[];
	default_selection_id: string | null;
};

export type RunFailures = {
	run_id: string;
	run_kind: string;
	annotation_status?: string;
	bookmarked_only?: boolean;
	selection: {
		selection_id: string;
		label: string;
		top_k: number;
		query_splits: string[];
		gallery_splits: string[];
		pairs_path: string;
	};
	queue_counts: {
		false_negatives: number;
		false_positives: number;
	};
	queue_totals?: {
		false_negatives: number;
		false_positives: number;
	};
	false_negatives: FailureRecord[];
	false_positives: FailureRecord[];
};

export type RunDisagreements = {
	run_id: string;
	run_kind: string;
	pairs_path: string;
	annotation_status?: string;
	bookmarked_only?: boolean;
	teacher_selection: RunSelection;
	student_selection: RunSelection;
	queue_counts: {
		teacher_ahead_positives: number;
		student_ahead_positives: number;
		student_confused_negatives: number;
		teacher_confused_negatives: number;
	};
	queue_totals?: {
		teacher_ahead_positives: number;
		student_ahead_positives: number;
		student_confused_negatives: number;
		teacher_confused_negatives: number;
	};
	teacher_ahead_positives: DisagreementRecord[];
	student_ahead_positives: DisagreementRecord[];
	student_confused_negatives: DisagreementRecord[];
	teacher_confused_negatives: DisagreementRecord[];
};
