<script lang="ts">
import PairAnnotation from '$lib/components/PairAnnotation.svelte';
import ReviewPairImage from '$lib/components/ReviewPairImage.svelte';
import type { PageData } from './$types';

let { data }: { data: PageData } = $props();
</script>

<section class="page">
	{#if data.error || !data.run || !data.failures}
		<header>
			<p class="eyebrow">Model behavior</p>
			<h2>Run Detail</h2>
		</header>
		<div class="empty-state">
			<p>Run detail unavailable</p>
			<span>{data.error ?? 'No run payload returned.'}</span>
		</div>
	{:else}
		<header class="hero">
			<div>
				<p class="eyebrow">Failure review</p>
				<h2>{data.run.run_id}</h2>
				<p class="lede">
					{data.run.run_kind} with selection <code>{data.failures.selection.selection_id}</code>. The
					queues below are derived from the saved embeddings and explicit pair labels, not a separate
					UI-only heuristic.
				</p>
			</div>

			<div class="selection-strip">
				{#each data.run.available_selections as selection}
					<a
						class:selected={selection.selection_id === data.failures.selection.selection_id}
						href={`/runs/${data.run.run_id}?selection=${selection.selection_id}${data.filters.annotationStatus ? `&annotation_status=${data.filters.annotationStatus}` : ''}`}
					>
						{selection.label}
					</a>
				{/each}
			</div>
		</header>

		<form class="filter-bar" method="GET">
			<input
				type="hidden"
				name="selection"
				value={data.filters.selection ?? data.failures.selection.selection_id}
			/>
			<label>
				<span>Review state</span>
				<select name="annotation_status">
					<option value="">All examples</option>
					<option value="unreviewed" selected={data.filters.annotationStatus === 'unreviewed'}>
						Unreviewed
					</option>
					<option value="reviewed" selected={data.filters.annotationStatus === 'reviewed'}>
						Reviewed
					</option>
					<option value="confirmed" selected={data.filters.annotationStatus === 'confirmed'}>
						Confirmed
					</option>
					<option
						value="incorrect_label"
						selected={data.filters.annotationStatus === 'incorrect_label'}
					>
						Incorrect label
					</option>
					<option value="interesting" selected={data.filters.annotationStatus === 'interesting'}>
						Interesting
					</option>
					<option
						value="needs_followup"
						selected={data.filters.annotationStatus === 'needs_followup'}
					>
						Needs follow-up
					</option>
				</select>
			</label>
			<label class="check">
				<input
					type="checkbox"
					name="bookmarked_only"
					value="true"
					checked={data.filters.bookmarkedOnly}
				/>
				<span>Bookmarked only</span>
			</label>
			<div class="filter-actions">
				<button type="submit">Apply</button>
				<a href={`/runs/${data.run.run_id}?selection=${data.failures.selection.selection_id}`}>Reset</a>
			</div>
		</form>

		<div class="stats">
			<div>
				<p>False negatives</p>
				<strong>{data.failures.queue_counts.false_negatives}</strong>
				{#if data.failures.queue_totals}
					<small>of {data.failures.queue_totals.false_negatives}</small>
				{/if}
			</div>
			<div>
				<p>False positives</p>
				<strong>{data.failures.queue_counts.false_positives}</strong>
				{#if data.failures.queue_totals}
					<small>of {data.failures.queue_totals.false_positives}</small>
				{/if}
			</div>
			<div>
				<p>Query splits</p>
				<strong>{data.failures.selection.query_splits.join(', ')}</strong>
			</div>
			<div>
				<p>Gallery splits</p>
				<strong>{data.failures.selection.gallery_splits.join(', ')}</strong>
			</div>
		</div>

			{#if data.disagreements}
				<section class="queue">
				<div class="queue-head">
					<div>
						<p class="queue-label">Teacher-student drift</p>
						<h3>Disagreement queues</h3>
					</div>
					<span>
						teacher-ahead positives {data.disagreements.queue_counts.teacher_ahead_positives},
						student-confused negatives {data.disagreements.queue_counts.student_confused_negatives}
					</span>
				</div>

				<div class="review-grid">
					{#if data.disagreements.teacher_ahead_positives.length > 0}
						<article class="mini-queue">
							<div class="mini-head">
								<p class="queue-label">Teacher high, student low</p>
								<h4>Teacher ahead on positives</h4>
							</div>
							{#each data.disagreements.teacher_ahead_positives as row}
								<div class="mini-row">
									<div class="pair-preview">
										<ReviewPairImage
											pairKey={row.pair_key}
											queryChipId={row.query_chip_id}
											candidateChipId={row.candidate_chip_id}
											alt={row.pair_key}
											size={240}
											pairLabel={row.pair_label}
											city={row.city}
											overlapFraction={row.overlap_fraction}
											reviewStatus={row.annotation?.status ?? null}
										/>
									</div>
									<div class="mini-copy">
										<p>{row.pair_label} in {row.city}</p>
										<strong>teacher rank {row.teacher_rank} vs student {row.student_rank}</strong>
										<span>
											teacher sim {row.teacher_similarity.toFixed(3)} vs student
											{row.student_similarity.toFixed(3)}
										</span>
									</div>
									<div class="mini-annotation">
										<PairAnnotation
											pairKey={row.pair_key}
											initialAnnotation={row.annotation ?? null}
											compact
										/>
									</div>
								</div>
							{/each}
						</article>
					{/if}

					{#if data.disagreements.student_confused_negatives.length > 0}
						<article class="mini-queue negative">
							<div class="mini-head">
								<p class="queue-label">Student high, teacher low</p>
								<h4>Student confused on hard negatives</h4>
							</div>
							{#each data.disagreements.student_confused_negatives as row}
								<div class="mini-row">
									<div class="pair-preview">
										<ReviewPairImage
											pairKey={row.pair_key}
											queryChipId={row.query_chip_id}
											candidateChipId={row.candidate_chip_id}
											alt={row.pair_key}
											size={240}
											pairLabel={row.pair_label}
											city={row.city}
											overlapFraction={row.overlap_fraction}
											reviewStatus={row.annotation?.status ?? null}
										/>
									</div>
									<div class="mini-copy">
										<p>{row.pair_label} in {row.city}</p>
										<strong>student rank {row.student_rank} vs teacher {row.teacher_rank}</strong>
										<span>
											student sim {row.student_similarity.toFixed(3)} vs teacher
											{row.teacher_similarity.toFixed(3)}
										</span>
									</div>
									<div class="mini-annotation">
										<PairAnnotation
											pairKey={row.pair_key}
											initialAnnotation={row.annotation ?? null}
											compact
										/>
									</div>
								</div>
							{/each}
						</article>
					{/if}
				</div>
			</section>
		{/if}

		<section class="queue">
			<div class="queue-head">
				<div>
					<p class="queue-label">Missed positives</p>
					<h3>False negatives</h3>
				</div>
				<span>Positive pairs ranked outside top {data.failures.selection.top_k}</span>
			</div>

			{#if data.failures.false_negatives.length === 0}
				<p class="empty-copy">No false negatives in the visible review slice.</p>
			{:else}
				<div class="review-grid">
					{#each data.failures.false_negatives as row}
						<article class="review-row">
							<div class="pair-preview">
								<ReviewPairImage
									pairKey={row.pair_key}
									queryChipId={row.query_chip_id}
									candidateChipId={row.candidate_chip_id}
									alt={row.pair_key}
									size={240}
									pairLabel={row.pair_label}
									city={row.city}
									overlapFraction={row.overlap_fraction}
									reviewStatus={row.annotation?.status ?? null}
								/>
							</div>
							<div class="review-copy">
								<div class="topline">
									<span>{row.city}</span>
									<span>{row.query_split}</span>
								</div>
								<h4>rank {row.rank}</h4>
								<p>{row.pair_label} at similarity {row.similarity.toFixed(3)}</p>
								<code>{row.query_chip_id} → {row.candidate_chip_id}</code>
							</div>
							<div class="review-copy">
								<div class="topline">
									<span>{Math.round(row.time_delta_seconds / 86400)} days</span>
									<span>{row.center_distance_m.toFixed(1)} m</span>
								</div>
								<h4>overlap {row.overlap_fraction.toFixed(2)}</h4>
								<p>expected positive, but outside top-{data.failures.selection.top_k}</p>
								<code>{row.candidate_chip_id}</code>
							</div>
							<div class="annotation-slot">
								<PairAnnotation
									pairKey={row.pair_key}
									initialAnnotation={row.annotation ?? null}
									compact
								/>
							</div>
						</article>
					{/each}
				</div>
			{/if}
		</section>

		<section class="queue">
			<div class="queue-head">
				<div>
					<p class="queue-label">Intrusive negatives</p>
					<h3>False positives</h3>
				</div>
				<span>Hard negatives retrieved inside top {data.failures.selection.top_k}</span>
			</div>

			{#if data.failures.false_positives.length === 0}
				<p class="empty-copy">No false positives in the visible review slice.</p>
			{:else}
				<div class="review-grid">
					{#each data.failures.false_positives as row}
						<article class="review-row negative">
							<div class="pair-preview">
								<ReviewPairImage
									pairKey={row.pair_key}
									queryChipId={row.query_chip_id}
									candidateChipId={row.candidate_chip_id}
									alt={row.pair_key}
									size={240}
									pairLabel={row.pair_label}
									city={row.city}
									overlapFraction={row.overlap_fraction}
									reviewStatus={row.annotation?.status ?? null}
								/>
							</div>
							<div class="review-copy">
								<div class="topline">
									<span>{row.city}</span>
									<span>{row.query_split}</span>
								</div>
								<h4>rank {row.rank}</h4>
								<p>hard negative at similarity {row.similarity.toFixed(3)}</p>
								<code>{row.query_chip_id} → {row.candidate_chip_id}</code>
							</div>
							<div class="review-copy">
								<div class="topline">
									<span>{Math.round(row.time_delta_seconds / 86400)} days</span>
									<span>{row.center_distance_m.toFixed(1)} m</span>
								</div>
								<h4>overlap {row.overlap_fraction.toFixed(2)}</h4>
								<p>should be rejected, but landed inside top-{data.failures.selection.top_k}</p>
								<code>{row.candidate_chip_id}</code>
							</div>
							<div class="annotation-slot">
								<PairAnnotation
									pairKey={row.pair_key}
									initialAnnotation={row.annotation ?? null}
									compact
								/>
							</div>
						</article>
					{/each}
				</div>
			{/if}
		</section>
	{/if}
</section>

<style>
	.eyebrow,
	.queue-label {
		margin: 0;
		color: #9fbab0;
		font-size: 0.76rem;
		letter-spacing: 0.15em;
		text-transform: uppercase;
	}

	h2 {
		margin: 0.35rem 0 0;
		font-size: 2rem;
		letter-spacing: -0.05em;
	}

	.lede {
		max-width: 48rem;
		margin: 0.75rem 0 0;
		color: #bcc9c4;
		line-height: 1.6;
	}

	code {
		font-family: 'Iosevka Etoile', monospace;
		font-size: 0.82rem;
		word-break: break-all;
	}

	.hero {
		display: grid;
		gap: 1rem;
	}

	.selection-strip {
		display: flex;
		flex-wrap: wrap;
		gap: 0.75rem;
	}

	.selection-strip a {
		padding: 0.55rem 0.8rem;
		border: 1px solid rgba(232, 239, 236, 0.12);
		color: #bcc9c4;
	}

	.selection-strip a.selected {
		background: rgba(217, 232, 200, 0.14);
		color: #eef7dc;
		border-color: rgba(217, 232, 200, 0.32);
	}

	.stats {
		display: grid;
		grid-template-columns: repeat(4, minmax(0, 1fr));
		gap: 1rem;
		margin-top: 1.5rem;
		padding-top: 1rem;
		border-top: 1px solid rgba(232, 239, 236, 0.12);
	}

	.stats p {
		margin: 0;
		color: #95aaa2;
		text-transform: uppercase;
		letter-spacing: 0.1em;
		font-size: 0.74rem;
	}

	.stats strong {
		display: block;
		margin-top: 0.3rem;
		font-size: 1.1rem;
	}

	.stats small {
		display: block;
		margin-top: 0.18rem;
		color: #95aaa2;
	}

	.filter-bar {
		display: flex;
		flex-wrap: wrap;
		align-items: end;
		gap: 0.8rem;
		margin-top: 1rem;
	}

	.filter-bar label {
		display: grid;
		gap: 0.35rem;
	}

	.check {
		display: flex;
		align-items: center;
		gap: 0.7rem;
		padding-top: 0.3rem;
	}

	.check input {
		width: 1rem;
		height: 1rem;
	}

	.filter-bar label span {
		font-size: 0.76rem;
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: #95aaa2;
	}

	.filter-bar select,
	.filter-actions button,
	.filter-actions a {
		border: 1px solid rgba(232, 239, 236, 0.12);
		background: rgba(18, 24, 26, 0.88);
		color: #e8efec;
		padding: 0.6rem 0.8rem;
		font: inherit;
		text-decoration: none;
	}

	.filter-actions {
		display: flex;
		gap: 0.7rem;
	}

	.filter-actions button {
		cursor: pointer;
		background: #d9e8c8;
		color: #122016;
		font-weight: 620;
	}

	.queue {
		margin-top: 2rem;
		padding-top: 1rem;
		border-top: 1px solid rgba(232, 239, 236, 0.12);
	}

	.queue-head {
		display: flex;
		align-items: end;
		justify-content: space-between;
		gap: 1rem;
		margin-bottom: 1rem;
	}

	h3 {
		margin: 0.25rem 0 0;
		font-size: 1.4rem;
		letter-spacing: -0.04em;
	}

	.queue-head span,
	.empty-copy {
		color: #95aaa2;
	}

	.review-grid {
		display: grid;
		gap: 1rem;
	}

	.mini-queue {
		padding-top: 1rem;
		border-top: 1px solid rgba(232, 239, 236, 0.08);
	}

	.mini-head h4 {
		margin: 0.25rem 0 0.8rem;
		font-size: 1.1rem;
	}

	.mini-row {
		display: grid;
		grid-template-columns: minmax(15rem, 18rem) minmax(0, 1fr);
		gap: 0.9rem;
		align-items: center;
		padding-top: 0.9rem;
		border-top: 1px solid rgba(232, 239, 236, 0.06);
	}

	.pair-preview {
		min-width: 0;
	}

	.mini-copy {
		display: grid;
		gap: 0.25rem;
	}

	.mini-copy p,
	.mini-copy strong,
	.mini-copy span {
		margin: 0;
	}

	.mini-copy p,
	.mini-copy span {
		color: #bcc9c4;
	}

	.mini-annotation {
		grid-column: 1 / -1;
	}

	.mini-queue.negative .mini-head h4 {
		color: #f0d0c7;
	}

	.review-row {
		display: grid;
		grid-template-columns: minmax(18rem, 24rem) repeat(2, minmax(0, 1fr));
		gap: 1rem;
		align-items: center;
		padding-top: 1rem;
		border-top: 1px solid rgba(232, 239, 236, 0.08);
	}

	.review-copy {
		display: grid;
		gap: 0.32rem;
	}

	.topline {
		display: flex;
		flex-wrap: wrap;
		gap: 0.55rem;
		color: #95aaa2;
		font-size: 0.76rem;
		text-transform: uppercase;
		letter-spacing: 0.08em;
	}

	.annotation-slot {
		grid-column: 1 / -1;
	}

	h4,
	.review-copy p,
	.review-copy code {
		margin: 0;
	}

	.review-copy p,
	.review-copy code {
		color: #bcc9c4;
	}

	.empty-state {
		min-height: 18rem;
		display: grid;
		place-items: center;
		text-align: center;
		border: 1px solid rgba(232, 239, 236, 0.08);
		background:
			radial-gradient(circle at top right, rgba(184, 246, 194, 0.12), transparent 30%),
			#151c1d;
	}

	.empty-state p {
		margin: 0;
		font-size: 1.1rem;
	}

	.empty-state span {
		display: block;
		margin-top: 0.45rem;
		color: #b7c5c0;
		line-height: 1.55;
	}

	@media (max-width: 1100px) {
		.stats {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}

		.review-row {
			grid-template-columns: 1fr;
		}

		.mini-row {
			grid-template-columns: 1fr 1fr;
		}

		.queue-head {
			flex-direction: column;
			align-items: start;
		}
	}
</style>
