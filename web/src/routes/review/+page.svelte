<script lang="ts">
import PairAnnotation from '$lib/components/PairAnnotation.svelte';
import ReviewPairImage from '$lib/components/ReviewPairImage.svelte';
import type { DisagreementRecord, FailureRecord, PairRecord } from '$lib/types';
import type { PageData } from './$types';

let { data }: { data: PageData } = $props();

const bookmarkedPairs = (pairs: PairRecord[]) =>
	pairs.filter((pair) => pair.annotation?.bookmarked);
const bookmarkedFailures = (rows: FailureRecord[]) =>
	rows.filter((row) => row.annotation?.bookmarked);
const bookmarkedDisagreements = (rows: DisagreementRecord[]) =>
	rows.filter((row) => row.annotation?.bookmarked);
</script>

<section class="page">
	<header class="hero">
		<div>
			<p class="eyebrow">Analyst worklist</p>
			<h2>Review Queue</h2>
			<p class="lede">
				A focused queue for triaging unreviewed pairs and the most informative model mistakes. This is the
				surface to keep open while training and evaluation continue.
			</p>
		</div>

		<form class="controls" method="GET">
			<label>
				<span>Review state</span>
				<select name="annotation_status">
					<option value="unreviewed" selected={data.reviewStatus === 'unreviewed'}>Unreviewed</option>
					<option value="reviewed" selected={data.reviewStatus === 'reviewed'}>Reviewed</option>
					<option value="interesting" selected={data.reviewStatus === 'interesting'}>Interesting</option>
					<option value="needs_followup" selected={data.reviewStatus === 'needs_followup'}>
						Needs follow-up
					</option>
					<option value="confirmed" selected={data.reviewStatus === 'confirmed'}>Confirmed</option>
					<option value="incorrect_label" selected={data.reviewStatus === 'incorrect_label'}>
						Incorrect label
					</option>
				</select>
			</label>
			<label class="check">
				<input
					type="checkbox"
					name="bookmarked_only"
					value="true"
					checked={data.bookmarkedOnly}
				/>
				<span>Bookmarked only</span>
			</label>
			{#if data.activeRun}
				<label>
					<span>Run</span>
					<select name="run_id">
						{#each data.runs as run}
							<option value={run.run_id} selected={data.activeRun.run_id === run.run_id}>
								{run.run_id}
							</option>
						{/each}
					</select>
				</label>
			{/if}
			<button type="submit">Refresh queue</button>
		</form>
	</header>

	{#if data.error}
		<div class="empty-state">
			<p>Review queue unavailable</p>
			<span>{data.error}</span>
		</div>
	{:else}
		{@const savedPairs = bookmarkedPairs(data.pairs)}
		{@const savedFailures = data.failures ? bookmarkedFailures([...data.failures.false_negatives, ...data.failures.false_positives]) : []}
		{@const savedDisagreements = data.disagreements ? bookmarkedDisagreements([...data.disagreements.teacher_ahead_positives, ...data.disagreements.student_confused_negatives]) : []}
		<div class="sections">
			{#if !data.bookmarkedOnly && (savedPairs.length > 0 || savedFailures.length > 0 || savedDisagreements.length > 0)}
				<section class="queue spotlight">
					<div class="queue-head">
						<div>
							<p class="queue-label">Saved now</p>
							<h3>Bookmarked</h3>
						</div>
						<span>{savedPairs.length + savedFailures.length + savedDisagreements.length} saved items</span>
					</div>
					<div class="card-grid">
						{#each [...savedPairs.slice(0, 3), ...savedFailures.slice(0, 3), ...savedDisagreements.slice(0, 3)] as row}
							<article class="card bookmarked">
								<ReviewPairImage
									pairKey={row.pair_key}
									queryChipId={row.query_chip_id}
									candidateChipId={row.candidate_chip_id}
									alt={row.pair_key}
									size={208}
									pairLabel={row.pair_label}
									city={row.city}
									overlapFraction={row.overlap_fraction}
									reviewStatus={row.annotation?.status ?? null}
								/>
								<div class="copy">
									<p>{row.pair_label} · {row.city}</p>
									<strong>bookmarked</strong>
									<span>{row.annotation ? row.annotation.status.replaceAll('_', ' ') : 'unreviewed'}</span>
								</div>
								<PairAnnotation pairKey={row.pair_key} initialAnnotation={row.annotation ?? null} compact />
							</article>
						{/each}
					</div>
				</section>
			{/if}

			<section class="queue">
				<div class="queue-head">
					<div>
						<p class="queue-label">Dataset labels</p>
						<h3>Pair review</h3>
					</div>
					<span>{data.pairs.length} items in this slice</span>
				</div>
				<div class="card-grid">
					{#each data.pairs as pair}
						<article class="card">
							<ReviewPairImage
								pairKey={pair.pair_key}
								queryChipId={pair.query_chip_id}
								candidateChipId={pair.candidate_chip_id}
								alt={pair.pair_key}
								size={208}
								pairLabel={pair.pair_label}
								city={pair.city}
								overlapFraction={pair.overlap_fraction}
								reviewStatus={pair.annotation?.status ?? null}
							/>
							<div class="copy">
								<p>{pair.pair_label} · {pair.city}</p>
								<strong>overlap {pair.overlap_fraction.toFixed(2)}</strong>
								<span>
									{pair.annotation ? pair.annotation.status.replaceAll('_', ' ') : 'unreviewed'}
								</span>
							</div>
							<PairAnnotation pairKey={pair.pair_key} initialAnnotation={pair.annotation ?? null} compact />
						</article>
					{/each}
				</div>
			</section>

			{#if data.failures}
				<section class="queue">
					<div class="queue-head">
						<div>
							<p class="queue-label">Model misses</p>
							<h3>Student failures</h3>
						</div>
						<span>
							{data.failures.queue_counts.false_negatives} FN,
							{data.failures.queue_counts.false_positives} FP
						</span>
					</div>
					<div class="card-grid">
						{#each [...data.failures.false_negatives.slice(0, 4), ...data.failures.false_positives.slice(0, 4)] as row}
						<article class="card">
							<ReviewPairImage
								pairKey={row.pair_key}
								queryChipId={row.query_chip_id}
								candidateChipId={row.candidate_chip_id}
								alt={row.pair_key}
								size={208}
								pairLabel={row.pair_label}
								city={row.city}
								overlapFraction={row.overlap_fraction}
								reviewStatus={row.annotation?.status ?? null}
							/>
								<div class="copy">
									<p>{row.pair_label} · {row.city}</p>
									<strong>rank {row.rank} · sim {row.similarity.toFixed(3)}</strong>
									<span>
										{row.annotation ? row.annotation.status.replaceAll('_', ' ') : 'unreviewed'}
									</span>
								</div>
								<PairAnnotation pairKey={row.pair_key} initialAnnotation={row.annotation ?? null} compact />
							</article>
						{/each}
					</div>
				</section>
			{/if}

			{#if data.disagreements}
				<section class="queue">
					<div class="queue-head">
						<div>
							<p class="queue-label">Adaptation gaps</p>
							<h3>Teacher-student disagreements</h3>
						</div>
						<span>
							teacher-ahead {data.disagreements.queue_counts.teacher_ahead_positives},
							student-confused {data.disagreements.queue_counts.student_confused_negatives}
						</span>
					</div>
					<div class="card-grid">
						{#each [...data.disagreements.teacher_ahead_positives.slice(0, 3), ...data.disagreements.student_confused_negatives.slice(0, 3)] as row}
						<article class="card">
							<ReviewPairImage
								pairKey={row.pair_key}
								queryChipId={row.query_chip_id}
								candidateChipId={row.candidate_chip_id}
								alt={row.pair_key}
								size={208}
								pairLabel={row.pair_label}
								city={row.city}
								overlapFraction={row.overlap_fraction}
								reviewStatus={row.annotation?.status ?? null}
							/>
								<div class="copy">
									<p>{row.pair_label} · {row.city}</p>
									<strong>T {row.teacher_rank} vs S {row.student_rank}</strong>
									<span>
										{row.annotation ? row.annotation.status.replaceAll('_', ' ') : 'unreviewed'}
									</span>
								</div>
								<PairAnnotation pairKey={row.pair_key} initialAnnotation={row.annotation ?? null} compact />
							</article>
						{/each}
					</div>
				</section>
			{/if}
		</div>
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

	.hero {
		display: grid;
		grid-template-columns: minmax(0, 1.1fr) minmax(18rem, 24rem);
		gap: 1.5rem;
		align-items: end;
	}

	h2 {
		margin: 0.35rem 0 0;
		font-size: clamp(2rem, 4vw, 3.2rem);
		letter-spacing: -0.05em;
	}

	.lede {
		max-width: 42rem;
		margin: 0.8rem 0 0;
		color: #bcc9c4;
		line-height: 1.6;
	}

	.controls {
		display: grid;
		gap: 0.8rem;
	}

	.controls label {
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

	.controls label span {
		font-size: 0.76rem;
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: #95aaa2;
	}

	.controls select,
	.controls button {
		border: 1px solid rgba(232, 239, 236, 0.12);
		background: rgba(18, 24, 26, 0.88);
		color: #e8efec;
		padding: 0.7rem 0.85rem;
		font: inherit;
	}

	.controls button {
		cursor: pointer;
		background: #d9e8c8;
		color: #122016;
		font-weight: 620;
	}

	.sections {
		display: grid;
		gap: 2rem;
		margin-top: 2rem;
	}

	.queue {
		padding-top: 1rem;
		border-top: 1px solid rgba(232, 239, 236, 0.12);
	}

	.queue.spotlight {
		border-top-color: rgba(247, 215, 170, 0.2);
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
	.copy span {
		color: #95aaa2;
	}

	.card-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(18rem, 1fr));
		gap: 1rem;
	}

	.card {
		padding-top: 1rem;
		border-top: 1px solid rgba(232, 239, 236, 0.08);
	}

	.card.bookmarked {
		border-top-color: rgba(247, 215, 170, 0.28);
	}

	.copy {
		display: grid;
		gap: 0.25rem;
		margin-top: 0.7rem;
	}

	.copy p,
	.copy strong,
	.copy span {
		margin: 0;
	}

	.copy p {
		color: #bcc9c4;
	}

	.empty-state {
		margin-top: 1.6rem;
		padding-top: 1rem;
		border-top: 1px solid rgba(232, 239, 236, 0.08);
	}

	.empty-state p {
		margin: 0 0 0.5rem;
	}

	.empty-state span {
		color: #bcc9c4;
	}

	@media (max-width: 980px) {
		.hero {
			grid-template-columns: 1fr;
		}

		.queue-head {
			flex-direction: column;
			align-items: start;
		}
	}
</style>
