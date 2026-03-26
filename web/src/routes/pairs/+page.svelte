<script lang="ts">
import PairAnnotation from '$lib/components/PairAnnotation.svelte';
import ReviewPairImage from '$lib/components/ReviewPairImage.svelte';
import type { PageData } from './$types';

let { data }: { data: PageData } = $props();
</script>

<section class="page">
	<header>
		<p class="eyebrow">Label inspection</p>
		<h2>Pair Inspector</h2>
		<p>
			Review labeled pairs as a queue, filter by annotation state, and push examples through a lightweight
			human review loop without leaving the browser.
		</p>
	</header>

	<div class="layout">
		<aside class="filters">
			<form method="GET">
				<label>
					<span>Pair label</span>
					<select name="pair_label">
						<option value="">All labels</option>
						{#each data.facets.pair_labels as label}
							<option value={label} selected={data.filters.pairLabel === label}>{label}</option>
						{/each}
					</select>
				</label>
				<label>
					<span>City</span>
					<select name="city">
						<option value="">All cities</option>
						{#each data.facets.cities as city}
							<option value={city} selected={data.filters.city === city}>{city}</option>
						{/each}
					</select>
				</label>
				<label>
					<span>Split</span>
					<select name="split">
						<option value="">All splits</option>
						{#each data.facets.splits as split}
							<option value={split} selected={data.filters.split === split}>{split}</option>
						{/each}
					</select>
				</label>
				<label>
					<span>Review state</span>
					<select name="annotation_status">
						<option value="">All pairs</option>
						{#each data.facets.annotation_statuses as status}
							<option value={status} selected={data.filters.annotationStatus === status}>
								{status.replaceAll('_', ' ')}
							</option>
						{/each}
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
				<div class="actions">
					<button type="submit">Apply</button>
					<a href="/pairs">Reset</a>
				</div>
			</form>
		</aside>

		<div class="content">
			<div class="content-head">
				<div>
					<p class="queue-label">Visible queue</p>
					<h3>{data.pairs.length} pairs</h3>
				</div>
				<p class="hint">Review artifacts preserve more signal and fall back to composite PNGs when needed.</p>
			</div>

			{#if data.error}
				<div class="notes">
					<p>Backend unavailable</p>
					<ul>
						<li>{data.error}</li>
						<li>Start `geogrok-obs-api` to browse real pair examples.</li>
					</ul>
				</div>
			{:else if data.pairs.length === 0}
				<div class="empty-state">
					<p>No pairs matched these filters</p>
					<span>Try widening the review-state or pair-label selection.</span>
				</div>
			{:else}
				<div class="pair-grid">
					{#each data.pairs as pair}
						<article class="pair-card">
							<div class="pair-strip">
								<ReviewPairImage
									pairKey={pair.pair_key}
									queryChipId={pair.query_chip_id}
									candidateChipId={pair.candidate_chip_id}
									alt={pair.pair_key}
									size={224}
									showControls
									pairLabel={pair.pair_label}
									city={pair.city}
									overlapFraction={pair.overlap_fraction}
									reviewStatus={pair.annotation?.status ?? null}
								/>
								<div class="meta">
									<span class:positive={pair.pair_group === 'positive'}>{pair.pair_label}</span>
									<span>city: {pair.city}</span>
									<span>overlap: {pair.overlap_fraction.toFixed(2)}</span>
									<span>delta: {Math.round(pair.time_delta_seconds / 86400)} days</span>
									<span>dist: {pair.center_distance_m.toFixed(1)} m</span>
									<span>
										review:
										{pair.annotation ? pair.annotation.status.replaceAll('_', ' ') : 'unreviewed'}
									</span>
								</div>
							</div>
							<PairAnnotation pairKey={pair.pair_key} initialAnnotation={pair.annotation ?? null} compact />
						</article>
					{/each}
				</div>
			{/if}
		</div>
	</div>
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

	header :global(p:last-child) {
		max-width: 48rem;
		margin: 0.7rem 0 0;
		color: #bcc9c4;
		line-height: 1.6;
	}

	.layout {
		display: grid;
		grid-template-columns: 17rem minmax(0, 1fr);
		gap: 1.5rem;
		margin-top: 1.5rem;
	}

	.filters {
		padding-top: 0.4rem;
		border-top: 1px solid rgba(232, 239, 236, 0.12);
	}

	.filters form {
		display: grid;
		gap: 1rem;
	}

	label {
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

	label span {
		font-size: 0.78rem;
		color: #aab8b3;
		text-transform: uppercase;
		letter-spacing: 0.1em;
	}

	select,
	button,
	.actions a {
		border: 1px solid rgba(232, 239, 236, 0.12);
		background: rgba(18, 24, 26, 0.88);
		color: #e8efec;
		padding: 0.7rem 0.85rem;
		font: inherit;
	}

	.actions {
		display: grid;
		grid-template-columns: 1fr auto;
		gap: 0.7rem;
	}

	button {
		cursor: pointer;
		background: #d9e8c8;
		color: #122016;
		font-weight: 620;
	}

	.actions a {
		display: grid;
		place-items: center;
		text-decoration: none;
	}

	.content {
		min-width: 0;
		padding-top: 0.4rem;
		border-top: 1px solid rgba(232, 239, 236, 0.12);
	}

	.content-head {
		display: flex;
		align-items: end;
		justify-content: space-between;
		gap: 1rem;
		margin-bottom: 1rem;
	}

	h3 {
		margin: 0.25rem 0 0;
		font-size: 1.5rem;
		letter-spacing: -0.04em;
	}

	.hint {
		margin: 0;
		max-width: 26rem;
		color: #95aaa2;
		text-align: right;
		line-height: 1.5;
	}

	.pair-grid {
		display: grid;
		gap: 1rem;
	}

	.pair-card {
		padding-top: 1rem;
		border-top: 1px solid rgba(232, 239, 236, 0.12);
	}

	.pair-strip {
		display: grid;
		grid-template-columns: minmax(0, 1.2fr) minmax(14rem, 18rem);
		gap: 1rem;
		align-items: start;
	}

	.meta {
		display: flex;
		flex-direction: column;
		gap: 0.55rem;
		padding: 1rem 0.3rem;
		color: #d9e6e1;
	}

	.meta span:first-child {
		color: #b8f6c2;
		font-weight: 620;
	}

	.meta span.positive {
		color: #b8f6c2;
	}

	.notes,
	.empty-state {
		margin-top: 1.2rem;
		padding-top: 1rem;
		border-top: 1px solid rgba(232, 239, 236, 0.08);
	}

	.notes p,
	.empty-state p {
		margin: 0 0 0.6rem;
		font-size: 0.95rem;
	}

	.notes ul {
		margin: 0;
		padding-left: 1rem;
		color: #bcc9c4;
		line-height: 1.8;
	}

	.empty-state span {
		color: #bcc9c4;
	}

	@media (max-width: 980px) {
		.layout {
			grid-template-columns: 1fr;
		}

		.content-head {
			flex-direction: column;
			align-items: start;
		}

		.hint {
			text-align: left;
		}
	}

	@media (max-width: 900px) {
		.pair-strip {
			grid-template-columns: 1fr;
		}

		.meta {
			padding: 0;
		}
	}
</style>
