<script lang="ts">
import ReviewChipImage from '$lib/components/ReviewChipImage.svelte';
import type { PageData } from './$types';

let { data }: { data: PageData } = $props();
</script>

<section class="page">
	<header>
		<p class="eyebrow">Dataset review</p>
		<h2>Chip Browser</h2>
		<p>Real PAN chips from the manifest, now preferring cached review artifacts over on-demand PNG quicklooks.</p>
	</header>

	<div class="layout">
		<aside class="filters">
			<form method="GET">
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
					<span>Sensor</span>
					<select name="sensor">
						<option value="">All sensors</option>
						{#each data.facets.sensors as sensor}
							<option value={sensor} selected={data.filters.sensor === sensor}>{sensor}</option>
						{/each}
					</select>
				</label>
				<label>
					<span>Modality</span>
					<select name="modality">
						<option value="">All modalities</option>
						{#each data.facets.modalities as modality}
							<option value={modality} selected={data.filters.modality === modality}>{modality}</option>
						{/each}
					</select>
				</label>
				<div class="actions">
					<button type="submit">Apply</button>
					<a href="/chips">Reset</a>
				</div>
			</form>
		</aside>

		<div class="canvas">
			<div class="canvas-head">
				<div>
					<p class="label">Visible slice</p>
					<h3>{data.chips.length} chips</h3>
				</div>
				<p class="hint">Review artifacts are cached for browsing and fall back to source-rendered quicklooks.</p>
			</div>

			{#if data.error}
				<div class="empty-state">
					<p>Backend unavailable</p>
					<span>{data.error}</span>
				</div>
			{:else if data.chips.length === 0}
				<div class="empty-state">
					<p>No chips matched these filters</p>
					<span>Try widening the city, split, or sensor selection.</span>
				</div>
			{:else}
				<div class="chip-grid">
					{#each data.chips as chip}
						<article class="chip-card">
							<ReviewChipImage
								chipId={chip.chip_id}
								alt={chip.chip_id}
								size={320}
								showOverlay
								overlayLabel={chip.city}
								overlayMeta={`${chip.split} · ${chip.sensor || chip.modality}`}
							/>
							<div class="chip-copy">
								<div class="chip-topline">
									<span>{chip.city}</span>
									<span>{chip.split}</span>
								</div>
								<h4>{chip.sensor || chip.modality}</h4>
								<p>{chip.width}×{chip.height}px at ({chip.x0}, {chip.y0})</p>
								<code>{chip.chip_id}</code>
							</div>
						</article>
					{/each}
				</div>
			{/if}
		</div>
	</div>
</section>

<style>
	.eyebrow,
	.label {
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
		max-width: 44rem;
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
	}

	.canvas {
		min-width: 0;
		padding-top: 0.4rem;
		border-top: 1px solid rgba(232, 239, 236, 0.12);
	}

	.canvas-head {
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
		max-width: 24rem;
		color: #95aaa2;
		text-align: right;
		line-height: 1.5;
	}

	.chip-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(14rem, 1fr));
		gap: 1rem;
	}

	.chip-card {
		display: grid;
		gap: 0.7rem;
	}

	.chip-copy {
		display: grid;
		gap: 0.28rem;
	}

	.chip-topline {
		display: flex;
		justify-content: space-between;
		gap: 0.8rem;
		color: #9fbab0;
		font-size: 0.78rem;
		text-transform: uppercase;
		letter-spacing: 0.08em;
	}

	h4 {
		margin: 0;
		font-size: 1rem;
	}

	.chip-copy p,
	code {
		margin: 0;
		color: #bcc9c4;
	}

	code {
		font-family: 'Iosevka Etoile', monospace;
		font-size: 0.8rem;
		word-break: break-all;
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

	@media (max-width: 980px) {
		.layout {
			grid-template-columns: 1fr;
		}

		.canvas-head {
			flex-direction: column;
			align-items: start;
		}

		.hint {
			text-align: left;
		}
	}
</style>
