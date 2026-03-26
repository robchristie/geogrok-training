<script lang="ts">
import { onMount } from 'svelte';

import ReviewChipImage from '$lib/components/ReviewChipImage.svelte';
import type { PairReviewArtifactRecord } from '$lib/types';

let {
	pairKey,
	queryChipId,
	candidateChipId,
	alt,
	size = 224,
	showControls = false,
	showOverlay = true,
	pairLabel = '',
	city = '',
	overlapFraction = null,
	reviewStatus = null,
}: {
	pairKey: string;
	queryChipId: string;
	candidateChipId: string;
	alt: string;
	size?: number;
	showControls?: boolean;
	showOverlay?: boolean;
	pairLabel?: string;
	city?: string;
	overlapFraction?: number | null;
	reviewStatus?: string | null;
} = $props();

let payload = $state<PairReviewArtifactRecord | null>(null);
let error = $state<string | null>(null);
let blackPoint = $state(0.02);
let whitePoint = $state(0.98);
let gamma = $state(1);

const pairMeta = $derived.by(() => {
	const parts = [pairLabel, city];
	if (overlapFraction !== null) {
		parts.push(`overlap ${overlapFraction.toFixed(2)}`);
	}
	if (reviewStatus) {
		parts.push(`review ${reviewStatus.replaceAll('_', ' ')}`);
	}
	return parts.filter(Boolean).join(' · ');
});

function resetDisplayControls(): void {
	blackPoint = 0.02;
	whitePoint = 0.98;
	gamma = 1;
}

onMount(() => {
	let cancelled = false;

	async function run(): Promise<void> {
		try {
			const response = await fetch(`/api/pairs/${pairKey}/review-artifact`);
			if (!response.ok) {
				throw new Error(`pair review artifact API returned ${response.status}`);
			}
			const parsed = (await response.json()) as PairReviewArtifactRecord;
			if (!cancelled) payload = parsed;
		} catch (cause) {
			if (!cancelled) {
				error = cause instanceof Error ? cause.message : 'Unknown pair review artifact error';
			}
		}
	}

	void run();
	return () => {
		cancelled = true;
	};
});
</script>

<div class="pair-shell">
	{#if showOverlay && pairMeta}
		<div class="pair-header">
			<span class="pair-pill">{pairLabel || 'pair review'}</span>
			<span class="pair-meta">{pairMeta}</span>
		</div>
	{/if}

	{#if showControls}
		<div class="controls">
			<label>
				<span>Black</span>
				<input type="range" min="0" max="0.45" step="0.01" bind:value={blackPoint} />
				<strong>{blackPoint.toFixed(2)}</strong>
			</label>
			<label>
				<span>White</span>
				<input type="range" min="0.55" max="1" step="0.01" bind:value={whitePoint} />
				<strong>{whitePoint.toFixed(2)}</strong>
			</label>
			<label>
				<span>Gamma</span>
				<input type="range" min="0.5" max="2.2" step="0.05" bind:value={gamma} />
				<strong>{gamma.toFixed(2)}</strong>
			</label>
			<button type="button" class="reset" onclick={resetDisplayControls}>Reset</button>
		</div>
	{/if}

	{#if payload}
		<div class="pair-review-image" aria-label={alt}>
			<ReviewChipImage
				chipId={queryChipId}
				alt={`${alt} query`}
				size={size}
				artifact={payload.query}
				blackPoint={blackPoint}
				whitePoint={whitePoint}
				{gamma}
				{showOverlay}
				overlayLabel="query"
				overlayMeta={queryChipId}
				overlayTone="query"
			/>
			<ReviewChipImage
				chipId={candidateChipId}
				alt={`${alt} candidate`}
				size={size}
				artifact={payload.candidate}
				blackPoint={blackPoint}
				whitePoint={whitePoint}
				{gamma}
				{showOverlay}
				overlayLabel="candidate"
				overlayMeta={candidateChipId}
				overlayTone={pairLabel === 'negative_hard' ? 'alert' : 'candidate'}
			/>
		</div>
	{:else}
		<img
			class="pair-fallback"
			src={`/api/pairs/${pairKey}/image?size=${size}&gap=12`}
			alt={alt}
			loading="lazy"
		/>
	{/if}
</div>

{#if error}
	<span class="artifact-note">{error}</span>
{/if}

<style>
	.pair-shell {
		display: grid;
		gap: 0.7rem;
	}

	.pair-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.7rem;
		flex-wrap: wrap;
	}

	.pair-pill,
	.pair-meta {
		font-size: 0.72rem;
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}

	.pair-pill {
		border: 1px solid rgba(217, 232, 200, 0.2);
		background: rgba(217, 232, 200, 0.12);
		color: #eef8dc;
		padding: 0.28rem 0.48rem;
	}

	.pair-meta {
		color: #99aea7;
	}

	.controls {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr)) auto;
		gap: 0.75rem;
		align-items: end;
		padding: 0.8rem;
		border: 1px solid rgba(232, 239, 236, 0.08);
		background: rgba(10, 14, 16, 0.76);
	}

	.controls label {
		display: grid;
		gap: 0.3rem;
	}

	.controls span,
	.controls strong {
		font-size: 0.72rem;
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}

	.controls span {
		color: #98aca5;
	}

	.controls strong {
		color: #e7efec;
		font-weight: 560;
	}

	.controls input[type='range'] {
		width: 100%;
		accent-color: #d9e8c8;
	}

	.reset {
		border: 1px solid rgba(232, 239, 236, 0.12);
		background: rgba(18, 24, 26, 0.88);
		color: #dbe7e2;
		padding: 0.68rem 0.9rem;
		font: inherit;
		cursor: pointer;
	}

	.pair-review-image {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 0.75rem;
		align-items: start;
	}

	.pair-fallback {
		display: block;
		width: 100%;
		aspect-ratio: 2.05;
		object-fit: cover;
		border: 1px solid rgba(232, 239, 236, 0.08);
		background: #111719;
	}

	.artifact-note {
		display: block;
		margin-top: 0.4rem;
		color: #8ea69d;
		font-size: 0.76rem;
		line-height: 1.4;
	}

	@media (max-width: 720px) {
		.controls {
			grid-template-columns: 1fr;
		}
	}
</style>
