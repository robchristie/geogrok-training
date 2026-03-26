<script lang="ts">
import { onMount } from 'svelte';

import type { Jp2k16DecodeResult } from '$lib/Jp2k16';
import { Jp2k16Decoder } from '$lib/Jp2k16';
import { hasKakaduDecoderAssets, loadChipReviewArtifact } from '$lib/reviewArtifacts';
import type { ReviewArtifactRecord } from '$lib/types';

let {
	chipId,
	alt,
	size = 320,
	artifact = null,
	blackPoint = 0.02,
	whitePoint = 0.98,
	gamma = 1,
	showOverlay = false,
	overlayLabel = '',
	overlayMeta = '',
	overlayTone = 'neutral',
}: {
	chipId: string;
	alt: string;
	size?: number;
	artifact?: ReviewArtifactRecord | null;
	blackPoint?: number;
	whitePoint?: number;
	gamma?: number;
	showOverlay?: boolean;
	overlayLabel?: string;
	overlayMeta?: string;
	overlayTone?: 'neutral' | 'query' | 'candidate' | 'alert';
} = $props();

let canvas = $state<HTMLCanvasElement | null>(null);
let error = $state<string | null>(null);
let resolvedArtifact = $state<ReviewArtifactRecord | null>(null);
let decodedArtifact = $state<Jp2k16DecodeResult | null>(null);
let decodedRange = $state<{ min: number; max: number }>({ min: 0, max: 255 });
let loading = $state(true);
let mode = $state<'j2c' | 'artifact' | 'fallback'>('fallback');

const fallbackSrc = $derived(`/api/chips/${chipId}/image?size=${size}`);
const overlayStats = $derived(
	resolvedArtifact
		? `${resolvedArtifact.bits_per_sample}-bit ${resolvedArtifact.artifact_kind.toUpperCase()}`
		: 'review artifact',
);

function sampleDisplayWindow(data: Uint8Array | Uint16Array | Int16Array): {
	min: number;
	max: number;
} {
	const step = Math.max(1, Math.floor(data.length / 16384));
	const sample: number[] = [];
	for (let index = 0; index < data.length; index += step) {
		sample.push(Number(data[index]));
	}
	sample.sort((left, right) => left - right);
	if (sample.length === 0) {
		return { min: 0, max: 255 };
	}
	return {
		min: sample[0],
		max: Math.max(sample[sample.length - 1], sample[0] + 1),
	};
}

function drawDecodedArtifact(target: HTMLCanvasElement, decoded: Jp2k16DecodeResult): void {
	const context = target.getContext('2d');
	if (!context) {
		throw new Error('Unable to acquire 2D canvas context.');
	}
	target.width = decoded.width;
	target.height = decoded.height;
	const image = context.createImageData(decoded.width, decoded.height);
	const rangeSpan = Math.max(1, decodedRange.max - decodedRange.min);
	const normalizedBlack = Math.max(0, Math.min(1, blackPoint));
	const normalizedWhite = Math.max(normalizedBlack + 0.01, Math.min(1, whitePoint));
	const low = decodedRange.min + normalizedBlack * rangeSpan;
	const high = decodedRange.min + normalizedWhite * rangeSpan;
	const span = Math.max(1, high - low);
	const safeGamma = Math.max(0.1, gamma);
	const channels = Math.max(1, decoded.components);
	for (let pixelIndex = 0; pixelIndex < decoded.width * decoded.height; pixelIndex += 1) {
		const value = Number(decoded.data[pixelIndex * channels]);
		const normalized = Math.max(0, Math.min(1, (value - low) / span));
		const corrected = normalized ** (1 / safeGamma);
		const scaled = Math.max(0, Math.min(255, Math.round(corrected * 255)));
		const offset = pixelIndex * 4;
		image.data[offset] = scaled;
		image.data[offset + 1] = scaled;
		image.data[offset + 2] = scaled;
		image.data[offset + 3] = 255;
	}
	context.putImageData(image, 0, 0);
}

onMount(() => {
	let cancelled = false;

	async function run(): Promise<void> {
		try {
			const currentArtifact =
				resolvedArtifact ?? artifact ?? (await loadChipReviewArtifact(chipId));
			if (cancelled) return;
			resolvedArtifact = currentArtifact;

			if (currentArtifact.artifact_kind === 'j2c' && (await hasKakaduDecoderAssets())) {
				const response = await fetch(currentArtifact.content_url);
				if (!response.ok) {
					throw new Error(`artifact content returned ${response.status}`);
				}
				const buffer = await response.arrayBuffer();
				if (cancelled) return;
				const decoder = new Jp2k16Decoder();
				const decoded = await decoder.decode(buffer);
				if (cancelled) return;
				decodedArtifact = decoded;
				decodedRange = sampleDisplayWindow(decoded.data);
				mode = 'j2c';
				return;
			}

			if (currentArtifact.artifact_kind === 'png') {
				mode = 'artifact';
				return;
			}
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Unknown review artifact error';
		} finally {
			if (!cancelled) loading = false;
		}
	}

	void run();
	return () => {
		cancelled = true;
	};
});

$effect(() => {
	if (mode !== 'j2c' || !canvas || !decodedArtifact) return;
	drawDecodedArtifact(canvas, decodedArtifact);
});
</script>

<div class="artifact-frame">
	{#if mode === 'j2c'}
		<canvas bind:this={canvas} class="artifact-canvas" aria-label={alt}></canvas>
	{:else if mode === 'artifact' && resolvedArtifact}
		<img class="artifact-image" src={resolvedArtifact.content_url} alt={alt} loading="lazy" />
	{:else}
		<img class="artifact-image" src={fallbackSrc} alt={alt} loading="lazy" />
	{/if}

	{#if showOverlay}
		<div class="overlay">
			<div class="overlay-top">
				{#if overlayLabel}
					<span class={`overlay-badge ${overlayTone}`}>{overlayLabel}</span>
				{/if}
				<span class="overlay-badge stats">{overlayStats}</span>
			</div>
			{#if overlayMeta}
				<div class="overlay-bottom">
					<span>{overlayMeta}</span>
				</div>
			{/if}
		</div>
	{/if}
</div>

{#if error}
	<span class="artifact-note">{error}</span>
{:else if loading}
	<span class="artifact-note">Loading review artifact…</span>
{/if}

<style>
	.artifact-frame {
		position: relative;
	}

	.artifact-canvas,
	.artifact-image {
		display: block;
		width: 100%;
		aspect-ratio: 1;
		object-fit: cover;
		border: 1px solid rgba(232, 239, 236, 0.08);
		background: #111719;
	}

	.overlay {
		position: absolute;
		inset: 0;
		pointer-events: none;
		display: flex;
		flex-direction: column;
		justify-content: space-between;
		padding: 0.65rem;
	}

	.overlay-top,
	.overlay-bottom {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.5rem;
		flex-wrap: wrap;
	}

	.overlay-bottom {
		justify-content: flex-start;
	}

	.overlay-badge,
	.overlay-bottom span {
		border: 1px solid rgba(232, 239, 236, 0.12);
		background: rgba(8, 12, 13, 0.72);
		backdrop-filter: blur(14px);
		color: #f2f6f4;
		padding: 0.28rem 0.52rem;
		font-size: 0.69rem;
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}

	.overlay-badge.query {
		border-color: rgba(217, 232, 200, 0.34);
		background: rgba(217, 232, 200, 0.18);
		color: #eff8dd;
	}

	.overlay-badge.candidate {
		border-color: rgba(176, 213, 255, 0.34);
		background: rgba(176, 213, 255, 0.16);
		color: #e4f0ff;
	}

	.overlay-badge.alert {
		border-color: rgba(247, 198, 170, 0.4);
		background: rgba(247, 198, 170, 0.16);
		color: #ffe6d8;
	}

	.overlay-badge.stats {
		color: #cad8d2;
	}

	.artifact-note {
		display: block;
		margin-top: 0.4rem;
		color: #8ea69d;
		font-size: 0.76rem;
		line-height: 1.4;
	}
</style>
