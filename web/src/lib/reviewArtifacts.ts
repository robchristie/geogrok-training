import type { ReviewArtifactRecord } from '$lib/types';

let kakaduAssetAvailability: Promise<boolean> | null = null;

export async function hasKakaduDecoderAssets(): Promise<boolean> {
	if (!kakaduAssetAvailability) {
		kakaduAssetAvailability = (async () => {
			try {
				const [jsResponse, wasmResponse] = await Promise.all([
					fetch('/kakadujs/kakadujs.js', { method: 'HEAD' }),
					fetch('/kakadujs/kakadujs.wasm', { method: 'HEAD' }),
				]);
				return jsResponse.ok && wasmResponse.ok;
			} catch {
				return false;
			}
		})();
	}
	return kakaduAssetAvailability;
}

export async function loadChipReviewArtifact(chipId: string): Promise<ReviewArtifactRecord> {
	const response = await fetch(`/api/chips/${chipId}/review-artifact`);
	if (!response.ok) {
		throw new Error(`review artifact API returned ${response.status}`);
	}
	return (await response.json()) as ReviewArtifactRecord;
}
