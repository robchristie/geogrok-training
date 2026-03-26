import type { PairFacets, PairRecord } from '$lib/types';
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ fetch, url }) => {
	const pairLabel = url.searchParams.get('pair_label');
	const city = url.searchParams.get('city');
	const split = url.searchParams.get('split');
	const annotationStatus = url.searchParams.get('annotation_status');
	const bookmarkedOnly = url.searchParams.get('bookmarked_only') === 'true';
	const params = new URLSearchParams({ limit: '16' });
	if (pairLabel) params.set('pair_label', pairLabel);
	if (city) params.set('city', city);
	if (split) params.set('split', split);
	if (annotationStatus) params.set('annotation_status', annotationStatus);
	if (bookmarkedOnly) params.set('bookmarked_only', 'true');

	let pairs: PairRecord[] = [];
	let facets: PairFacets = {
		pair_labels: [],
		cities: [],
		splits: [],
		annotation_statuses: [],
	};
	let error: string | null = null;

	try {
		const [pairsResponse, facetsResponse] = await Promise.all([
			fetch(`/api/pairs?${params.toString()}`),
			fetch('/api/pair-facets'),
		]);
		if (!pairsResponse.ok) {
			throw new Error(`pairs API returned ${pairsResponse.status}`);
		}
		if (!facetsResponse.ok) {
			throw new Error(`pair facets API returned ${facetsResponse.status}`);
		}
		[pairs, facets] = (await Promise.all([pairsResponse.json(), facetsResponse.json()])) as [
			PairRecord[],
			PairFacets,
		];
	} catch (cause) {
		error = cause instanceof Error ? cause.message : 'Unknown error';
	}

	return {
		pairs,
		facets,
		filters: {
			pairLabel,
			city,
			split,
			annotationStatus,
			bookmarkedOnly,
		},
		error,
	};
};
