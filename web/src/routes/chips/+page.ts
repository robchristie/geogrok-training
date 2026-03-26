import type { ChipFacets, ChipRecord } from '$lib/types';
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ fetch, url }) => {
	const params = new URLSearchParams();
	const city = url.searchParams.get('city');
	const split = url.searchParams.get('split');
	const modality = url.searchParams.get('modality');
	const sensor = url.searchParams.get('sensor');

	if (city) params.set('city', city);
	if (split) params.set('split', split);
	if (modality) params.set('modality', modality);
	if (sensor) params.set('sensor', sensor);
	params.set('limit', '48');

	try {
		const [chipsResponse, facetsResponse] = await Promise.all([
			fetch(`/api/chips?${params.toString()}`),
			fetch('/api/chip-facets'),
		]);
		if (!chipsResponse.ok) {
			throw new Error(`chips API returned ${chipsResponse.status}`);
		}
		if (!facetsResponse.ok) {
			throw new Error(`chip facets API returned ${facetsResponse.status}`);
		}
		const [chips, facets] = (await Promise.all([chipsResponse.json(), facetsResponse.json()])) as [
			ChipRecord[],
			ChipFacets,
		];
		return {
			chips,
			facets,
			filters: { city, split, modality, sensor },
			error: null,
		};
	} catch (error) {
		const message = error instanceof Error ? error.message : 'Unknown observability error';
		return {
			chips: [] as ChipRecord[],
			facets: { cities: [], splits: [], modalities: [], sensors: [] } as ChipFacets,
			filters: { city, split, modality, sensor },
			error: message,
		};
	}
};
