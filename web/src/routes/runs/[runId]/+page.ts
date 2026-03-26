import type { RunDetail, RunDisagreements, RunFailures } from '$lib/types';
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ fetch, params, url }) => {
	const selection = url.searchParams.get('selection');
	const annotationStatus = url.searchParams.get('annotation_status');
	const bookmarkedOnly = url.searchParams.get('bookmarked_only') === 'true';
	const query = new URLSearchParams({ limit: '12', top_k: '10' });
	if (selection) query.set('selection', selection);
	if (annotationStatus) query.set('annotation_status', annotationStatus);
	if (bookmarkedOnly) query.set('bookmarked_only', 'true');

	try {
		const detailResponse = await fetch(`/api/runs/${params.runId}`);
		if (!detailResponse.ok) {
			throw new Error(`run API returned ${detailResponse.status}`);
		}
		const run = (await detailResponse.json()) as RunDetail;
		const failuresResponse = await fetch(`/api/runs/${params.runId}/failures?${query.toString()}`);
		if (!failuresResponse.ok) {
			throw new Error(`failure API returned ${failuresResponse.status}`);
		}
		const failures = (await failuresResponse.json()) as RunFailures;
		let disagreements: RunDisagreements | null = null;
		if (run.run_kind === 'pan_adapt_benchmark') {
			const disagreementQuery = new URLSearchParams({ limit: '8' });
			if (annotationStatus) disagreementQuery.set('annotation_status', annotationStatus);
			if (bookmarkedOnly) disagreementQuery.set('bookmarked_only', 'true');
			const disagreementResponse = await fetch(
				`/api/runs/${params.runId}/disagreements?${disagreementQuery.toString()}`,
			);
			if (!disagreementResponse.ok) {
				throw new Error(`disagreement API returned ${disagreementResponse.status}`);
			}
			disagreements = (await disagreementResponse.json()) as RunDisagreements;
		}
		return {
			run,
			failures,
			disagreements,
			filters: {
				selection,
				annotationStatus,
				bookmarkedOnly,
			},
			error: null,
		};
	} catch (error) {
		const message = error instanceof Error ? error.message : 'Unknown observability error';
		return {
			run: null as RunDetail | null,
			failures: null as RunFailures | null,
			disagreements: null as RunDisagreements | null,
			filters: {
				selection,
				annotationStatus,
				bookmarkedOnly,
			},
			error: message,
		};
	}
};
