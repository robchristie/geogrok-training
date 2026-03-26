import type { PairRecord, RunDisagreements, RunFailures, RunSummary } from '$lib/types';
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ fetch, url }) => {
	const reviewStatus = url.searchParams.get('annotation_status') ?? 'unreviewed';
	const runIdParam = url.searchParams.get('run_id');
	const bookmarkedOnly = url.searchParams.get('bookmarked_only') === 'true';

	try {
		const runsResponse = await fetch('/api/runs');
		if (!runsResponse.ok) {
			throw new Error(`runs API returned ${runsResponse.status}`);
		}
		const runs = (await runsResponse.json()) as RunSummary[];
		const panAdaptRun =
			(runIdParam ? runs.find((run) => run.run_id === runIdParam) : null) ??
			runs.find((run) => run.run_kind === 'pan_adapt_benchmark') ??
			runs[0] ??
			null;

		const pairQuery = new URLSearchParams({
			annotation_status: reviewStatus,
			limit: '10',
		});
		if (bookmarkedOnly) pairQuery.set('bookmarked_only', 'true');

		const requests: Array<Promise<Response>> = [fetch(`/api/pairs?${pairQuery.toString()}`)];
		if (panAdaptRun) {
			const runFailureQuery = new URLSearchParams({
				selection: 'student',
				annotation_status: reviewStatus,
				limit: '8',
			});
			const runDisagreementQuery = new URLSearchParams({
				annotation_status: reviewStatus,
				limit: '6',
			});
			if (bookmarkedOnly) {
				runFailureQuery.set('bookmarked_only', 'true');
				runDisagreementQuery.set('bookmarked_only', 'true');
			}
			requests.push(
				fetch(`/api/runs/${panAdaptRun.run_id}/failures?${runFailureQuery.toString()}`),
				fetch(`/api/runs/${panAdaptRun.run_id}/disagreements?${runDisagreementQuery.toString()}`),
			);
		}

		const responses = await Promise.all(requests);
		for (const response of responses) {
			if (!response.ok) {
				throw new Error(`review API returned ${response.status}`);
			}
		}

		const pairs = (await responses[0].json()) as PairRecord[];
		let failures: RunFailures | null = null;
		let disagreements: RunDisagreements | null = null;
		if (panAdaptRun && responses.length >= 3) {
			failures = (await responses[1].json()) as RunFailures;
			disagreements = (await responses[2].json()) as RunDisagreements;
		}

		return {
			runs,
			activeRun: panAdaptRun,
			reviewStatus,
			bookmarkedOnly,
			pairs,
			failures,
			disagreements,
			error: null,
		};
	} catch (error) {
		const message = error instanceof Error ? error.message : 'Unknown observability error';
		return {
			runs: [] as RunSummary[],
			activeRun: null as RunSummary | null,
			reviewStatus,
			bookmarkedOnly,
			pairs: [] as PairRecord[],
			failures: null as RunFailures | null,
			disagreements: null as RunDisagreements | null,
			error: message,
		};
	}
};
