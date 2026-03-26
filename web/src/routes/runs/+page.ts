import type { RunSummary } from '$lib/types';
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ fetch }) => {
	let runs: RunSummary[] = [];
	let error: string | null = null;

	try {
		const response = await fetch('/api/runs');
		if (!response.ok) {
			throw new Error(`HTTP ${response.status}`);
		}
		runs = (await response.json()) as RunSummary[];
	} catch (cause) {
		error = cause instanceof Error ? cause.message : 'Unknown error';
	}

	return { runs, error };
};
