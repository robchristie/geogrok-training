<script lang="ts">
import type { PairAnnotation } from '$lib/types';

type ReviewStatus =
	| 'unreviewed'
	| 'confirmed'
	| 'incorrect_label'
	| 'interesting'
	| 'needs_followup';

const statusOptions: Array<{ value: ReviewStatus; label: string }> = [
	{ value: 'unreviewed', label: 'Unreviewed' },
	{ value: 'confirmed', label: 'Confirmed' },
	{ value: 'incorrect_label', label: 'Incorrect' },
	{ value: 'interesting', label: 'Interesting' },
	{ value: 'needs_followup', label: 'Follow-up' },
];

let {
	pairKey,
	initialAnnotation = null,
	compact = false,
}: {
	pairKey: string;
	initialAnnotation?: PairAnnotation | null;
	compact?: boolean;
} = $props();

const initialStatus = () => (initialAnnotation?.status as ReviewStatus | undefined) ?? 'unreviewed';
const initialNote = () => initialAnnotation?.note ?? '';
const initialSaved = () => initialAnnotation;
const initialBookmarked = () => initialAnnotation?.bookmarked ?? false;

let status = $state<ReviewStatus>(initialStatus());
let note = $state(initialNote());
let saved = $state<PairAnnotation | null>(initialSaved());
let bookmarked = $state(initialBookmarked());
let saving = $state(false);
let error = $state<string | null>(null);

async function save() {
	saving = true;
	error = null;
	try {
		const response = await fetch(`/api/annotations/pairs/${pairKey}`, {
			method: 'POST',
			headers: { 'content-type': 'application/json' },
			body: JSON.stringify({ status, note, tags: saved?.tags ?? [], bookmarked }),
		});
		if (!response.ok) {
			throw new Error(`annotation API returned ${response.status}`);
		}
		saved = (await response.json()) as PairAnnotation;
		status = (saved.status as ReviewStatus | undefined) ?? 'unreviewed';
		note = saved.note ?? '';
		bookmarked = saved.bookmarked ?? false;
	} catch (cause) {
		error = cause instanceof Error ? cause.message : 'Unknown annotation error';
	} finally {
		saving = false;
	}
}
</script>

<div class:compact class="annotation">
	<div class="annotation-topline">
		<p>Review</p>
		<div class="topline-meta">
			<button
				type="button"
				class:marked={bookmarked}
				class="bookmark"
				aria-label="Toggle bookmark"
				title="Toggle bookmark"
				onclick={() => {
					bookmarked = !bookmarked;
				}}
			>
				{bookmarked ? '★' : '☆'}
			</button>
			{#if saved}
				<span>{saved.updated_at.slice(0, 19).replace('T', ' ')}</span>
			{/if}
		</div>
	</div>

	<div class="status-strip">
		{#each statusOptions as option}
			<button
				type="button"
				class:selected={status === option.value}
				onclick={() => {
					status = option.value;
				}}
			>
				{option.label}
			</button>
		{/each}
	</div>

	<textarea
		rows={compact ? 2 : 3}
		bind:value={note}
		placeholder="Add a note about label quality, clouds, geometry, or why this matters."
	></textarea>

	<div class="annotation-actions">
		<button class="save" type="button" onclick={save} disabled={saving}>
			{saving ? 'Saving…' : 'Save review'}
		</button>
		{#if error}
			<span class="error">{error}</span>
		{:else if saved}
			<span class="saved">{saved.status.replaceAll('_', ' ')}</span>
		{/if}
	</div>
</div>

<style>
	.annotation {
		display: grid;
		gap: 0.55rem;
		padding-top: 0.7rem;
		border-top: 1px solid rgba(232, 239, 236, 0.08);
	}

	.annotation.compact {
		padding-top: 0.55rem;
	}

	.annotation-topline {
		display: flex;
		justify-content: space-between;
		gap: 0.8rem;
	}

	.topline-meta {
		display: flex;
		align-items: center;
		gap: 0.55rem;
	}

	.annotation-topline p,
	.annotation-topline span {
		margin: 0;
		font-size: 0.72rem;
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: #9fbab0;
	}

	.status-strip {
		display: flex;
		flex-wrap: wrap;
		gap: 0.45rem;
	}

	.status-strip button,
	.save,
	.bookmark {
		border: 1px solid rgba(232, 239, 236, 0.12);
		background: rgba(18, 24, 26, 0.88);
		color: #dbe7e2;
		padding: 0.45rem 0.7rem;
		font: inherit;
		cursor: pointer;
	}

	.status-strip button.selected {
		border-color: rgba(217, 232, 200, 0.36);
		background: rgba(217, 232, 200, 0.14);
		color: #eef7dc;
	}

	.bookmark {
		padding: 0.35rem 0.55rem;
		line-height: 1;
	}

	.bookmark.marked {
		border-color: rgba(247, 215, 170, 0.45);
		background: rgba(247, 215, 170, 0.16);
		color: #f7d7aa;
	}

	textarea {
		width: 100%;
		border: 1px solid rgba(232, 239, 236, 0.12);
		background: rgba(18, 24, 26, 0.88);
		color: #e8efec;
		padding: 0.65rem 0.75rem;
		font: inherit;
		resize: vertical;
	}

	.annotation-actions {
		display: flex;
		align-items: center;
		gap: 0.7rem;
	}

	.save {
		background: #d9e8c8;
		color: #122016;
		font-weight: 620;
	}

	.save:disabled {
		opacity: 0.6;
		cursor: wait;
	}

	.saved,
	.error {
		font-size: 0.82rem;
	}

	.saved {
		color: #b8f6c2;
	}

	.error {
		color: #f0c5bb;
	}
</style>
