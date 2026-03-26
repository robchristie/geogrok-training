<script lang="ts">
import type { PageData } from './$types';

let { data }: { data: PageData } = $props();
</script>

<section class="page">
	<header>
		<p class="eyebrow">Model behavior</p>
		<h2>Run Explorer</h2>
		<p>
			This route will connect to <code>/api/runs</code> and later to run-specific failure queues,
			teacher-student disagreements, and annotation-ready review lists.
		</p>
	</header>

	<div class="table-shell">
		<div class="row head">
			<span>run</span>
			<span>kind</span>
			<span>teacher</span>
			<span>student</span>
			<span>key metric</span>
		</div>
		{#if data.error}
			<div class="row error">
				<span>{data.error}</span>
				<span>backend unavailable</span>
				<span>start `geogrok-obs-api`</span>
				<span>and the Svelte dev server</span>
				<span>to load real runs</span>
			</div>
		{:else}
			{#each data.runs as run}
				<div class="row">
					<span><a class="run-link" href={`/runs/${run.run_id}`}>{run.run_id}</a></span>
					<span>{run.run_kind}</span>
					<span>{run.teacher_model ?? '—'}</span>
					<span>{run.student_model ?? '—'}</span>
					<span>
						{#if run.metrics['student.exact_recall_at_10'] !== undefined}
							student.exact_R@10 {run.metrics['student.exact_recall_at_10'].toFixed(3)}
						{:else if run.metrics['best.exact_recall_at_10'] !== undefined}
							best.exact_R@10 {run.metrics['best.exact_recall_at_10'].toFixed(3)}
						{:else if run.metrics['train.samples_per_second_mean'] !== undefined}
							train.samples/s {run.metrics['train.samples_per_second_mean'].toFixed(2)}
						{:else}
							no summary metric
						{/if}
					</span>
				</div>
			{/each}
		{/if}
	</div>

	<div class="detail">
		<p>Planned follow-up</p>
		<ul>
			<li>link top-k retrievals to pair inspector</li>
			<li>materialize false positive and false negative queues</li>
			<li>surface adversarial hard-negative reviews directly</li>
		</ul>
	</div>
</section>

<style>
	.eyebrow {
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
		margin: 0.7rem 0 0;
		max-width: 48rem;
		color: #bcc9c4;
		line-height: 1.6;
	}

	code {
		font-family: 'Iosevka Etoile', monospace;
		font-size: 0.92em;
	}

	.table-shell {
		margin-top: 1.5rem;
		border-top: 1px solid rgba(232, 239, 236, 0.12);
	}

	.row {
		display: grid;
		grid-template-columns: 1.5fr 1.1fr 1.2fr 1.1fr 1fr;
		gap: 1rem;
		padding: 0.9rem 0;
		border-bottom: 1px solid rgba(232, 239, 236, 0.08);
		color: #d7e4df;
	}

	.row.head {
		color: #91a9a0;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		font-size: 0.72rem;
	}

	.row.error {
		color: #f0c5bb;
	}

	.run-link {
		color: #e8efec;
		border-bottom: 1px solid rgba(184, 246, 194, 0.35);
	}

	.detail {
		margin-top: 1.6rem;
	}

	.detail p {
		margin: 0 0 0.6rem;
	}

	.detail ul {
		margin: 0;
		padding-left: 1rem;
		color: #bcc9c4;
		line-height: 1.8;
	}

	@media (max-width: 920px) {
		.row {
			grid-template-columns: 1fr;
			gap: 0.3rem;
		}
	}
</style>
