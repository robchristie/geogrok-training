<script lang="ts">
const sections = [
	{
		title: 'Review Queue',
		href: '/review',
		copy: 'Work through unreviewed pairs, failures, and teacher-student disagreements as a deliberate analyst queue.',
	},
	{
		title: 'Chip Browser',
		href: '/chips',
		copy: 'Filter by city, split, sensor, and modality. Inspect real training chips before trusting any metric.',
	},
	{
		title: 'Pair Inspector',
		href: '/pairs',
		copy: 'Review positive_exact, positive_weak, and negative_hard pairs side by side with space for later annotation.',
	},
	{
		title: 'Run Explorer',
		href: '/runs',
		copy: 'Browse benchmark runs, compare teacher and student behavior, and surface confusable failure modes.',
	},
];
</script>

<section class="hero">
	<div class="copy">
		<p class="kicker">Reference-only observability scaffold</p>
		<h2>Image-first review for PAN chips, pairs, and benchmark failures.</h2>
		<p class="lede">
			This UI is being built as a lightweight analyst workspace for this repo. It reuses ideas from
			<code>reference/geogrok</code> without pulling in the full product stack.
		</p>
	</div>
	<div class="poster">
		<div class="grain"></div>
		<div class="stripe stripe-a"></div>
		<div class="stripe stripe-b"></div>
		<div class="label">pair-aware review</div>
	</div>
</section>

<section class="index-grid">
	{#each sections as section}
		<a class="entry" href={section.href}>
			<h3>{section.title}</h3>
			<p>{section.copy}</p>
		</a>
	{/each}
</section>

<section class="support">
	<div>
		<h3>Planned reuse</h3>
		<p>
			SvelteKit review surfaces live here. High-bit-depth browser decode and shader-driven display are
			expected to adapt selected viewer pieces from the reference UI and Kakadu bindings.
		</p>
	</div>
	<div>
		<h3>Hard rule</h3>
		<p>
			Review artifacts are derived for humans only. Training and evaluation continue to use raw
			manifest-backed reads through the GDAL/Kakadu path.
		</p>
	</div>
</section>

<style>
	.hero {
		display: grid;
		grid-template-columns: minmax(0, 1.1fr) minmax(320px, 0.9fr);
		gap: 1.5rem;
		min-height: 50vh;
		align-items: stretch;
	}

	.copy {
		padding: 1rem 0 0.4rem;
		max-width: 54rem;
	}

	.kicker {
		margin: 0 0 0.5rem;
		color: #b8f6c2;
		font-size: 0.78rem;
		letter-spacing: 0.16em;
		text-transform: uppercase;
	}

	h2 {
		margin: 0;
		font-size: clamp(2.6rem, 6vw, 5.5rem);
		line-height: 0.96;
		letter-spacing: -0.06em;
		max-width: 11ch;
	}

	.lede {
		max-width: 38rem;
		margin: 1.1rem 0 0;
		color: #c9d6d1;
		font-size: 1.02rem;
		line-height: 1.55;
	}

	.poster {
		position: relative;
		min-height: 26rem;
		border: 1px solid rgba(232, 239, 236, 0.08);
		background:
			linear-gradient(130deg, rgba(255, 255, 255, 0.06), transparent 40%),
			linear-gradient(180deg, #161e20 0%, #111718 100%);
		overflow: hidden;
	}

	.grain {
		position: absolute;
		inset: 0;
		background-image:
			linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
			linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
		background-size: 18px 18px;
		opacity: 0.4;
	}

	.stripe {
		position: absolute;
		border-radius: 999px;
		filter: blur(0.2px);
	}

	.stripe-a {
		inset: 14% 18% 54% 10%;
		background: linear-gradient(90deg, rgba(184, 246, 194, 0.28), rgba(255, 255, 255, 0.05));
		transform: rotate(-12deg);
	}

	.stripe-b {
		inset: 48% 10% 18% 34%;
		background: linear-gradient(90deg, rgba(255, 224, 188, 0.2), rgba(255, 255, 255, 0.03));
		transform: rotate(11deg);
	}

	.label {
		position: absolute;
		right: 1rem;
		bottom: 1rem;
		padding: 0.35rem 0.6rem;
		font-size: 0.75rem;
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: #111718;
		background: #b8f6c2;
	}

	.index-grid {
		display: grid;
		grid-template-columns: repeat(4, minmax(0, 1fr));
		gap: 1rem;
		margin-top: 1.6rem;
	}

	.entry {
		padding: 1rem 0 1.1rem;
		border-top: 1px solid rgba(232, 239, 236, 0.12);
		transition: transform 160ms ease;
	}

	.entry:hover {
		transform: translateY(-3px);
	}

	.entry h3,
	.support h3 {
		margin: 0 0 0.35rem;
		font-size: 1rem;
	}

	.entry p,
	.support p {
		margin: 0;
		color: #b8c6c1;
		line-height: 1.55;
	}

	.support {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 1.6rem;
		margin-top: 2.2rem;
		padding-top: 1rem;
		border-top: 1px solid rgba(232, 239, 236, 0.08);
	}

	code {
		font-family: 'Iosevka Etoile', monospace;
		font-size: 0.92em;
	}

	@media (max-width: 900px) {
		.hero,
		.index-grid,
		.support {
			grid-template-columns: 1fr;
		}

		h2 {
			max-width: 12ch;
		}
	}
</style>
