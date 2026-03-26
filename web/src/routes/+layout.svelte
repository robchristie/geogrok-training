<script lang="ts">
import { navItems } from '$lib/nav';
</script>

<svelte:head>
	<title>GeoGrok Observability</title>
	<meta
		name="description"
		content="Review chips, inspect pair labels, and examine benchmark failures for PAN-first geospatial training."
	/>
</svelte:head>

<div class="app-shell">
	<header class="topbar">
		<div class="brand-block">
			<p class="eyebrow">GeoGrok Training</p>
			<h1>Observability Lab</h1>
		</div>
		<nav>
			{#each navItems as item}
				<a href={item.href}>{item.label}</a>
			{/each}
		</nav>
	</header>

	<main>
		<slot />
	</main>
</div>

<style>
	:global(body) {
		margin: 0;
		font-family:
			'Iosevka Etoile', 'IBM Plex Sans', 'Segoe UI', sans-serif;
		background:
			radial-gradient(circle at top left, rgba(209, 255, 214, 0.22), transparent 32%),
			radial-gradient(circle at top right, rgba(247, 215, 170, 0.14), transparent 28%),
			linear-gradient(180deg, #0d1112 0%, #111719 100%);
		color: #e8efec;
		min-height: 100vh;
	}

	:global(*) {
		box-sizing: border-box;
	}

	:global(a) {
		color: inherit;
		text-decoration: none;
	}

	.app-shell {
		min-height: 100vh;
		padding: 1.4rem 1.4rem 2rem;
	}

	.topbar {
		display: flex;
		align-items: end;
		justify-content: space-between;
		gap: 1rem;
		padding-bottom: 1.2rem;
		border-bottom: 1px solid rgba(232, 239, 236, 0.12);
	}

	.brand-block h1 {
		margin: 0;
		font-size: clamp(1.6rem, 2vw, 2.4rem);
		font-weight: 620;
		letter-spacing: -0.05em;
	}

	.eyebrow {
		margin: 0 0 0.3rem;
		color: #9fbab0;
		font-size: 0.72rem;
		text-transform: uppercase;
		letter-spacing: 0.16em;
	}

	nav {
		display: flex;
		flex-wrap: wrap;
		gap: 0.85rem;
	}

	nav a {
		padding: 0.35rem 0 0.2rem;
		color: #c6d3ce;
		border-bottom: 1px solid transparent;
		transition:
			color 140ms ease,
			border-color 140ms ease;
	}

	nav a:hover {
		color: #f2f7f5;
		border-color: rgba(188, 246, 196, 0.65);
	}

	main {
		padding-top: 1.4rem;
	}

	@media (max-width: 760px) {
		.topbar {
			flex-direction: column;
			align-items: start;
		}
	}
</style>
