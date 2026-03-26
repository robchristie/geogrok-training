import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

const extraAllowedHosts = (process.env.VITE_ALLOWED_HOSTS ?? '')
	.split(',')
	.map((host) => host.trim())
	.filter(Boolean);

const allowedHosts = Array.from(new Set(['lv426.yutani.tech', ...extraAllowedHosts]));

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		host: '127.0.0.1',
		port: 4173,
		allowedHosts,
		proxy: {
			'/api': 'http://127.0.0.1:8787',
			'/health': 'http://127.0.0.1:8787',
		},
	},
});
