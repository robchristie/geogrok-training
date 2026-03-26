/// <reference lib="webworker" />

type KakaduModuleFactory = (opts: {
	locateFile?: (path: string, prefix?: string) => string;
}) => Promise<unknown>;

type DecodeJob = { id: number; buffer: ArrayBuffer };

let moduleInstance: Record<string, unknown> | null = null;

const KAKADU_JS_URL = '/kakadujs/kakadujs.js';
const KAKADU_WASM_URL = '/kakadujs/kakadujs.wasm';

async function loadKakaduModule(): Promise<Record<string, unknown>> {
	if (moduleInstance) return moduleInstance;

	let mod: Record<string, unknown>;
	try {
		mod = (await import(/* @vite-ignore */ KAKADU_JS_URL)) as Record<string, unknown>;
	} catch {
		throw new Error(`Missing Kakadu WASM artifacts at ${KAKADU_JS_URL} and ${KAKADU_WASM_URL}.`);
	}

	const factory =
		(mod.default as KakaduModuleFactory | undefined) ??
		(mod.KakaduModule as KakaduModuleFactory | undefined) ??
		(mod.createModule as KakaduModuleFactory | undefined);
	if (!factory) {
		throw new Error('Failed to load Kakadu WASM module factory.');
	}

	moduleInstance = (await factory({
		locateFile: (path: string) => {
			if (path.endsWith('.wasm')) return KAKADU_WASM_URL;
			return new URL(path, KAKADU_JS_URL).toString();
		},
	})) as Record<string, unknown>;
	return moduleInstance;
}

function copySamples(
	samples: Uint8Array | Uint16Array | Int16Array,
): Uint8Array | Uint16Array | Int16Array {
	if (samples instanceof Uint8Array) return new Uint8Array(samples);
	if (samples instanceof Uint16Array) return new Uint16Array(samples);
	if (samples instanceof Int16Array) return new Int16Array(samples);
	throw new Error('Unsupported sample type');
}

self.onmessage = async (event: MessageEvent<DecodeJob>) => {
	const { id, buffer } = event.data;

	try {
		const mod = await loadKakaduModule();
		const DecoderCtor = mod.HTJ2KDecoder as
			| (new () => {
					getEncodedBuffer(length: number): Uint8Array;
					readHeader(): void;
					decode(): void;
					getFrameInfo(): {
						width: number;
						height: number;
						bitsPerSample: number;
						componentCount: number;
						isSigned: boolean;
					};
					getDecodedBuffer(): Uint8Array;
			  })
			| undefined;
		if (!DecoderCtor) {
			throw new Error('HTJ2KDecoder is not available in the Kakadu WASM module.');
		}

		const decoder = new DecoderCtor();
		const encodedView = decoder.getEncodedBuffer(buffer.byteLength);
		encodedView.set(new Uint8Array(buffer));
		decoder.readHeader();
		decoder.decode();

		const info = decoder.getFrameInfo();
		const decodedBytes = decoder.getDecodedBuffer();
		const bytesPerSample = info.bitsPerSample <= 8 ? 1 : 2;
		const sampleCount = info.width * info.height * info.componentCount;

		let samples: Uint8Array | Uint16Array | Int16Array;
		if (bytesPerSample === 1) {
			samples = decodedBytes.subarray(0, sampleCount);
		} else {
			if (decodedBytes.byteOffset % 2 !== 0) {
				throw new Error(`Unexpected decoded buffer alignment: ${decodedBytes.byteOffset}`);
			}
			samples = info.isSigned
				? new Int16Array(decodedBytes.buffer, decodedBytes.byteOffset, sampleCount)
				: new Uint16Array(decodedBytes.buffer, decodedBytes.byteOffset, sampleCount);
		}

		const copied = copySamples(samples);
		self.postMessage(
			{
				id,
				result: {
					width: info.width,
					height: info.height,
					data: copied,
					bitsPerSample: info.bitsPerSample,
					isSigned: info.isSigned,
					components: info.componentCount,
				},
			},
			[copied.buffer],
		);
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		self.postMessage({ id, error: message });
	}
};
