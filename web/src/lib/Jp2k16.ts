export type Jp2k16DecodeResult = {
	width: number;
	height: number;
	data: Uint8Array | Uint16Array | Int16Array;
	bitsPerSample: number;
	isSigned: boolean;
	components: number;
};

export class Jp2k16Decoder {
	private worker: Worker;
	private callbacks: Map<
		number,
		{ resolve: (value: Jp2k16DecodeResult) => void; reject: (error: unknown) => void }
	> = new Map();
	private nextId = 0;

	constructor() {
		this.worker = new Worker(new URL('./jp2k16.worker.ts', import.meta.url), {
			type: 'module',
		});

		this.worker.onmessage = (event: MessageEvent) => {
			const { id, result, error } = event.data as {
				id: number;
				result?: Jp2k16DecodeResult;
				error?: string;
			};
			const callback = this.callbacks.get(id);
			if (!callback) return;
			this.callbacks.delete(id);
			if (error) callback.reject(new Error(error));
			else if (!result) callback.reject(new Error('Decode failed'));
			else callback.resolve(result);
		};

		this.worker.onerror = (event) => {
			const error = new Error(`JP2K worker error: ${event.message || 'unknown error'}`);
			for (const { reject } of this.callbacks.values()) reject(error);
			this.callbacks.clear();
		};
	}

	decode(buffer: ArrayBuffer): Promise<Jp2k16DecodeResult> {
		return new Promise((resolve, reject) => {
			const id = this.nextId++;
			this.callbacks.set(id, { resolve, reject });
			this.worker.postMessage({ id, buffer }, [buffer]);
		});
	}
}
