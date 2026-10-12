import { spawn, type ChildProcessWithoutNullStreams } from "child_process";
import { EventEmitter } from "events";
import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";

const protocolPrefix = "PW\t";
const startupTimeout = 30_000;
const shutdownTimeout = 5_000;
type PageFunction<Arg, R> = string | ((arg: Arg) => R | Promise<R>);

interface PendingRequest {
    resolve: (payload: Buffer) => void;
    reject: (error: Error) => void;
    timeout?: NodeJS.Timeout;
}

export class NativeConsoleMessage {
    public constructor(
        private readonly messageType: string,
        private readonly messageText: string
    ) {}

    public type(): string {
        return this.messageType;
    }

    public text(): string {
        return this.messageText;
    }

    public toString(): string {
        return this.messageText;
    }
}

export class NativePage {
    private readonly events = new EventEmitter();
    private readonly pending = new Map<number, PendingRequest>();
    private nextRequestId = 1;
    private defaultTimeout = 30_000;
    private exited = false;
    private closed = false;
    private aborted = false;
    private exitError?: Error;
    private terminalError?: Error;
    private closePromise?: Promise<void>;
    private readonly exitPromise: Promise<void>;

    private constructor(private readonly process: ChildProcessWithoutNullStreams) {
        this.exitPromise = new Promise((resolve) => {
            process.once("close", () => {
                this.exited = true;
                this.closed = true;
                resolve();
            });
        });
        this.initializeExitHandler();
    }

    public static async launch(): Promise<NativePage> {
        const executable = process.env.BABYLON_NATIVE_PLAYGROUND;
        if (!executable) {
            throw new Error("BABYLON_NATIVE_PLAYGROUND must point to a BabylonNative Playground executable built with Playwright bridge support.");
        }
        if (!fs.existsSync(executable)) {
            throw new Error(`BABYLON_NATIVE_PLAYGROUND does not exist: ${executable}`);
        }

        const child = spawn(executable, ["--playwright"], {
            cwd: path.dirname(executable),
            stdio: ["pipe", "pipe", "pipe"],
            windowsHide: true,
        });
        const page = new NativePage(child);
        try {
            await page.waitUntilReady();
            return page;
        } catch (error) {
            page.abort(error instanceof Error ? error : new Error(String(error)));
            try {
                await page.close();
            } catch (cleanupError) {
                throw new AggregateError([error, cleanupError], "BabylonNative startup and cleanup failed.");
            }
            throw error;
        }
    }

    public async evaluate<R, Arg>(pageFunction: PageFunction<Arg, R>, arg: Arg): Promise<R>;
    public async evaluate<R>(pageFunction: PageFunction<void, R>, arg?: unknown): Promise<R>;
    public async evaluate<R, Arg>(pageFunction: PageFunction<Arg, R>, arg?: Arg): Promise<R> {
        const serializedArg = arg === undefined ? "undefined" : JSON.stringify(arg);
        if (serializedArg === undefined) {
            throw new Error("NativePage.evaluate only supports JSON-serializable arguments.");
        }

        const invocation = typeof pageFunction === "string" ? `(0, eval)(${JSON.stringify(pageFunction)})` : `(${pageFunction.toString()})(${serializedArg})`;
        const source = `(async () => { const value = await ${invocation}; return JSON.stringify({ value }); })()`;
        const payload = await this.request("EVALUATE", [Buffer.from(source).toString("base64")]);
        const envelope = JSON.parse(payload.toString("utf8"));
        return envelope.value;
    }

    public async setViewportSize(viewportSize: { width: number; height: number }): Promise<void> {
        if (!Number.isInteger(viewportSize.width) || viewportSize.width <= 0 || !Number.isInteger(viewportSize.height) || viewportSize.height <= 0) {
            throw new Error(`Invalid native viewport size: ${viewportSize.width}x${viewportSize.height}`);
        }
        await this.request("RESIZE", [viewportSize.width.toString(), viewportSize.height.toString()]);
    }

    public async goto(_url: string, _options?: { timeout?: number }): Promise<null> {
        return null;
    }

    public async waitForSelector(selector: string, _options?: { timeout?: number }): Promise<null> {
        if (selector !== "#babylon-canvas") {
            throw new Error(`NativePage has no DOM and cannot wait for selector "${selector}".`);
        }
        return null;
    }

    public async waitForFunction<R, Arg>(pageFunction: PageFunction<Arg, R>, arg?: Arg): Promise<void> {
        const deadline = this.defaultTimeout === 0 ? Number.POSITIVE_INFINITY : Date.now() + this.defaultTimeout;
        const serializedArg = arg === undefined ? "undefined" : JSON.stringify(arg);
        if (serializedArg === undefined) {
            throw new Error("NativePage.waitForFunction only supports JSON-serializable arguments.");
        }
        const invocation = typeof pageFunction === "string" ? `(0, eval)(${JSON.stringify(pageFunction)})` : `(${pageFunction.toString()})(${serializedArg})`;
        const source = `(async () => JSON.stringify({ value: !!(await ${invocation}) }))()`;
        while (!JSON.parse((await this.request("EVALUATE", [Buffer.from(source).toString("base64")])).toString("utf8")).value) {
            if (Date.now() >= deadline) {
                throw new Error(`NativePage.waitForFunction timed out after ${this.defaultTimeout}ms.`);
            }
            await new Promise((resolve) => setTimeout(resolve, 50));
        }
    }

    public setDefaultTimeout(timeout: number): void {
        if (!Number.isFinite(timeout) || timeout < 0) {
            throw new Error(`Invalid native page timeout: ${timeout}`);
        }
        this.defaultTimeout = timeout;
    }

    public on(event: "console", listener: (message: NativeConsoleMessage) => void): this {
        this.events.on(event, listener);
        return this;
    }

    public off(event: "console", listener: (message: NativeConsoleMessage) => void): this {
        this.events.off(event, listener);
        return this;
    }

    public screenshot(): Promise<Buffer> {
        return this.request("SCREENSHOT");
    }

    public abort(error: Error): void {
        this.aborted = true;
        this.failPendingRequests(error);
        if (!this.exited && this.process.pid) {
            this.process.kill("SIGKILL");
        }
    }

    public async close(): Promise<void> {
        if (this.closed) {
            this.throwIfShutdownFailed();
            return;
        }
        this.closePromise ??= this.closeProcess();
        await this.closePromise;
    }

    private async closeProcess(): Promise<void> {
        this.failPendingRequests(new Error("BabylonNative is closing."));
        // The host's stdin EOF handler posts WM_CLOSE without needing a JS-thread round trip.
        this.process.stdin.end();
        if (await this.waitForExit(shutdownTimeout)) {
            this.throwIfShutdownFailed();
            return;
        }
        if (this.process.pid) {
            this.process.kill("SIGKILL");
        }
        if (!(await this.waitForExit(shutdownTimeout))) {
            throw new Error("BabylonNative did not exit after forced termination.");
        }
        throw new Error(`BabylonNative did not close within ${shutdownTimeout}ms and was terminated.`);
    }

    private throwIfShutdownFailed(): void {
        if (!this.aborted && this.exitError) {
            throw this.exitError;
        }
    }

    private async waitForExit(timeout: number): Promise<boolean> {
        let timer: NodeJS.Timeout | undefined;
        try {
            return await Promise.race([
                this.exitPromise.then(() => true),
                new Promise<boolean>((resolve) => {
                    timer = setTimeout(() => resolve(false), timeout);
                }),
            ]);
        } finally {
            clearTimeout(timer);
        }
    }

    private async waitUntilReady(): Promise<void> {
        const lines = readline.createInterface({ input: this.process.stdout });
        lines.on("line", (line) => this.handleLine(line));
        this.process.stderr.on("data", (chunk: Buffer) => process.stderr.write(`[BabylonNative] ${chunk.toString("utf8")}`));

        await new Promise<void>((resolve, reject) => {
            const timeout = setTimeout(() => reject(new Error(`BabylonNative did not become ready within ${startupTimeout}ms.`)), startupTimeout);
            const onReady = () => {
                clearTimeout(timeout);
                this.events.off("exit", onExit);
                resolve();
            };
            const onExit = (error: Error) => {
                clearTimeout(timeout);
                this.events.off("ready", onReady);
                reject(error);
            };
            this.events.once("ready", onReady);
            this.events.once("exit", onExit);
        });
    }

    private handleLine(line: string): void {
        if (!line.startsWith(protocolPrefix)) {
            return;
        }

        const fields = line.split("\t");
        if (fields[1] === "0" && fields[2] === "READY") {
            this.events.emit("ready");
            return;
        }
        if (fields[1] === "0" && fields[2] === "CONSOLE" && fields.length === 5) {
            this.events.emit("console", new NativeConsoleMessage(fields[3], Buffer.from(fields[4], "base64").toString("utf8")));
            return;
        }

        const id = Number(fields[1]);
        const pending = this.pending.get(id);
        if (!pending) {
            return;
        }
        this.pending.delete(id);
        if (pending.timeout) {
            clearTimeout(pending.timeout);
        }

        const payload = Buffer.from(fields[3] ?? "", "base64");
        if (fields[2] === "OK") {
            pending.resolve(payload);
        } else {
            pending.reject(new Error(payload.toString("utf8") || `BabylonNative request ${id} failed.`));
        }
    }

    private request(method: string, fields: string[] = []): Promise<Buffer> {
        if (this.terminalError || this.exited) {
            return Promise.reject(this.terminalError ?? new Error("BabylonNative exited before the request could be sent."));
        }

        const id = this.nextRequestId++;
        return new Promise<Buffer>((resolve, reject) => {
            const pending: PendingRequest = { resolve, reject };
            const requestTimeout = this.defaultTimeout || 100_000;
            if (requestTimeout > 0) {
                pending.timeout = setTimeout(() => {
                    this.abort(new Error(`BabylonNative ${method} request timed out after ${requestTimeout}ms.`));
                }, requestTimeout);
            }
            this.pending.set(id, pending);
            this.process.stdin.write([protocolPrefix.slice(0, -1), id, method, ...fields].join("\t") + "\n");
        });
    }

    private failPendingRequests(error: Error): void {
        this.terminalError ??= error;
        for (const pending of this.pending.values()) {
            if (pending.timeout) {
                clearTimeout(pending.timeout);
            }
            pending.reject(this.terminalError);
        }
        this.pending.clear();
        this.events.emit("exit", this.terminalError);
    }

    private initializeExitHandler(): void {
        this.process.once("exit", (code, signal) => {
            this.exited = true;
            const detail = signal ? `signal ${signal}` : `exit code ${code ?? "unknown"}`;
            const error = new Error(`BabylonNative exited with ${detail}.`);
            if (code !== 0 || signal) {
                this.exitError = error;
            }
            this.failPendingRequests(error);
        });
        this.process.once("error", (error) => this.failPendingRequests(error));
        this.process.stdin.once("error", (error) => this.abort(error));
    }
}
