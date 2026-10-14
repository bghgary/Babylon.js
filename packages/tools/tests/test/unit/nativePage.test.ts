import { ChildProcess, spawn } from "child_process";
import { PassThrough } from "stream";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { vi } from "vitest";
import { NativePage } from "../playwright/nativePage";

vi.mock("child_process", async (importOriginal) => ({
    ...(await importOriginal<typeof import("child_process")>()),
    spawn: vi.fn(),
}));

describe("NativePage lifecycle", () => {
    let child: ReturnType<typeof makeChild>;
    let requests: string[];

    function makeChild() {
        return Object.assign(new ChildProcess(), {
            pid: 12345,
            stdin: new PassThrough(),
            stdout: new PassThrough(),
            stderr: new PassThrough(),
            kill: vi.fn(() => true),
        });
    }

    async function launch() {
        const result = NativePage.launch();
        child.stdout.write("PW\t0\tREADY\n");
        return result;
    }

    function finish(code: number | null = 0, signal: NodeJS.Signals | null = null) {
        child.emit("exit", code, signal);
        child.stdout.end();
        child.stderr.end();
        child.emit("close", code, signal);
    }

    function reply(id: number, payload: string | Buffer, status = "OK") {
        child.stdout.write(`PW\t${id}\t${status}\t${Buffer.from(payload).toString("base64")}\n`);
    }

    beforeEach(() => {
        vi.useFakeTimers();
        vi.stubEnv("BABYLON_NATIVE_PLAYGROUND", process.execPath);
        child = makeChild();
        requests = [];
        child.stdin.on("data", (data: Buffer) => requests.push(data.toString()));
        vi.mocked(spawn).mockReturnValue(child);
    });

    afterEach(() => {
        finish();
        child.stdin.destroy();
        vi.unstubAllEnvs();
        vi.useRealTimers();
        vi.clearAllMocks();
    });

    it("waits for READY and transports evaluation and screenshots", async () => {
        const page = await launch();
        const evaluated = page.evaluate(() => 42);
        expect(requests[0]).toMatch(/^PW\t1\tEVALUATE\t/);
        reply(1, '{"value":42}');
        await expect(evaluated).resolves.toBe(42);
        const screenshot = page.screenshot();
        reply(2, Buffer.from([137, 80, 78, 71]));
        await expect(screenshot).resolves.toEqual(Buffer.from([137, 80, 78, 71]));
        finish();
        await page.close();
    });

    it("keeps the transport usable after a remote JavaScript error", async () => {
        const page = await launch();
        const assertion = expect(page.evaluate("bad()")).rejects.toThrow("scene failed");
        reply(1, "scene failed", "ERROR");
        await assertion;
        const next = page.evaluate("1");
        reply(2, '{"value":1}');
        await expect(next).resolves.toBe(1);
    });

    it("forwards native console messages", async () => {
        const page = await launch();
        const listener = vi.fn();
        page.on("console", listener);
        child.stdout.write(`PW\t0\tCONSOLE\terror\t${Buffer.from("native error").toString("base64")}\n`);
        expect(listener).toHaveBeenCalledOnce();
        expect(listener.mock.calls[0][0].text()).toBe("native error");
        page.off("console", listener);
    });

    it("rejects pending and subsequent requests after a crash", async () => {
        const page = await launch();
        const pending = expect(page.evaluate("1")).rejects.toThrow("exit code 3");
        finish(3);
        await pending;
        await expect(page.screenshot()).rejects.toThrow("exit code 3");
        await expect(page.close()).rejects.toThrow("exit code 3");
        expect(requests).toHaveLength(1);
    });

    it("closes concurrently and idempotently using stdin EOF", async () => {
        const page = await launch();
        const first = page.close();
        const second = page.close();
        expect(child.stdin.writableEnded).toBe(true);
        expect(requests).toEqual([]);
        finish();
        await Promise.all([first, second, page.close()]);
        expect(child.kill).not.toHaveBeenCalled();
    });

    it("cancels outstanding requests when closing", async () => {
        const page = await launch();
        const pending = expect(page.screenshot()).rejects.toThrow("closing");
        const closing = page.close();
        finish();
        await pending;
        await closing;
        await expect(page.evaluate("1")).rejects.toThrow("closing");
    });

    it("waits for streams to close, not just process exit", async () => {
        const page = await launch();
        let settled = false;
        const closing = page.close().then(() => {
            settled = true;
        });
        child.emit("exit", 0, null);
        await vi.advanceTimersByTimeAsync(100);
        expect(settled).toBe(false);
        child.emit("close", 0, null);
        await closing;
    });

    it("reports a nonzero exit during normal shutdown", async () => {
        const page = await launch();
        const closing = expect(page.close()).rejects.toThrow("exit code 1");
        finish(1);
        await closing;
    });

    it("force-kills and reports an unresponsive shutdown", async () => {
        const page = await launch();
        child.kill.mockImplementation(() => {
            finish(null, "SIGKILL");
            return true;
        });
        const closing = expect(page.close()).rejects.toThrow("did not close within 5000ms");
        await vi.advanceTimersByTimeAsync(5000);
        await closing;
        expect(child.kill).toHaveBeenCalledWith("SIGKILL");
    });

    it("bounds the wait even if forced termination never produces close", async () => {
        const page = await launch();
        const closing = expect(page.close()).rejects.toThrow("did not exit after forced termination");
        await vi.advanceTimersByTimeAsync(10000);
        await closing;
    });

    it("kills a host that never announces READY", async () => {
        child.kill.mockImplementation(() => {
            finish(null, "SIGKILL");
            return true;
        });
        const startup = expect(NativePage.launch()).rejects.toThrow("did not become ready within 30000ms");
        await vi.advanceTimersByTimeAsync(30000);
        await startup;
        expect(child.kill).toHaveBeenCalledOnce();
    });

    it("reports a spawn failure without waiting for the startup deadline", async () => {
        const startup = expect(NativePage.launch()).rejects.toThrow("spawn failed");
        child.emit("error", new Error("spawn failed"));
        child.emit("close", -1, null);
        await startup;
    });

    it("aborts all outstanding requests on timeout and ignores late responses", async () => {
        const page = await launch();
        page.setDefaultTimeout(20);
        const first = expect(page.evaluate("1")).rejects.toThrow("timed out after 20ms");
        const second = expect(page.screenshot()).rejects.toThrow("timed out after 20ms");
        await vi.advanceTimersByTimeAsync(20);
        await Promise.all([first, second]);
        reply(1, '{"value":1}');
        await expect(page.evaluate("2")).rejects.toThrow("timed out after 20ms");
        expect(requests).toHaveLength(2);
        expect(child.kill).toHaveBeenCalledOnce();
        finish(null, "SIGKILL");
        await page.close();
    });

    it("preserves the initial abort reason without masking it with SIGKILL", async () => {
        const page = await launch();
        const pending = expect(page.evaluate("1")).rejects.toThrow("original failure");
        page.abort(new Error("original failure"));
        page.abort(new Error("later failure"));
        finish(null, "SIGKILL");
        await pending;
        await page.close();
        await expect(page.screenshot()).rejects.toThrow("original failure");
    });

    it("aborts on stdin errors", async () => {
        const page = await launch();
        const pending = expect(page.evaluate("1")).rejects.toThrow("broken pipe");
        child.stdin.emit("error", new Error("broken pipe"));
        await pending;
        expect(child.kill).toHaveBeenCalledWith("SIGKILL");
    });

    it("rejects invalid viewport and timeout values before sending commands", async () => {
        const page = await launch();
        await expect(page.setViewportSize({ width: 0, height: 400 })).rejects.toThrow("Invalid native viewport");
        expect(() => page.setDefaultTimeout(-1)).toThrow("Invalid native page timeout");
        expect(requests).toEqual([]);
    });
});

describe("NativePage external transport", () => {
    let directory: string;
    let filename: string;

    beforeEach(() => {
        directory = fs.mkdtempSync(path.join(os.tmpdir(), "native-page-transport-"));
        filename = path.join(directory, "transport.cjs");
        fs.writeFileSync(
            filename,
            `
            const state = { requests: [], closed: 0, aborted: undefined };
            module.exports = {
                state,
                createTransport: async (onConsole, signal) => {
                    state.signal = signal;
                    state.console = onConsole;
                    return {
                        request: async (method, fields, timeoutMs) => {
                            state.requests.push({ method, fields, timeoutMs });
                            if (method === "SCREENSHOT") { return Buffer.from([137, 80, 78, 71]); }
                            return Buffer.from('{"value":42}');
                        },
                        abort: error => { state.aborted = error.message; },
                        close: async () => { state.closed++; }
                    };
                }
            };
        `
        );
        vi.stubEnv("BABYLON_NATIVE_TRANSPORT", filename);
    });

    afterEach(() => {
        vi.unstubAllEnvs();
        delete require.cache[filename];
        fs.rmSync(filename);
        fs.rmdirSync(directory);
    });

    function fixture(): {
        state: {
            requests: { method: string; fields: string[]; timeoutMs: number }[];
            closed: number;
            aborted?: string;
            signal: AbortSignal;
            console: (level: string, text: string) => void;
        };
    } {
        return require(filename);
    }

    it("uses the existing evaluation, viewport, screenshot and console APIs", async () => {
        const page = await NativePage.launch();
        expect(await page.evaluate(() => 42)).toBe(42);
        await page.setViewportSize({ width: 600, height: 400 });
        expect(await page.screenshot()).toEqual(Buffer.from([137, 80, 78, 71]));
        const listener = vi.fn();
        page.on("console", listener);
        fixture().state.console("error", "native error");
        expect(listener.mock.calls[0][0].text()).toBe("native error");
        const requests = fixture().state.requests;
        expect(requests.map((request) => request.method)).toEqual(["EVALUATE", "RESIZE", "SCREENSHOT"]);
        expect(Buffer.from(requests[0].fields[0], "base64").toString()).toContain("JSON.stringify({ value })");
        expect(requests[1].fields).toEqual(["600", "400"]);
        expect(fixture().state.signal).toBeInstanceOf(AbortSignal);
        await page.close();
    });

    it("passes command deadlines and closes exactly once after abort", async () => {
        const page = await NativePage.launch();
        page.setDefaultTimeout(123);
        await page.screenshot();
        expect(fixture().state.requests[0].timeoutMs).toBe(123);
        page.abort(new Error("failed scene"));
        await Promise.all([page.close(), page.close()]);
        expect(fixture().state.aborted).toBe("failed scene");
        expect(fixture().state.closed).toBe(1);
        await expect(page.evaluate("1")).rejects.toThrow("failed scene");
    });

    it("rejects an invalid factory or startup deadline instead of falling back to desktop", async () => {
        fs.writeFileSync(filename, "module.exports = {};");
        await expect(NativePage.launch()).rejects.toThrow("must export createTransport");
        delete require.cache[filename];
        fs.writeFileSync(filename, "module.exports = {createTransport: async () => { throw new Error('must not launch'); }};");
        vi.stubEnv("BABYLON_NATIVE_STARTUP_TIMEOUT_MS", "-1");
        await expect(NativePage.launch()).rejects.toThrow("must be a positive integer");
    });
});
