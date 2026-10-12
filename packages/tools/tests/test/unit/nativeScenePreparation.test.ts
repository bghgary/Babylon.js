import { vi } from "vitest";
import { evaluatePrepareScene } from "../playwright/visualizationPlaywright.utils";

vi.mock("@playwright/test", () => ({ test: {}, expect: {} }));
vi.mock("@tools/test-tools", () => ({ getGlobalConfig: vi.fn() }));

const globalConfig = {
    root: "https://cdn.babylonjs.com",
    snippetUrl: "https://snippet.babylonjs.com",
    pgRoot: "https://playground.babylonjs.com",
};

describe("native scene construction retries", () => {
    const attempts = { count: 0 };
    const testWindow = { seed: 0, Math: { random: Math.random }, scene: undefined };
    const sceneMetadata = {
        playgroundId: "#TEST#1",
        snippetSceneCall: "createScene(engine)",
        snippetCode: "function createScene() { globalThis.sceneAttempts.count++; throw new TypeError('permanent scene error'); }",
    };

    beforeEach(() => {
        attempts.count = 0;
        testWindow.scene = undefined;
        vi.useFakeTimers();
        vi.stubGlobal("window", testWindow);
        vi.stubGlobal("engine", {});
        vi.stubGlobal("sceneAttempts", attempts);
        vi.stubGlobal("BABYLON", {
            SceneLoader: { OnPluginActivatedObservable: { clear: vi.fn() } },
        });
        vi.spyOn(console, "error").mockImplementation(() => {});
    });

    afterEach(() => {
        vi.unstubAllGlobals();
        vi.restoreAllMocks();
        vi.useRealTimers();
    });

    it("reports a native construction error after one attempt with no retry timers", async () => {
        vi.stubGlobal("__babylonNativePlaywright", true);
        await expect(evaluatePrepareScene({ sceneMetadata, globalConfig })).rejects.toThrow("permanent scene error");
        expect(attempts.count).toBe(1);
        expect(console.error).toHaveBeenCalledOnce();
        expect(vi.getTimerCount()).toBe(0);
    });

    it("retains the five delayed retries for browser scenes", async () => {
        vi.stubGlobal("__babylonNativePlaywright", false);
        const result = evaluatePrepareScene({ sceneMetadata, globalConfig });
        const assertion = expect(result).rejects.toThrow("permanent scene error");
        await vi.advanceTimersByTimeAsync(2500);
        await assertion;
        expect(attempts.count).toBe(6);
        expect(vi.getTimerCount()).toBe(0);
    });

    it("still awaits asynchronous native scene creation", async () => {
        vi.stubGlobal("__babylonNativePlaywright", true);
        await expect(
            evaluatePrepareScene({
                sceneMetadata: { ...sceneMetadata, snippetCode: "async function createScene() { return { ready: true }; }" },
                globalConfig,
            })
        ).resolves.toBe(true);
        expect(testWindow.scene).toEqual({ ready: true });
    });
});
