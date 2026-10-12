import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { vi } from "vitest";
import { test as playwrightTest } from "@playwright/test";
import { NativePage } from "../playwright/nativePage";
import { evaluatePlaywrightVisTests } from "../playwright/visualizationPlaywright.utils";

const registrations = {
    beforeAll: vi.fn(),
    beforeEach: vi.fn(),
    afterAll: vi.fn(),
    afterEach: vi.fn(),
};

describe("native visualization startup scope", () => {
    let temporary: string;

    beforeEach(() => {
        vi.spyOn(playwrightTest, "beforeAll").mockImplementation(registrations.beforeAll);
        vi.spyOn(playwrightTest, "beforeEach").mockImplementation(registrations.beforeEach);
        vi.spyOn(playwrightTest, "afterAll").mockImplementation(registrations.afterAll);
        vi.spyOn(playwrightTest, "afterEach").mockImplementation(registrations.afterEach);
        temporary = fs.mkdtempSync(path.join(os.tmpdir(), "native-hooks-"));
        const config = path.join(temporary, "config.json");
        fs.writeFileSync(config, JSON.stringify({ tests: [] }));
        vi.stubEnv("CONFIG_PATH", config);
        vi.spyOn(console, "log").mockImplementation(() => {});
    });

    afterEach(() => {
        vi.unstubAllEnvs();
        vi.restoreAllMocks();
        vi.clearAllMocks();
        fs.rmSync(temporary, { recursive: true });
    });

    it("fails startup in beforeEach without a secondary cleanup failure", async () => {
        const launch = vi.spyOn(NativePage, "launch").mockRejectedValue(new Error("startup failed"));
        await evaluatePlaywrightVisTests("native");
        expect(registrations.beforeAll).not.toHaveBeenCalled();
        expect(launch).not.toHaveBeenCalled();
        const before = registrations.beforeEach.mock.calls[0][0];
        await expect(before()).rejects.toThrow("startup failed");
        const after = registrations.afterEach.mock.calls[0][0];
        await expect(after({}, {})).resolves.toBeUndefined();
        expect(launch).toHaveBeenCalledOnce();
    });

    it("preserves browser beforeAll initialization", async () => {
        await evaluatePlaywrightVisTests("webgl2");
        expect(registrations.beforeAll).toHaveBeenCalledOnce();
    });
});
