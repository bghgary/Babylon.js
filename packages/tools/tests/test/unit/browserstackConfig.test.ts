import { execSync } from "child_process";
import { version } from "playwright/package.json";
import { vi } from "vitest";
import { getCdpEndpoint } from "../../browserstack.config";

vi.mock("child_process", async (importOriginal) => ({
    ...(await importOriginal<typeof import("child_process")>()),
    execSync: vi.fn(() => {
        throw new Error("Configuration must not start a subprocess.");
    }),
}));

vi.mock("browserstack-local", () => ({ default: { Local: class {} } }));

describe("BrowserStack configuration", () => {
    it("uses the installed Playwright version without invoking npx", () => {
        const endpoint = new URL(getCdpEndpoint("chrome@latest:OSX Sonoma", "native configuration regression"));
        const caps = JSON.parse(endpoint.searchParams.get("caps")!);
        expect(caps["client.playwrightVersion"]).toBe(version);
        expect(caps.browser).toBe("chrome");
        expect(caps.os).toBe("OSX");
        expect(execSync).not.toHaveBeenCalled();
    });
});
