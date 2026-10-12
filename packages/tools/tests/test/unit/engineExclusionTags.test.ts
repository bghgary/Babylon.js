import { getActiveEngineExclusionTags, isEngineExcluded, resolveNativeRunIdentity } from "../playwright/engineExclusionTags";

describe("native exclusion tags", () => {
    const windows = getActiveEngineExclusionTags("native", { platform: "win32", graphicsApi: "d3d11" });
    const linux = getActiveEngineExclusionTags("native", { platform: "unix", graphicsApi: "vk" });

    it("combines all-native, API, platform and compound identity", () => {
        expect(windows).toEqual(["native", "native-d3d11", "native-win32", "native-win32-d3d11"]);
    });

    it("matches exclusions with OR semantics", () => {
        expect(isEngineExcluded(["native-d3d11", "native-vk"], windows)).toBe(true);
        expect(isEngineExcluded(["native-d3d11", "native-vk"], linux)).toBe(true);
        expect(isEngineExcluded(["native"], linux)).toBe(true);
        expect(isEngineExcluded(["webgpu", "native-gl"], windows)).toBe(false);
    });

    it("keeps compound exclusions specific to their platform and API", () => {
        expect(isEngineExcluded(["native-win32-d3d11"], windows)).toBe(true);
        expect(isEngineExcluded(["native-win32-d3d11"], linux)).toBe(false);
        expect(isEngineExcluded(["native-win32-vk"], windows)).toBe(false);
    });

    it("normalizes declared platform and API aliases", () => {
        expect(resolveNativeRunIdentity({ BABYLON_NATIVE_PLATFORM: "Windows", BABYLON_NATIVE_GRAPHICS_API: "DX11" })).toEqual({
            platform: "win32",
            graphicsApi: "d3d11",
        });
    });

    it("does not invent backend identity when none was supplied", () => {
        expect(getActiveEngineExclusionTags("native", resolveNativeRunIdentity({}))).toEqual(["native"]);
        expect(isEngineExcluded(undefined, windows)).toBe(false);
    });

    it("does not apply native exclusions to browser projects", () => {
        expect(getActiveEngineExclusionTags("webgl2")).toEqual(["webgl2"]);
        expect(isEngineExcluded(["native", "native-d3d11"], getActiveEngineExclusionTags("webgl2"))).toBe(false);
    });
});
