/**
 * Exclusion tags for visualization config `excludedEngines`.
 *
 * Web (unchanged): "webgl1" | "webgl2" | "webgpu" | ...
 *
 * Native (granular, combinable — OR semantics like today):
 *   "native"              → every native backend/platform
 *   "native-d3d11"        → D3D11 only
 *   "native-d3d12"        → D3D12 only
 *   "native-vk"           → Vulkan only
 *   "native-gl"           → desktop OpenGL only
 *   "native-gles"         → OpenGL ES only
 *   "native-metal"        → Metal only
 *   "native-win32" | "native-uwp" | "native-android" | "native-ios" | "native-osx" | "native-unix"
 *   optional compounds    → "native-win32-d3d11" (platform + backend both active)
 *
 * Examples:
 *   excludedEngines: ["native"]
 *   excludedEngines: ["native-d3d11", "native-vk"]
 *   excludedEngines: ["webgpu", "native-gles"]
 *   excludedEngines: ["native-android-gles", "native-ios-metal"]
 */

export type NativeGraphicsApi = "d3d11" | "d3d12" | "vk" | "gl" | "gles" | "metal" | "webgpu" | "noop" | "unknown";
export type NativePlatform = "win32" | "uwp" | "android" | "ios" | "osx" | "unix" | "unknown";

export interface NativeRunIdentity {
    graphicsApi: NativeGraphicsApi;
    platform: NativePlatform;
}

const graphicsApiAliases: Record<string, NativeGraphicsApi> = {
    d3d11: "d3d11",
    dx11: "d3d11",
    direct3d11: "d3d11",
    d3d12: "d3d12",
    dx12: "d3d12",
    direct3d12: "d3d12",
    vk: "vk",
    vulkan: "vk",
    gl: "gl",
    opengl: "gl",
    gles: "gles",
    opengles: "gles",
    es: "gles",
    metal: "metal",
    mtl: "metal",
    webgpu: "webgpu",
    wgpu: "webgpu",
    noop: "noop",
    none: "noop",
};

const platformAliases: Record<string, NativePlatform> = {
    win32: "win32",
    windows: "win32",
    win: "win32",
    uwp: "uwp",
    winrt: "uwp",
    android: "android",
    ios: "ios",
    osx: "osx",
    macos: "osx",
    mac: "osx",
    unix: "unix",
    linux: "unix",
    x11: "unix",
};

function normalizeToken(value: string | undefined | null): string {
    return (value ?? "").trim().toLowerCase().replace(/[_\s]+/g, "-");
}

export function normalizeNativeGraphicsApi(value: string | undefined | null): NativeGraphicsApi {
    const key = normalizeToken(value).replace(/^native-/, "");
    return graphicsApiAliases[key] ?? (key ? "unknown" : "unknown");
}

export function normalizeNativePlatform(value: string | undefined | null): NativePlatform {
    const key = normalizeToken(value).replace(/^native-/, "");
    return platformAliases[key] ?? (key ? "unknown" : "unknown");
}

/**
 * Resolve native run identity from env (CI/local).
 *   BABYLON_NATIVE_GRAPHICS_API = d3d11 | d3d12 | vk | gl | gles | metal | ...
 *   BABYLON_NATIVE_PLATFORM     = win32 | uwp | android | ios | osx | unix | ...
 */
export function resolveNativeRunIdentity(env: NodeJS.ProcessEnv = process.env): NativeRunIdentity {
    return {
        graphicsApi: normalizeNativeGraphicsApi(env.BABYLON_NATIVE_GRAPHICS_API),
        platform: normalizeNativePlatform(env.BABYLON_NATIVE_PLATFORM),
    };
}

/**
 * Active tags for the current Playwright engine project / native host.
 * A test is skipped when any `excludedEngines` entry is in this set.
 */
export function getActiveEngineExclusionTags(engineType: string, nativeIdentity?: NativeRunIdentity): string[] {
    const type = engineType.toLowerCase();
    if (type !== "native") {
        return [type];
    }

    const identity = nativeIdentity ?? resolveNativeRunIdentity();
    const tags = new Set<string>(["native"]);

    if (identity.graphicsApi !== "unknown") {
        tags.add(`native-${identity.graphicsApi}`);
    }
    if (identity.platform !== "unknown") {
        tags.add(`native-${identity.platform}`);
    }
    if (identity.graphicsApi !== "unknown" && identity.platform !== "unknown") {
        // Compound: exclude only this platform+backend pairing
        tags.add(`native-${identity.platform}-${identity.graphicsApi}`);
    }

    return [...tags];
}

export function isEngineExcluded(excludedEngines: string[] | undefined, activeTags: string[]): boolean {
    if (!excludedEngines || excludedEngines.length === 0) {
        return false;
    }
    const active = new Set(activeTags.map((t) => t.toLowerCase()));
    return excludedEngines.some((tag) => active.has(String(tag).toLowerCase()));
}
