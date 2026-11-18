import { expect, test } from "@playwright/test";
import { getGlobalConfig } from "@tools/test-tools";

declare const BABYLON: typeof import("core/index") & typeof import("loaders/index");

test.describe("glTF 2.1 WebP textures", () => {
    for (const storage of ["uri", "bufferView"] as const) {
        test(`loads a WebP image from a ${storage}`, async ({ page }) => {
            await page.setContent('<canvas id="babylon-canvas"></canvas>');
            await page.addScriptTag({ url: `${getGlobalConfig().baseUrl}/babylon.js` });
            await page.addScriptTag({ url: `${getGlobalConfig().baseUrl}/loaders/babylonjs.loaders.min.js` });
            const result = await page.evaluate(async (imageStorage) => {
                const uri = "data:image/webp;base64,UklGRhwAAABXRUJQVlA4TA8AAAAvAUAAAAcQ/Y/+ByKi/wEA";
                const byteLength = atob(uri.substring(uri.indexOf(",") + 1)).length;
                const gltf = {
                    asset: { version: "2.1", minVersion: "2.1" },
                    scene: 0,
                    scenes: [{ nodes: [0] }],
                    nodes: [{ mesh: 0 }],
                    meshes: [{ primitives: [{ attributes: { POSITION: 0, TEXCOORD_0: 1 }, material: 0 }] }],
                    accessors: [
                        { componentType: 5126, count: 3, type: "VEC3", min: [0, 0, 0], max: [0, 0, 0] },
                        { componentType: 5126, count: 3, type: "VEC2" },
                    ],
                    materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0 } } }],
                    textures: [{ source: 0 }],
                    images: [imageStorage === "uri" ? { uri } : { bufferView: 0, mimeType: "image/webp" }],
                    ...(imageStorage === "bufferView"
                        ? { buffers: [{ uri: uri.replace("data:image/webp;", "data:application/octet-stream;"), byteLength }], bufferViews: [{ buffer: 0, byteLength }] }
                        : {}),
                };
                const canvas = document.getElementById("babylon-canvas") as HTMLCanvasElement;
                const engine = new BABYLON.Engine(canvas);
                const scene = new BABYLON.Scene(engine);
                try {
                    const loaded = await BABYLON.ImportMeshAsync(`data:${JSON.stringify(gltf)}`, scene);
                    const material = loaded.meshes[1].material as InstanceType<typeof BABYLON.PBRMaterial>;
                    const texture = material.albedoTexture as InstanceType<typeof BABYLON.Texture>;
                    const pixels = await texture.readPixels();
                    if (!pixels) {
                        throw new Error("Unable to read the loaded texture");
                    }
                    const bytes = new Uint8Array(pixels.buffer, pixels.byteOffset, pixels.byteLength);
                    return { size: texture.getSize(), pixel: Array.from(bytes.slice(0, 4)) };
                } finally {
                    scene.dispose();
                    engine.dispose();
                }
            }, storage);
            expect(result.size).toEqual({ width: 2, height: 2 });
            expect(result.pixel[0]).toBeGreaterThan(240);
            expect(result.pixel[1]).toBeLessThan(15);
            expect(result.pixel[2]).toBeLessThan(15);
            expect(result.pixel[3]).toBe(255);
        });
    }
});
