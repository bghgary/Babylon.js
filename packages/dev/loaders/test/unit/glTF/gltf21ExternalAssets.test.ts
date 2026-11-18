import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { VertexBuffer } from "core/Buffers/buffer";
import { NullEngine } from "core/Engines/nullEngine";
import { type IFileRequest } from "core/Misc/fileRequest";
import { Observable } from "core/Misc/observable";
import { Tools } from "core/Misc/tools.pure";
import { type Texture } from "core/Materials/Textures/texture";
import { Scene } from "core/scene";
import { FreeCamera } from "core/Cameras/freeCamera";
import { Vector3 } from "core/Maths/math.vector.pure";
import { FlowGraphState } from "core/FlowGraph/flowGraph";
import { GLTFFileLoader } from "loaders/glTF/glTFFileLoader.pure";
import { type IGLTF } from "loaders/glTF/2.0/glTFLoaderInterfaces";
import { GetPathToObjectConverter } from "loaders/glTF/2.0/Extensions/objectModelMapping";
import { GetKHRInteractivityImportResults } from "loaders/glTF/2.0/Extensions/KHR_interactivity";
import { registerGLTFExtension, unregisterGLTFExtension } from "loaders/glTF/2.0/glTFLoaderExtensionRegistry";
import "loaders/glTF/2.0/glTFLoader";
import "loaders/glTF/2.0/Extensions/KHR_lights_punctual";

describe("glTF 2.1 external assets", () => {
    let engine: NullEngine;
    let scene: Scene;

    beforeEach(() => {
        engine = new NullEngine();
        scene = new Scene(engine);
    });

    afterEach(() => {
        unregisterGLTFExtension("TEST_nested_complete");
        vi.restoreAllMocks();
        scene.dispose();
        engine.dispose();
    });

    it("instantiates each referenced scene separately after existing children", async () => {
        const inner = { asset: { version: "2.0" }, scene: 0, scenes: [{ nodes: [0] }], nodes: [{ name: "inner" }] };
        const data = {
            json: {
                asset: { version: "2.1" },
                scene: 0,
                scenes: [{ nodes: [0, 1] }],
                nodes: [{ name: "first", children: [2], externalAsset: 0 }, { name: "second", externalAsset: 0 }, { name: "existing" }],
                externalAssets: [{ file: 0 }],
                files: [{ mimeType: "model/gltf+json", uri: `data:model/gltf+json;base64,${btoa(JSON.stringify(inner))}` }],
            },
        };
        const container = await new GLTFFileLoader().loadAssetContainerAsync(scene, data, "");
        const first = container.transformNodes.find((node) => node.name === "first")!;
        const second = container.transformNodes.find((node) => node.name === "second")!;
        const instances = container.transformNodes.filter((node) => node.name === "inner");
        expect(instances).toHaveLength(2);
        expect(first.getChildren().map((node) => node.name)).toEqual(["existing", "inner"]);
        expect(second.getChildren().map((node) => node.name)).toEqual(["inner"]);
        expect(instances[0]).not.toBe(instances[1]);
    });

    it("runs each nested interactivity graph against its own asset instance", async () => {
        const inner = {
            asset: { version: "2.0" },
            scene: 0,
            scenes: [{ nodes: [0] }],
            nodes: [{ name: "interactive" }],
            extensionsUsed: ["KHR_interactivity"],
            extensions: {
                KHR_interactivity: {
                    graphs: [
                        {
                            types: [{ signature: "float3" }],
                            declarations: [{ op: "event/onStart" }, { op: "pointer/set" }],
                            nodes: [
                                { declaration: 0, flows: { out: { node: 1 } } },
                                {
                                    declaration: 1,
                                    configuration: { pointer: { value: ["/nodes/0/translation"] }, type: { value: [0] } },
                                    values: { value: { type: 0, value: [1, 2, 3] } },
                                },
                            ],
                        },
                    ],
                },
            },
        };
        const data = {
            json: {
                asset: { version: "2.1" },
                scene: 0,
                scenes: [{ nodes: [0, 1] }],
                nodes: [
                    { name: "first", externalAsset: 0 },
                    { name: "second", externalAsset: 0 },
                ],
                externalAssets: [{ file: 0 }],
                files: [{ mimeType: "model/gltf+json", uri: `data:model/gltf+json;base64,${btoa(JSON.stringify(inner))}` }],
            },
        };
        scene.activeCamera = new FreeCamera("camera", Vector3.Zero(), scene);
        const loader = new GLTFFileLoader();
        const complete = loader.whenCompleteAsync();
        await loader.loadAsync(scene, data, "");
        await complete;
        const results = GetKHRInteractivityImportResults(scene);
        expect(results).toHaveLength(2);
        expect(results[0].glTF).not.toBe(results[1].glTF);
        expect(results.map((result) => result.graphs[0].flowGraph?.state)).toEqual([FlowGraphState.Started, FlowGraphState.Started]);
        scene.render();
        const instances = scene.transformNodes.filter((node) => node.name === "interactive");
        expect(instances).toHaveLength(2);
        expect(instances.map((node) => node.position.asArray())).toEqual([
            [1, 2, 3],
            [1, 2, 3],
        ]);
    });

    it("attaches external assets to nodes that also reference meshes", async () => {
        const inner = { asset: { version: "2.0" }, scene: 0, scenes: [{ nodes: [0] }], nodes: [{ name: "inner" }] };
        const data = {
            json: {
                asset: { version: "2.1" },
                scene: 0,
                scenes: [{ nodes: [0, 1] }],
                nodes: [
                    { name: "single", mesh: 0, externalAsset: 0 },
                    { name: "multi", mesh: 1, externalAsset: 0 },
                ],
                meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }, { primitives: [{ attributes: { POSITION: 0 } }, { attributes: { POSITION: 0 } }] }],
                accessors: [{ componentType: 5126, count: 3, type: "VEC3", min: [0, 0, 0], max: [0, 0, 0] }],
                externalAssets: [{ file: 0 }],
                files: [{ mimeType: "model/gltf+json", uri: `data:model/gltf+json;base64,${btoa(JSON.stringify(inner))}` }],
            },
        };
        const container = await new GLTFFileLoader().loadAssetContainerAsync(scene, data, "");
        const single = container.meshes.find((mesh) => mesh.name === "single")!;
        const multi = container.transformNodes.find((node) => node.name === "multi")!;
        expect(single.getChildren().map((node) => node.name)).toEqual(["inner"]);
        expect(multi.getChildren().map((node) => node.name)).toEqual(["multi_primitive0", "multi_primitive1", "inner"]);
        expect(container.meshes).toContain(single);
        expect(container.meshes.filter((mesh) => mesh.name.startsWith("multi_primitive"))).toHaveLength(2);
    });

    it("does not attach a scene when the referenced asset has no default", async () => {
        const inner = { asset: { version: "2.0" }, scenes: [{ nodes: [0] }], nodes: [{ name: "inner" }] };
        const data = {
            json: {
                asset: { version: "2.1" },
                scene: 0,
                scenes: [{ nodes: [0] }],
                nodes: [{ name: "parent", externalAsset: 0 }],
                externalAssets: [{ file: 0 }],
                files: [{ mimeType: "model/gltf+json", uri: `data:model/gltf+json;base64,${btoa(JSON.stringify(inner))}` }],
            },
        };
        const container = await new GLTFFileLoader().loadAssetContainerAsync(scene, data, "");
        expect(container.transformNodes.map((node) => node.name)).toEqual(["parent"]);
    });

    it("rejects an unsupported referenced asset even without a default scene", async () => {
        const inner = { asset: { version: "3.0" } };
        const data = {
            json: {
                asset: { version: "2.1" },
                scene: 0,
                scenes: [{ nodes: [0] }],
                nodes: [{ externalAsset: 0 }],
                externalAssets: [{ file: 0 }],
                files: [{ mimeType: "model/gltf+json", uri: `data:model/gltf+json;base64,${btoa(JSON.stringify(inner))}` }],
            },
        };
        await expect(new GLTFFileLoader().loadAssetContainerAsync(scene, data, "")).rejects.toThrow(/unsupported version/i);
    });

    it("resolves aliases into parent buffer views and propagates visibility across assets", async () => {
        const values = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
        const bytes = new Uint8Array(values.buffer);
        const inner = {
            asset: { version: "2.1" },
            scene: 0,
            scenes: [{ nodes: [0] }],
            nodes: [{ name: "inner", mesh: 0 }],
            meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
            accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: "VEC3", min: [0, 0, 0], max: [1, 1, 0] }],
            bufferViews: [{ buffer: 0, byteLength: bytes.byteLength }],
            buffers: [{ uri: "mesh.bin", byteLength: bytes.byteLength }],
        };
        const data = {
            json: {
                asset: { version: "2.1" },
                scene: 0,
                scenes: [{ nodes: [0] }],
                nodes: [{ name: "parent", visible: false, externalAsset: 0 }],
                externalAssets: [{ file: 0 }],
                files: [
                    { mimeType: "model/gltf+json", uri: `data:model/gltf+json;base64,${btoa(JSON.stringify(inner))}`, aliases: [{ alias: "mesh.bin", file: 1 }] },
                    { mimeType: "application/gltf-buffer", bufferView: 0 },
                ],
                bufferViews: [{ buffer: 0, byteLength: bytes.byteLength }],
                buffers: [{ byteLength: bytes.byteLength }],
            },
            bin: { byteLength: bytes.byteLength, readAsync: async (offset: number, length: number) => bytes.subarray(offset, offset + length) },
        };
        const container = await new GLTFFileLoader().loadAssetContainerAsync(scene, data, "");
        const mesh = container.meshes.find((entry) => entry.name === "inner")!;
        expect(Array.from(mesh.getVerticesData(VertexBuffer.PositionKind)!)).toEqual(Array.from(values));
        expect(mesh.isVisible).toBe(false);
        const visibility = GetPathToObjectConverter(data.json as IGLTF).convert("/nodes/0/visible");
        visibility.info.set!(true, visibility.object);
        expect(mesh.isVisible).toBe(true);
        visibility.info.set!(false, visibility.object);
        expect(mesh.isVisible).toBe(false);
        expect(container.geometries).toContain(mesh.geometry);
    });

    it("uses the aliased file's media type for an image", async () => {
        const inner = {
            asset: { version: "2.0" },
            scene: 0,
            scenes: [{ nodes: [0] }],
            nodes: [{ mesh: 0 }],
            meshes: [{ primitives: [{ attributes: { POSITION: 0 }, material: 0 }] }],
            materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0 } } }],
            textures: [{ source: 0 }],
            images: [{ uri: "image.png" }],
            accessors: [{ componentType: 5126, count: 3, type: "VEC3", min: [0, 0, 0], max: [0, 0, 0] }],
        };
        const data = {
            json: {
                asset: { version: "2.1" },
                scene: 0,
                scenes: [{ nodes: [0] }],
                nodes: [{ externalAsset: 0 }],
                externalAssets: [{ file: 0 }],
                files: [
                    { mimeType: "model/gltf+json", uri: `data:model/gltf+json;base64,${btoa(JSON.stringify(inner))}`, aliases: [{ alias: "image.png", file: 1 }] },
                    { mimeType: "image/webp", uri: `data:image/webp;base64,${btoa("image bytes")}` },
                ],
            },
        };
        const container = await new GLTFFileLoader().loadAssetContainerAsync(scene, data, "");
        expect(container.textures).toHaveLength(1);
        expect((container.textures[0] as Texture).mimeType).toBe("image/webp");
    });

    it("uses the aliased file's media type for a nested external asset", async () => {
        const leaf = new TextEncoder().encode(
            JSON.stringify({
                asset: { version: "2.1" },
                scene: 0,
                scenes: [{ nodes: [0] }],
                nodes: [{ name: "leaf", externalAsset: 0 }],
                externalAssets: [{ file: 0 }],
                files: [{ mimeType: "model/gltf+json", uri: "final.gltf" }],
            })
        );
        const paddedLength = Math.ceil(leaf.length / 4) * 4;
        const glb = new Uint8Array(20 + paddedLength);
        const view = new DataView(glb.buffer);
        view.setUint32(0, 0x46546c67, true);
        view.setUint32(4, 2, true);
        view.setUint32(8, glb.length, true);
        view.setUint32(12, paddedLength, true);
        view.setUint32(16, 0x4e4f534a, true);
        glb.fill(0x20, 20);
        glb.set(leaf, 20);
        const inner = {
            asset: { version: "2.1" },
            scene: 0,
            scenes: [{ nodes: [0] }],
            nodes: [{ externalAsset: 0 }],
            externalAssets: [{ file: 0 }],
            files: [{ mimeType: "model/gltf+json", uri: "leaf.gltf" }],
        };
        const data = {
            json: {
                asset: { version: "2.1" },
                scene: 0,
                scenes: [{ nodes: [0] }],
                nodes: [{ externalAsset: 0 }],
                externalAssets: [{ file: 0 }],
                files: [
                    { mimeType: "model/gltf+json", uri: `data:model/gltf+json;base64,${btoa(JSON.stringify(inner))}`, aliases: [{ alias: "leaf.gltf", file: 1 }] },
                    { mimeType: "model/gltf-binary", bufferView: 0, aliases: [{ alias: "final.gltf", file: 2 }] },
                    {
                        mimeType: "model/gltf+json",
                        uri: `data:model/gltf+json;base64,${btoa(JSON.stringify({ asset: { version: "2.0" }, scene: 0, scenes: [{ nodes: [0] }], nodes: [{ name: "final" }] }))}`,
                    },
                ],
                bufferViews: [{ buffer: 0, byteLength: glb.length }],
                buffers: [{ byteLength: glb.length }],
            },
            bin: { byteLength: glb.length, readAsync: async (offset: number, length: number) => glb.subarray(offset, offset + length) },
        };
        const container = await new GLTFFileLoader().loadAssetContainerAsync(scene, data, "");
        expect(container.transformNodes.some((node) => node.name === "leaf")).toBe(true);
        expect(container.transformNodes.some((node) => node.name === "final")).toBe(true);
    });

    it("loads an embedded binary asset from a buffer view", async () => {
        const inner = new TextEncoder().encode(JSON.stringify({ asset: { version: "2.0" }, scene: 0, scenes: [{ nodes: [0] }], nodes: [{ name: "binary" }] }));
        const length = Math.ceil(inner.length / 4) * 4;
        const bytes = new Uint8Array(20 + length);
        const view = new DataView(bytes.buffer);
        view.setUint32(0, 0x46546c67, true);
        view.setUint32(4, 2, true);
        view.setUint32(8, bytes.length, true);
        view.setUint32(12, length, true);
        view.setUint32(16, 0x4e4f534a, true);
        bytes.fill(0x20, 20);
        bytes.set(inner, 20);
        const data = {
            json: {
                asset: { version: "2.1" },
                scene: 0,
                scenes: [{ nodes: [0] }],
                nodes: [{ externalAsset: 0 }],
                externalAssets: [{ file: 0 }],
                files: [{ mimeType: "model/gltf-binary", bufferView: 0 }],
                bufferViews: [{ buffer: 0, byteLength: bytes.length }],
                buffers: [{ byteLength: bytes.length }],
            },
            bin: { byteLength: bytes.byteLength, readAsync: async (offset: number, count: number) => bytes.subarray(offset, offset + count) },
        };
        const container = await new GLTFFileLoader().loadAssetContainerAsync(scene, data, "");
        expect(container.transformNodes.some((node) => node.name === "binary")).toBe(true);
    });

    it("loads an embedded JSON asset from an explicitly referenced chunk", async () => {
        const inner = new TextEncoder().encode(JSON.stringify({ asset: { version: "2.0" }, scene: 0, scenes: [{ nodes: [0] }], nodes: [{ name: "chunk" }] }));
        const data = {
            json: {
                asset: { version: "2.1" },
                scene: 0,
                scenes: [{ nodes: [0] }],
                nodes: [{ externalAsset: 0 }],
                externalAssets: [{ file: 0 }],
                files: [{ mimeType: "model/gltf+json", bufferView: 0 }],
                bufferViews: [{ buffer: 0, byteLength: inner.length }],
                buffers: [{ chunk: 1, byteLength: inner.length }],
            },
            bin: null,
            chunks: [
                { byteLength: 0, readAsync: async () => new Uint8Array() },
                { byteLength: inner.length, readAsync: async (offset: number, count: number) => inner.subarray(offset, offset + count) },
            ],
        };
        const container = await new GLTFFileLoader().loadAssetContainerAsync(scene, data, "");
        expect(container.transformNodes.some((node) => node.name === "chunk")).toBe(true);
    });

    it("returns separate meshes and materials for each instance without adding them to the scene", async () => {
        const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
        const inner = {
            asset: { version: "2.0" },
            scene: 0,
            scenes: [{ nodes: [0] }],
            nodes: [{ name: "inner", mesh: 0 }],
            meshes: [{ primitives: [{ attributes: { POSITION: 0 }, material: 0 }] }],
            materials: [{ name: "material" }],
            accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: "VEC3", min: [0, 0, 0], max: [1, 1, 0] }],
            bufferViews: [{ buffer: 0, byteLength: positions.byteLength }],
            buffers: [{ uri: `data:application/octet-stream;base64,${Buffer.from(positions.buffer).toString("base64")}`, byteLength: positions.byteLength }],
        };
        const data = {
            json: {
                asset: { version: "2.1" },
                scene: 0,
                scenes: [{ nodes: [0, 1] }],
                nodes: [{ externalAsset: 0 }, { externalAsset: 0 }],
                externalAssets: [{ file: 0 }],
                files: [{ mimeType: "model/gltf+json", uri: `data:model/gltf+json;base64,${btoa(JSON.stringify(inner))}` }],
            },
        };
        const container = await new GLTFFileLoader().loadAssetContainerAsync(scene, data, "");
        const meshes = container.meshes.filter((mesh) => mesh.name === "inner");
        expect(meshes).toHaveLength(2);
        expect(meshes[0].material).not.toBe(meshes[1].material);
        expect(container.materials).toContain(meshes[0].material);
        expect(container.materials).toContain(meshes[1].material);
        expect(scene.meshes).toHaveLength(0);
        expect(scene.materials).toHaveLength(0);
    });

    it("resolves resources relative to an external asset's location", async () => {
        const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
        const inner = {
            asset: { version: "2.0" },
            scene: 0,
            scenes: [{ nodes: [0] }],
            nodes: [{ name: "inner", mesh: 0 }],
            meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
            accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: "VEC3", min: [0, 0, 0], max: [1, 1, 0] }],
            bufferViews: [{ buffer: 0, byteLength: positions.byteLength }],
            buffers: [{ uri: "mesh.bin", byteLength: positions.byteLength }],
        };
        const payloads = new Map<string, Uint8Array>([
            ["https://example.test/other/inner.gltf", new TextEncoder().encode(JSON.stringify(inner))],
            ["https://example.test/other/mesh.bin", new Uint8Array(positions.buffer)],
        ]);
        const requests: string[] = [];
        vi.spyOn(GLTFFileLoader.prototype, "_loadFile").mockImplementation((_scene, url, success) => {
            requests.push(String(url));
            const payload = payloads.get(String(url));
            if (!payload) {
                throw new Error(`Unexpected request: ${url}`);
            }
            success(payload.slice().buffer);
            return { abort: () => {}, onCompleteObservable: new Observable<IFileRequest>() };
        });
        const data = {
            json: {
                asset: { version: "2.1" },
                scene: 0,
                scenes: [{ nodes: [0] }],
                nodes: [{ externalAsset: 0 }],
                externalAssets: [{ file: 0 }],
                files: [{ mimeType: "model/gltf+json", uri: "https://example.test/other/inner.gltf" }],
            },
        };
        const container = await new GLTFFileLoader().loadAssetContainerAsync(scene, data, "https://example.test/root/");
        expect(container.meshes.some((mesh) => mesh.name === "inner")).toBe(true);
        expect(requests).toEqual(["https://example.test/other/inner.gltf", "https://example.test/other/mesh.bin"]);
    });

    it("rejects a recursive reference to the current asset", async () => {
        const data = {
            json: {
                asset: { version: "2.1" },
                scene: 0,
                scenes: [{ nodes: [0] }],
                nodes: [{ externalAsset: 0 }],
                externalAssets: [{ file: 0 }],
                files: [{ mimeType: "model/gltf+json", uri: "./root.gltf" }],
            },
        };
        await expect(new GLTFFileLoader().loadAssetContainerAsync(scene, data, "https://example.test/", undefined, "root.gltf")).rejects.toThrow(/recursive external asset/i);
    });

    it("keeps an external asset's lights in the parent visibility hierarchy", async () => {
        const inner = {
            asset: { version: "2.0" },
            extensionsUsed: ["KHR_lights_punctual"],
            extensions: { KHR_lights_punctual: { lights: [{ type: "point" }] } },
            scene: 0,
            scenes: [{ nodes: [0] }],
            nodes: [{ extensions: { KHR_lights_punctual: { light: 0 } } }],
        };
        const data = {
            json: {
                asset: { version: "2.1" },
                scene: 0,
                scenes: [{ nodes: [0] }],
                nodes: [{ visible: false, externalAsset: 0 }],
                externalAssets: [{ file: 0 }],
                files: [{ mimeType: "model/gltf+json", uri: `data:model/gltf+json;base64,${btoa(JSON.stringify(inner))}` }],
            },
        };
        const loader = new GLTFFileLoader();
        const extensions: string[] = [];
        loader.onExtensionLoaded = (extension) => extensions.push(extension.name);
        const container = await loader.loadAssetContainerAsync(scene, data, "");
        expect(container.lights).toHaveLength(1);
        expect(extensions).toContain("KHR_lights_punctual");
        expect(container.lights[0].isEnabled()).toBe(false);
        const visibility = GetPathToObjectConverter(data.json as IGLTF).convert("/nodes/0/visible");
        visibility.info.set!(true, visibility.object);
        expect(container.lights[0].isEnabled()).toBe(true);
    });

    it("does not inherit aliases into a further nested asset", async () => {
        const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
        const nested = {
            asset: { version: "2.0" },
            scene: 0,
            scenes: [{ nodes: [0] }],
            nodes: [{ name: "nested", mesh: 0 }],
            meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
            accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: "VEC3", min: [0, 0, 0], max: [1, 1, 0] }],
            bufferViews: [{ buffer: 0, byteLength: positions.byteLength }],
            buffers: [{ uri: "mesh.bin", byteLength: positions.byteLength }],
        };
        const child = {
            asset: { version: "2.1" },
            scene: 0,
            scenes: [{ nodes: [0] }],
            nodes: [{ externalAsset: 0 }],
            externalAssets: [{ file: 0 }],
            files: [{ mimeType: "model/gltf+json", uri: `data:model/gltf+json;base64,${btoa(JSON.stringify(nested))}` }],
        };
        const data = {
            json: {
                asset: { version: "2.1" },
                scene: 0,
                scenes: [{ nodes: [0] }],
                nodes: [{ externalAsset: 0 }],
                externalAssets: [{ file: 0 }],
                files: [
                    { mimeType: "model/gltf+json", uri: `data:model/gltf+json;base64,${btoa(JSON.stringify(child))}`, aliases: [{ alias: "mesh.bin", file: 1 }] },
                    { mimeType: "application/gltf-buffer", uri: "unexpected.bin" },
                ],
            },
        };
        const requests: string[] = [];
        vi.spyOn(GLTFFileLoader.prototype, "_loadFile").mockImplementation((_scene, url, success) => {
            requests.push(String(url));
            if (url !== "https://example.test/mesh.bin") {
                throw new Error(`Unexpected request: ${url}`);
            }
            success(positions.buffer);
            return { abort: () => {}, onCompleteObservable: new Observable<IFileRequest>() };
        });
        const container = await new GLTFFileLoader().loadAssetContainerAsync(scene, data, "https://example.test/");
        expect(container.meshes.some((mesh) => mesh.name === "nested")).toBe(true);
        expect(requests).toEqual(["https://example.test/mesh.bin"]);
    });

    it("waits for nested completion before reporting parent completion", async () => {
        let release!: () => void;
        const pending = new Promise<void>((resolve) => {
            release = resolve;
        });
        registerGLTFExtension("TEST_nested_complete", false, (loader) => ({
            name: "TEST_nested_complete",
            enabled: true,
            onReady: () => {
                if (loader.gltf.asset.generator === "nested") {
                    loader._completePromises.push(pending);
                }
            },
        }));
        const inner = { asset: { version: "2.0", generator: "nested" }, scene: 0, scenes: [{ nodes: [0] }], nodes: [{}] };
        const data = {
            json: {
                asset: { version: "2.1" },
                scene: 0,
                scenes: [{ nodes: [0] }],
                nodes: [{ externalAsset: 0 }],
                externalAssets: [{ file: 0 }],
                files: [{ mimeType: "model/gltf+json", uri: `data:model/gltf+json;base64,${btoa(JSON.stringify(inner))}` }],
            },
        };
        const loader = new GLTFFileLoader();
        let completed = false;
        const completion = loader.whenCompleteAsync().then(() => {
            completed = true;
        });
        await loader.loadAssetContainerAsync(scene, data, "");
        await new Promise<void>((resolve) => Tools.SetImmediate(resolve));
        expect(completed).toBe(false);
        release();
        await completion;
        expect(completed).toBe(true);
    });

    it("reports a nested error after the asset is ready", async () => {
        let fail!: (reason: Error) => void;
        const pending = new Promise<void>((_resolve, reject) => {
            fail = reject;
        });
        registerGLTFExtension("TEST_nested_complete", false, (loader) => ({
            name: "TEST_nested_complete",
            enabled: true,
            onReady: () => {
                if (loader.gltf.asset.generator === "nested") {
                    loader._completePromises.push(pending);
                }
            },
        }));
        const inner = { asset: { version: "2.0", generator: "nested" }, scene: 0, scenes: [{ nodes: [0] }], nodes: [{}] };
        const data = {
            json: {
                asset: { version: "2.1" },
                scene: 0,
                scenes: [{ nodes: [0] }],
                nodes: [{ externalAsset: 0 }],
                externalAssets: [{ file: 0 }],
                files: [{ mimeType: "model/gltf+json", uri: `data:model/gltf+json;base64,${btoa(JSON.stringify(inner))}` }],
            },
        };
        const loader = new GLTFFileLoader();
        const completion = loader.whenCompleteAsync();
        await loader.loadAssetContainerAsync(scene, data, "");
        await new Promise<void>((resolve) => Tools.SetImmediate(resolve));
        fail(new Error("nested completion failed"));
        await expect(completion).rejects.toThrow("nested completion failed");
    });
});
