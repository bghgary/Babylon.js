import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { Vector3 } from "core/Maths/math.vector";
import { TransformNode } from "core/Meshes/transformNode";
import { FreeCamera } from "core/Cameras/freeCamera";
import { PointLight } from "core/Lights/pointLight";
import { RectAreaLight } from "core/Lights/rectAreaLight";
import { CreateBox } from "core/Meshes/Builders/boxBuilder";
import { Mesh } from "core/Meshes/mesh";
import { SubMesh } from "core/Meshes/subMesh";
import { PBRMaterial } from "core/Materials/PBR/pbrMaterial";
import { Material } from "core/Materials/material";
import { MultiMaterial } from "core/Materials/multiMaterial";
import { ImportMeshAsync } from "core/Loading";
import { type IAccessor, type ICamera, type IAnimationChannel } from "babylonjs-gltf2interface";
import { GLTF2Export } from "serializers/glTF/2.0";
import { DataWriter } from "serializers/glTF/2.0/dataWriter";
import { GLTFLoader } from "loaders/glTF/2.0/glTFLoader.pure";
import { GLTFFileLoader } from "loaders/glTF/glTFFileLoader.pure";
import { type IGLTF as IRuntimeGLTF } from "loaders/glTF/2.0/glTFLoaderInterfaces";
import { GetGLTFNodeVisibility, GetGLTFVisibilityAnimationTarget } from "loaders/glTF/2.0/glTFLoaderVisibility";
import { GetGLTFBoundingVolume } from "loaders/glTF/2.0/glTFBoundingVolume";
import { GetGLTFNodeUniqueIds, GetGLTFValueByUniqueId } from "loaders/glTF/2.0/glTFUniqueIds";
import { GetPathToObjectConverter } from "loaders/glTF/2.0/Extensions/objectModelMapping";
import "loaders/glTF/2.0/Extensions/KHR_animation_pointer";
import "loaders/glTF/2.0/Extensions/KHR_animation_pointer.data";
import {
    CreateTestScene,
    CreateTriangleGltf,
    ExportGltfAsync,
    ExportGltfWithBinaryAsync,
    GetBlobFile,
    GetTextFile,
    ImportGltfAsync,
    ParseGltfJson,
    ReadAccessor,
    RoundTripGltfAsync,
    RoundTripGltfWithBinaryAsync,
    ToDataUri,
} from "./glTFTestUtils";

function CreateVisibilityAnimationGltf() {
    const source = CreateTriangleGltf();
    source.asset = { version: "2.1", minVersion: "2.1" };
    source.extensionsUsed = ["KHR_animation_pointer"];
    source.nodes = [
        { name: "parent", visible: true, children: [1] },
        { name: "triangle", mesh: 0 },
    ];
    const times = new Float32Array([0, 1, 2]);
    const values = new Uint8Array([1, 0, 1]);
    source.buffers.push({ byteLength: 15, uri: ToDataUri(times, values) });
    source.bufferViews.push({ buffer: 1, byteOffset: 0, byteLength: 12 }, { buffer: 1, byteOffset: 12, byteLength: 3 });
    source.accessors.push(
        { bufferView: 3, componentType: 5126, count: 3, type: "SCALAR" as IAccessor["type"], min: [0], max: [2] },
        { bufferView: 4, componentType: 5121, count: 3, type: "SCALAR" as IAccessor["type"] }
    );
    source.animations = [
        {
            samplers: [{ input: 3, output: 4, interpolation: "STEP" }],
            channels: [{ sampler: 0, target: { path: "pointer" as IAnimationChannel["target"]["path"], extensions: { KHR_animation_pointer: { pointer: "/nodes/0/visible" } } } }],
        },
    ];
    return source;
}

describe("glTF exporter", () => {
    let testScene: ReturnType<typeof CreateTestScene>;

    beforeEach(() => {
        testScene = CreateTestScene();
    });

    afterEach(() => {
        testScene.dispose();
    });

    describe("output containers", () => {
        it("writes a .gltf file that references a separate .bin file", async () => {
            CreateBox("box", {}, testScene.scene);

            const data = await GLTF2Export.GLTFAsync(testScene.scene, "model.gltf", { exportWithoutWaitingForScene: true });
            const gltf = ParseGltfJson(GetTextFile(data.files["model.gltf"]));

            expect(gltf.asset.version).toBe("2.0");
            expect(gltf.asset.minVersion).toBeUndefined();
            expect(gltf.buffers).toHaveLength(1);
            expect(gltf.buffers[0].uri).toBe("model.bin");
            expect(GetBlobFile(data.files["model.bin"]).size).toBe(gltf.buffers[0].byteLength);
        });

        it("writes glTF 2.1 only when requested", async () => {
            CreateBox("box", {}, testScene.scene);

            const defaultGltf = await ExportGltfAsync(testScene.scene, { gltfVersion: undefined });
            const gltf = await ExportGltfAsync(testScene.scene, { gltfVersion: "2.1" });

            expect(defaultGltf.asset.version).toBe("2.0");
            expect(gltf.asset).toMatchObject({ version: "2.1", minVersion: "2.1" });
        });

        it("rejects unsupported export versions", async () => {
            // @ts-expect-error Unsupported versions can be supplied by JavaScript callers.
            await expect(GLTF2Export.GLTFAsync(testScene.scene, "model", { gltfVersion: "3.0" })).rejects.toThrow("Unsupported glTF export version: 3.0");
            // @ts-expect-error Unsupported versions can be supplied by JavaScript callers.
            await expect(GLTF2Export.GLBAsync(testScene.scene, "model", { glbVersion: 4 })).rejects.toThrow("Unsupported binary glTF container version: 4");
        });

        it("writes glTF 2.1 in a binary container", async () => {
            CreateBox("box", {}, testScene.scene);
            const data = await GLTF2Export.GLBAsync(testScene.scene, "model", { gltfVersion: "2.1", exportWithoutWaitingForScene: true });
            const glb = new DataView(await GetBlobFile(data.files["model.glb"]).arrayBuffer());
            const jsonLength = glb.getUint32(12, true);
            const gltf = ParseGltfJson(new TextDecoder().decode(new Uint8Array(glb.buffer, 20, jsonLength)));

            expect(glb.getUint32(4, true)).toBe(2);
            expect(gltf.asset).toMatchObject({ version: "2.1", minVersion: "2.1" });
        });

        it("writes and loads a version 3 binary container", async () => {
            CreateBox("box-é", {}, testScene.scene);
            const data = await GLTF2Export.GLBAsync(testScene.scene, "model", { gltfVersion: "2.1", glbVersion: 3, exportWithoutWaitingForScene: true });
            const bytes = new Uint8Array(await GetBlobFile(data.files["model.glb"]).arrayBuffer());
            const glb = new DataView(bytes.buffer);

            expect(glb.getUint32(0, true)).toBe(0x46546c67);
            expect(glb.getUint32(4, true)).toBe(3);
            expect(glb.getBigUint64(8, true)).toBe(BigInt(bytes.length));
            expect(glb.getUint32(16, true)).toBe(0x4e4f534a);
            expect(glb.getUint32(20, true)).toBe(0);
            const jsonLength = Number(glb.getBigUint64(24, true));
            const jsonEnd = 32 + jsonLength;
            expect(jsonEnd % 8).not.toBe(0);
            const gltf = ParseGltfJson(new TextDecoder().decode(bytes.subarray(32, jsonEnd)));
            expect(gltf.asset).toMatchObject({ version: "2.1", minVersion: "2.1" });
            expect(gltf.nodes.some((node) => node.name === "box-é")).toBe(true);
            expect(gltf.buffers[0].chunk).toBe(1);

            const binHeader = Math.ceil(jsonEnd / 8) * 8;
            expect(Array.from(bytes.subarray(jsonEnd, binHeader)).every((byte) => byte === 0)).toBe(true);
            expect(glb.getUint32(binHeader, true)).toBe(0x004e4942);
            expect(glb.getUint32(binHeader + 4, true)).toBe(0);
            expect(Number(glb.getBigUint64(binHeader + 8, true))).toBe(gltf.buffers[0].byteLength);
            expect(binHeader + 16 + gltf.buffers[0].byteLength).toBe(bytes.length);

            const imported = CreateTestScene();
            try {
                const result = await ImportMeshAsync(bytes, imported.scene, { pluginExtension: ".glb" });
                expect(result.meshes.some((mesh) => mesh.name === "box-é" && mesh.getTotalVertices() > 0)).toBe(true);
            } finally {
                imported.dispose();
            }
        });

        it("writes a version 3 container for a glTF 2.0 asset without a chunk property", async () => {
            CreateBox("box", {}, testScene.scene);
            const data = await GLTF2Export.GLBAsync(testScene.scene, "model", { glbVersion: 3, exportWithoutWaitingForScene: true });
            const bytes = new Uint8Array(await GetBlobFile(data.files["model.glb"]).arrayBuffer());
            const glb = new DataView(bytes.buffer);
            const jsonLength = Number(glb.getBigUint64(24, true));
            const gltf = ParseGltfJson(new TextDecoder().decode(bytes.subarray(32, 32 + jsonLength)));

            expect(gltf.asset.version).toBe("2.0");
            expect(gltf.buffers[0].chunk).toBeUndefined();
            const imported = CreateTestScene();
            try {
                const result = await ImportMeshAsync(bytes, imported.scene, { pluginExtension: ".glb" });
                expect(result.meshes.some((mesh) => mesh.name === "box" && mesh.getTotalVertices() > 0)).toBe(true);
            } finally {
                imported.dispose();
            }
        });

        it("writes version 3 containers without buffer data", async () => {
            const data = await GLTF2Export.GLBAsync(testScene.scene, "model", { gltfVersion: "2.1", glbVersion: 3, exportWithoutWaitingForScene: true });
            const bytes = new Uint8Array(await GetBlobFile(data.files["model.glb"]).arrayBuffer());
            const glb = new DataView(bytes.buffer);
            const jsonEnd = 32 + Number(glb.getBigUint64(24, true));
            const gltf = ParseGltfJson(new TextDecoder().decode(bytes.subarray(32, jsonEnd)));
            const binHeader = Math.ceil(jsonEnd / 8) * 8;
            expect(gltf.buffers).toBeUndefined();
            expect(glb.getBigUint64(binHeader + 8, true)).toBe(0n);
            expect(binHeader + 16).toBe(bytes.length);
        });

        it("writes safe 64-bit lengths in little-endian order", () => {
            const writer = new DataWriter(8);
            writer.writeUInt64(0x100000005);
            expect(Array.from(writer.getOutputData())).toEqual([5, 0, 0, 0, 1, 0, 0, 0]);
            expect(() => writer.writeUInt64(Number.MAX_SAFE_INTEGER + 1)).toThrow();
        });

        it("exports a thumbnail in glTF and both binary container versions", async () => {
            const bytes = Uint8Array.from(atob("UklGRhwAAABXRUJQVlA4TA8AAAAvAUAAAAcQ/Y/+ByKi/wEA"), (character) => character.charCodeAt(0));
            const thumbnail = new Blob([bytes], { type: "image/webp" });
            const options = { gltfVersion: "2.1" as const, thumbnail, exportWithoutWaitingForScene: true };
            const data = await GLTF2Export.GLTFAsync(testScene.scene, "model", options);
            const gltf = ParseGltfJson(GetTextFile(data.files["model.gltf"]));
            const image = gltf.images?.[gltf.asset.thumbnail!];
            if (!image?.uri) {
                throw new Error("Expected an external thumbnail");
            }
            expect(new Uint8Array(await GetBlobFile(data.files[image.uri]).arrayBuffer())).toEqual(bytes);

            for (const glbVersion of [2, 3] as const) {
                const binary = await GLTF2Export.GLBAsync(testScene.scene, "model", { ...options, glbVersion });
                const glb = new Uint8Array(await GetBlobFile(binary.files["model.glb"]).arrayBuffer());
                const loaded = await GLTFLoader.LoadThumbnailAsync(testScene.scene, glb, "");
                expect(loaded?.mimeType).toBe("image/webp");
                expect(loaded?.data).toEqual(bytes);
            }
        });

        it("rejects thumbnails for glTF 2.0 and unsupported image types", async () => {
            const png = new Blob([new Uint8Array([1])], { type: "image/png" });
            const invalid = new Blob([new Uint8Array([1])], { type: "image/gif" });
            await expect(GLTF2Export.GLTFAsync(testScene.scene, "model", { thumbnail: png, exportWithoutWaitingForScene: true })).rejects.toThrow(/thumbnail.*2.1/i);
            await expect(GLTF2Export.GLTFAsync(testScene.scene, "model", { gltfVersion: "2.1", thumbnail: invalid, exportWithoutWaitingForScene: true })).rejects.toThrow(
                /thumbnail.*image\/gif/i
            );
        });

        it("writes a binary glTF with a JSON chunk followed by a BIN chunk", async () => {
            CreateBox("box", {}, testScene.scene);

            const data = await GLTF2Export.GLBAsync(testScene.scene, "model", { exportWithoutWaitingForScene: true });
            expect(Object.keys(data.files)).toEqual(["model.glb"]);

            const glb = new DataView(await GetBlobFile(data.files["model.glb"]).arrayBuffer());
            expect(glb.getUint32(0, true)).toBe(0x46546c67); // "glTF"
            expect(glb.getUint32(4, true)).toBe(2);
            expect(glb.getUint32(8, true)).toBe(glb.byteLength);

            const jsonLength = glb.getUint32(12, true);
            expect(glb.getUint32(16, true)).toBe(0x4e4f534a); // "JSON"
            expect(jsonLength % 4).toBe(0);

            const binHeader = 20 + jsonLength;
            const binLength = glb.getUint32(binHeader, true);
            expect(glb.getUint32(binHeader + 4, true)).toBe(0x004e4942); // "BIN\0"
            expect(binLength % 4).toBe(0);
            expect(binHeader + 8 + binLength).toBe(glb.byteLength);

            const gltf = ParseGltfJson(new TextDecoder().decode(new Uint8Array(glb.buffer, 20, jsonLength)));
            expect(gltf.buffers[0].uri).toBeUndefined();
            expect(gltf.buffers[0].byteLength).toBeLessThanOrEqual(binLength);
        });
    });

    describe("scene graph", () => {
        it("preserves node hierarchy, names and transforms of an imported asset", async () => {
            const source = CreateTriangleGltf();
            source.nodes = [
                { name: "parent", translation: [1, 2, 3], rotation: [0, 0.38268343, 0, 0.92387953], scale: [2, 2, 2], children: [1, 2] },
                { name: "child", mesh: 0, translation: [0, 1, 0] },
                { name: "empty", scale: [1, 3, 1] },
            ];

            const gltf = await RoundTripGltfAsync(testScene.scene, source);

            const parent = gltf.nodes.find((n) => n.name === "parent");
            const child = gltf.nodes.find((n) => n.name === "child");
            const empty = gltf.nodes.find((n) => n.name === "empty");
            if (!parent || !child || !empty || !parent.children) {
                throw new Error("Expected the exported node hierarchy");
            }
            expect(gltf.scenes[0].nodes).toEqual([gltf.nodes.indexOf(parent)]);
            expect(parent.children.map((i) => gltf.nodes[i].name).sort()).toEqual(["child", "empty"]);
            expect(parent.translation).toEqual([1, 2, 3]);
            expect(parent.scale).toEqual([2, 2, 2]);
            if (!parent.rotation) {
                throw new Error("Expected the parent rotation");
            }
            expect(parent.rotation[1]).toBeCloseTo(0.38268343);
            expect(parent.rotation[3]).toBeCloseTo(0.92387953);
            expect(child.translation).toEqual([0, 1, 0]);
            expect(empty.scale).toEqual([1, 3, 1]);
            expect(child.mesh).toBe(0);
        });

        it("exports only the nodes accepted by shouldExportNode", async () => {
            const kept = new TransformNode("kept", testScene.scene);
            const skipped = new TransformNode("skipped", testScene.scene);
            skipped.position = new Vector3(1, 0, 0);

            const gltf = await ExportGltfAsync(testScene.scene, { shouldExportNode: (node) => node !== skipped });

            expect(gltf.nodes.map((n) => n.name)).toEqual([kept.name]);
        });

        it("exports explicitly selected hierarchical node visibility", async () => {
            const hidden = new TransformNode("hidden", testScene.scene);
            const child = CreateBox("child", {}, testScene.scene);
            child.parent = hidden;
            const { gltf, bin } = await ExportGltfWithBinaryAsync(testScene.scene, {
                gltfVersion: "2.1",
                nodeVisibility: (node) => (node === hidden ? false : node === child ? true : undefined),
            });

            const exportedParent = gltf.nodes.find((node) => node.name === "hidden");
            const exportedChild = gltf.nodes.find((node) => node.name === "child");
            expect(exportedParent?.visible).toBe(false);
            expect(exportedChild?.visible).toBe(true);
            expect(exportedParent?.children?.map((index) => gltf.nodes[index].name)).toContain("child");

            const imported = CreateTestScene();
            try {
                gltf.buffers[0].uri = ToDataUri(new Uint8Array(bin));
                await ImportGltfAsync(imported.scene, gltf);
                expect(imported.scene.getMeshByName("child")?.isVisible).toBe(false);
            } finally {
                imported.dispose();
            }
        });

        it("rejects node visibility when exporting glTF 2.0", async () => {
            await expect(ExportGltfAsync(testScene.scene, { nodeVisibility: () => false })).rejects.toThrow(/visibility.*2.1/i);
        });

        it("rejects invalid authored node visibility", async () => {
            new TransformNode("node", testScene.scene);
            // @ts-expect-error JavaScript callers can supply a non-boolean value.
            await expect(ExportGltfAsync(testScene.scene, { gltfVersion: "2.1", nodeVisibility: () => null })).rejects.toThrow(/invalid visibility/i);
        });

        it("round trips loaded local visibility independently of inherited visibility", async () => {
            const source = CreateTriangleGltf();
            source.asset = { version: "2.1", minVersion: "2.1" };
            source.nodes = [
                { name: "parent", visible: false, children: [1] },
                { name: "triangle", mesh: 0, visible: true },
            ];
            const gltf = await RoundTripGltfAsync(testScene.scene, source, { gltfVersion: "2.1", nodeVisibility: GetGLTFNodeVisibility });
            expect(gltf.nodes.find((node) => node.name === "parent")?.visible).toBe(false);
            expect(gltf.nodes.find((node) => node.name === "triangle")?.visible).toBe(true);
        });

        it("round trips loaded visibility animation", async () => {
            await ImportGltfAsync(testScene.scene, CreateVisibilityAnimationGltf());
            const { gltf, bin } = await ExportGltfWithBinaryAsync(testScene.scene, {
                gltfVersion: "2.1",
                nodeVisibility: GetGLTFNodeVisibility,
                nodeVisibilityAnimationTarget: (target) => GetGLTFVisibilityAnimationTarget(target),
            });
            expect(gltf.extensionsRequired).toContain("KHR_animation_pointer");
            const channel = gltf.animations![0].channels[0];
            const sampler = gltf.animations![0].samplers[channel.sampler];
            expect(sampler.interpolation).toBe("STEP");
            expect(gltf.accessors[sampler.output].componentType).toBe(5121);
            expect(channel.target.extensions?.KHR_animation_pointer).toEqual({ pointer: `/nodes/${gltf.nodes.findIndex((node) => node.name === "parent")}/visible` });
            expect(ReadAccessor(gltf, bin, sampler.output)).toEqual([1, 0, 1]);
            gltf.buffers[0].uri = ToDataUri(new Uint8Array(bin));
            const imported = CreateTestScene();
            try {
                await ImportGltfAsync(imported.scene, gltf);
                const mesh = imported.scene.getMeshByName("triangle")!;
                const group = imported.scene.animationGroups[0];
                group.start(false);
                group.goToFrame(60);
                expect(mesh.isVisible).toBe(false);
                group.goToFrame(120);
                expect(mesh.isVisible).toBe(true);
            } finally {
                imported.dispose();
            }
        });

        it("honors animation filters and rejects invalid visibility animations", async () => {
            await ImportGltfAsync(testScene.scene, CreateVisibilityAnimationGltf());
            const options = { gltfVersion: "2.1" as const, nodeVisibility: GetGLTFNodeVisibility, nodeVisibilityAnimationTarget: GetGLTFVisibilityAnimationTarget };
            const filtered = await ExportGltfAsync(testScene.scene, { ...options, shouldExportAnimation: () => false });
            expect(filtered.animations).toBeUndefined();
            expect(filtered.extensionsUsed ?? []).not.toContain("KHR_animation_pointer");
            await expect(ExportGltfAsync(testScene.scene, { ...options, shouldExportNode: (node) => node.name !== "parent" })).rejects.toThrow(/not exported/i);
            const group = testScene.scene.animationGroups[0];
            const targeted = group.targetedAnimations[0];
            const keys = targeted.animation.getKeys();
            const interpolation = keys[0].interpolation;
            keys[0].interpolation = undefined;
            await expect(ExportGltfAsync(testScene.scene, options)).rejects.toThrow(/invalid visibility animation/i);
            keys[0].interpolation = interpolation;
            group.addTargetedAnimation(targeted.animation.clone(), targeted.target);
            await expect(ExportGltfAsync(testScene.scene, options)).rejects.toThrow(/duplicate visibility animation/i);
            await expect(ExportGltfAsync(testScene.scene, { gltfVersion: "2.1", nodeVisibilityAnimationTarget: GetGLTFVisibilityAnimationTarget })).rejects.toThrow(
                /requires nodeVisibility/i
            );
        });

        it("loads visibility animation targeting a node outside the selected scene", async () => {
            const source = CreateVisibilityAnimationGltf();
            source.scenes[0].nodes = [1];
            await ImportGltfAsync(testScene.scene, source);
            const target = testScene.scene.animationGroups[0].targetedAnimations[0].target;
            expect(GetGLTFVisibilityAnimationTarget(target)).toBeUndefined();
        });

        it.each([
            ["sibling", false, false],
            ["self", false, false],
            ["child", false, false],
            ["sibling", true, false],
            ["self", true, false],
            ["child", true, false],
            ["sibling", false, true],
            ["child", true, true],
        ])("preserves visibility and skinning with a %s joint, right-handed: %s, multiple primitives: %s", async (jointLocation, rightHanded, multiple) => {
            testScene.scene.useRightHandedSystem = rightHanded;
            const source = CreateTriangleGltf();
            const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
            const joints = new Uint8Array(12);
            const weights = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
            const times = new Float32Array([0, 1]);
            const translations = new Float32Array([2, 0, 0, 3, 0, 0]);
            source.asset = { version: "2.1", minVersion: "2.1" };
            source.scenes[0].nodes = [0];
            source.nodes = [
                { name: "parent", visible: false, translation: [5, 0, 0], children: jointLocation === "child" ? [1] : [1, 2] },
                {
                    name: "skinned",
                    mesh: 0,
                    skin: 0,
                    translation: [2, 0, 0],
                    children: jointLocation === "child" ? [2] : undefined,
                    boundingVolume: { shape: 0, translation: [0, 0, 2] },
                },
                { name: "joint" },
            ];
            const preserveUniqueId = jointLocation === "sibling" && rightHanded && !multiple;
            if (preserveUniqueId) {
                source.uniqueIds = { skinnedNode: { pointer: "/nodes/1" } };
            }
            source.shapes = [{ type: "sphere", sphere: { radius: 2 } }];
            source.skins = [jointLocation === "self" ? { joints: [1], skeleton: 1 } : { joints: [2], skeleton: 2 }];
            const primitive = { attributes: { POSITION: 0, JOINTS_0: 1, WEIGHTS_0: 2 } };
            source.meshes = [{ primitives: multiple ? [primitive, structuredClone(primitive)] : [primitive] }];
            source.accessors = [
                { bufferView: 0, componentType: 5126, count: 3, type: "VEC3" as IAccessor["type"], min: [0, 0, 0], max: [1, 1, 0] },
                { bufferView: 1, componentType: 5121, count: 3, type: "VEC4" as IAccessor["type"] },
                { bufferView: 2, componentType: 5126, count: 3, type: "VEC4" as IAccessor["type"] },
                { bufferView: 3, componentType: 5126, count: 2, type: "SCALAR" as IAccessor["type"], min: [0], max: [1] },
                { bufferView: 4, componentType: 5126, count: 2, type: "VEC3" as IAccessor["type"] },
            ];
            source.bufferViews = [
                { buffer: 0, byteLength: 36 },
                { buffer: 0, byteOffset: 36, byteLength: 12 },
                { buffer: 0, byteOffset: 48, byteLength: 48 },
                { buffer: 0, byteOffset: 96, byteLength: 8 },
                { buffer: 0, byteOffset: 104, byteLength: 24 },
            ];
            source.buffers = [{ byteLength: 128, uri: ToDataUri(positions, joints, weights, times, translations) }];
            source.animations = [{ samplers: [{ input: 3, output: 4 }], channels: [{ sampler: 0, target: { node: 1, path: "translation" } }] }];
            await ImportGltfAsync(testScene.scene, source);
            const meshName = multiple ? "skinned_primitive0" : "skinned";
            const sourceMesh = testScene.scene.getMeshByName(meshName);
            if (!sourceMesh) {
                throw new Error("Expected imported skinned mesh");
            }
            const sourceGroup = testScene.scene.animationGroups[0];
            sourceGroup.start(false);
            sourceGroup.goToFrame(sourceGroup.from);
            const sourceVertex = () => {
                testScene.scene.skeletons[0].prepare(true);
                return Vector3.TransformCoordinates(Vector3.FromArray(sourceMesh.getPositionData(true)!), sourceMesh.computeWorldMatrix(true));
            };
            const sourcePosition = sourceVertex();
            sourceGroup.goToFrame(sourceGroup.to);
            const animatedPosition = sourceVertex();
            sourceGroup.goToFrame(sourceGroup.from);
            const sourceNode = testScene.scene.transformNodes.find((entry) => entry.name === "skinned")!;
            const sourceBounds = GetGLTFBoundingVolume(sourceNode)!.getBoundingInfo().boundingBox;
            const sourceCenter = sourceBounds.centerWorld.clone();
            const { gltf, bin } = await ExportGltfWithBinaryAsync(testScene.scene, {
                gltfVersion: "2.1",
                nodeVisibility: GetGLTFNodeVisibility,
                nodeBoundingVolume: (node) => (node instanceof TransformNode ? GetGLTFBoundingVolume(node) : undefined),
                preserveNodeUniqueIds: preserveUniqueId,
            });
            if (preserveUniqueId) {
                expect(GetGLTFValueByUniqueId(gltf, "skinnedNode")).toEqual(expect.objectContaining({ name: "skinned" }));
            }
            if (jointLocation === "sibling" && rightHanded && !multiple) {
                const boundsOnly = await ExportGltfAsync(testScene.scene, {
                    gltfVersion: "2.1",
                    nodeBoundingVolume: (node) => (node instanceof TransformNode ? GetGLTFBoundingVolume(node) : undefined),
                });
                const owner = boundsOnly.nodes.find((entry) => entry.boundingVolume);
                expect(owner?.children?.some((index) => boundsOnly.nodes[index].mesh !== undefined)).toBe(true);
            }
            gltf.buffers[0].uri = ToDataUri(new Uint8Array(bin));
            const imported = CreateTestScene();
            imported.scene.useRightHandedSystem = rightHanded;
            try {
                const container = await new GLTFFileLoader().loadAssetContainerAsync(imported.scene, { json: gltf }, "");
                const importedNode = container.transformNodes.find((entry) => entry.name === "skinned")!;
                const importedBounds = GetGLTFBoundingVolume(importedNode)!.getBoundingInfo().boundingBox;
                expect(Vector3.Distance(importedBounds.centerWorld, sourceCenter)).toBeLessThan(1e-5);
                const mesh = container.meshes.find((entry) => entry.name === meshName);
                expect(mesh?.isVisible).toBe(false);
                if (multiple) {
                    expect(container.meshes.find((entry) => entry.name === "skinned_primitive1")?.isVisible).toBe(false);
                }
                if (!mesh) {
                    throw new Error("Expected reimported skinned mesh");
                }
                container.skeletons[0].prepare(true);
                const position = Vector3.TransformCoordinates(Vector3.FromArray(mesh.getPositionData(true)!), mesh.computeWorldMatrix(true));
                expect(Vector3.Distance(position, sourcePosition)).toBeLessThan(1e-5);
                const parentIndex = gltf.nodes.findIndex((node) => node.name === "parent");
                const parent = GetPathToObjectConverter(gltf as IRuntimeGLTF).convert(`/nodes/${parentIndex}/visible`);
                parent.info.set!(true, parent.object);
                expect(mesh?.isVisible).toBe(true);
                if (multiple) {
                    expect(container.meshes.find((entry) => entry.name === "skinned_primitive1")?.isVisible).toBe(true);
                }
                const group = container.animationGroups[0];
                group.start(false);
                group.goToFrame(group.to);
                container.skeletons[0].prepare(true);
                const animated = Vector3.TransformCoordinates(Vector3.FromArray(mesh.getPositionData(true)!), mesh.computeWorldMatrix(true));
                expect(Vector3.Distance(animated, animatedPosition)).toBeLessThan(1e-5);
                if (jointLocation === "sibling" && rightHanded && !multiple) {
                    sourceMesh.parent = new TransformNode("new parent", testScene.scene);
                    await expect(ExportGltfAsync(testScene.scene, { gltfVersion: "2.1", nodeVisibility: GetGLTFNodeVisibility })).rejects.toThrow(/hierarchy has changed/i);
                }
            } finally {
                imported.dispose();
            }
        });

        it("preserves imported camera direction when retaining nodes for visibility", async () => {
            const source = CreateTriangleGltf();
            source.asset = { version: "2.1", minVersion: "2.1" };
            source.nodes.push({ name: "camera node", camera: 0, translation: [2, 1, 3], rotation: [0, 0.38268343, 0, 0.92387953] });
            source.scenes[0].nodes.push(1);
            source.cameras = [{ name: "camera", type: "perspective" as ICamera["type"], perspective: { yfov: 1, znear: 0.1 } }];
            await ImportGltfAsync(testScene.scene, source);
            const camera = testScene.scene.getCameraByName("camera");
            if (!camera) {
                throw new Error("Expected imported camera");
            }
            const direction = camera.getDirection(Vector3.Forward());
            const { gltf, bin } = await ExportGltfWithBinaryAsync(testScene.scene, { gltfVersion: "2.1", nodeVisibility: GetGLTFNodeVisibility });
            gltf.buffers[0].uri = ToDataUri(new Uint8Array(bin));
            const imported = CreateTestScene();
            try {
                await ImportGltfAsync(imported.scene, gltf);
                const roundTripCamera = imported.scene.getCameraByName("camera");
                if (!roundTripCamera) {
                    throw new Error("Expected reimported camera");
                }
                expect(Vector3.Distance(roundTripCamera.getDirection(Vector3.Forward()), direction)).toBeLessThan(1e-5);
            } finally {
                imported.dispose();
            }
        });

        it("preserves camera and light nodes when authored visibility is supplied", async () => {
            testScene.scene.useRightHandedSystem = true;
            const cameraParent = new TransformNode("camera parent", testScene.scene);
            const camera = new FreeCamera("camera", Vector3.Zero(), testScene.scene);
            camera.parent = cameraParent;
            const lightParent = new TransformNode("light parent", testScene.scene);
            const light = new PointLight("light", Vector3.Zero(), testScene.scene);
            light.parent = lightParent;
            const areaParent = new TransformNode("area parent", testScene.scene);
            const area = new RectAreaLight("area", Vector3.Zero(), 1, 1, testScene.scene);
            area.parent = areaParent;

            const gltf = await ExportGltfAsync(testScene.scene, {
                gltfVersion: "2.1",
                nodeVisibility: (node) => (node === camera || node === light || node === area ? false : undefined),
            });
            for (const [parentName, childName] of [
                ["camera parent", "camera"],
                ["light parent", "light"],
                ["area parent", "area"],
            ]) {
                const parent = gltf.nodes.find((node) => node.name === parentName);
                const child = gltf.nodes.find((node) => node.name === childName);
                expect(child?.visible).toBe(false);
                expect(parent?.children?.map((index) => gltf.nodes[index].name)).toContain(childName);
            }
            const bounded = await ExportGltfAsync(testScene.scene, {
                gltfVersion: "2.1",
                nodeBoundingVolume: (node) => (node === camera || node === light || node === area ? { shape: { type: "sphere" } } : undefined),
            });
            for (const name of ["camera", "light", "area"]) {
                expect(bounded.nodes.find((node) => node.name === name)?.boundingVolume).toBeDefined();
            }

            const imported = CreateTestScene();
            try {
                await ImportGltfAsync(imported.scene, gltf);
                expect(imported.scene.getCameraByName("camera")).not.toBeNull();
                expect(imported.scene.getLightByName("light")?.isEnabled()).toBe(false);
                expect(imported.scene.getLightByName("area")?.isEnabled()).toBe(false);
            } finally {
                imported.dispose();
            }
        });

        it("preserves imported coordinates when retaining roots for node visibility", async () => {
            const source = CreateTriangleGltf();
            source.nodes[0].translation = [1, 2, 3];
            await ImportGltfAsync(testScene.scene, source);
            const sourceMesh = testScene.scene.getMeshByName("triangle");
            if (!sourceMesh) {
                throw new Error("Expected imported triangle");
            }
            const sourcePosition = Vector3.TransformCoordinates(Vector3.FromArray(sourceMesh.getVerticesData("position")!, 3), sourceMesh.computeWorldMatrix(true));
            const { gltf, bin } = await ExportGltfWithBinaryAsync(testScene.scene, { gltfVersion: "2.1", nodeVisibility: () => undefined });
            gltf.buffers[0].uri = ToDataUri(new Uint8Array(bin));

            const imported = CreateTestScene();
            try {
                await ImportGltfAsync(imported.scene, gltf);
                const mesh = imported.scene.getMeshByName("triangle");
                if (!mesh) {
                    throw new Error("Expected reimported triangle");
                }
                const position = Vector3.TransformCoordinates(Vector3.FromArray(mesh.getVerticesData("position")!, 3), mesh.computeWorldMatrix(true));
                expect(Vector3.Distance(position, sourcePosition)).toBeLessThan(1e-5);
            } finally {
                imported.dispose();
            }
        });

        it.each([false, true])("round trips authored bounding volumes in a right-handed scene: %s", async (rightHanded) => {
            testScene.scene.useRightHandedSystem = rightHanded;
            const source = CreateTriangleGltf();
            source.asset = { version: "2.1", minVersion: "2.1" };
            source.nodes = [
                {
                    name: "bounded",
                    translation: [1, 2, 3],
                    rotation: [0, 0.38268343, 0, 0.92387953],
                    children: [1, 2],
                    boundingVolume: { shape: 0, translation: [2, 3, 4], rotation: [0, 0, 0.38268343, 0.92387953], scale: [1, 2, 3] },
                },
                { name: "triangle", mesh: 0 },
                { name: "other", boundingVolume: { shape: 0 } },
            ];
            source.shapes = [{ type: "box", box: { size: [2, 4, 6] } }];
            await ImportGltfAsync(testScene.scene, source);
            const sourceNode = testScene.scene.transformNodes.find((node) => node.name === "bounded")!;
            const sourceBounds = GetGLTFBoundingVolume(sourceNode)!.getBoundingInfo().boundingBox;
            const minimum = sourceBounds.minimumWorld.clone();
            const maximum = sourceBounds.maximumWorld.clone();
            const { gltf, bin } = await ExportGltfWithBinaryAsync(testScene.scene, {
                gltfVersion: "2.1",
                nodeBoundingVolume: (node) => (node instanceof TransformNode ? GetGLTFBoundingVolume(node) : undefined),
            });
            expect(gltf.shapes).toHaveLength(1);
            expect(gltf.nodes.find((node) => node.name === "bounded")?.boundingVolume?.shape).toBe(0);
            expect(gltf.nodes.find((node) => node.name === "other")?.boundingVolume?.shape).toBe(0);
            gltf.buffers[0].uri = ToDataUri(new Uint8Array(bin));
            const imported = CreateTestScene();
            imported.scene.useRightHandedSystem = rightHanded;
            try {
                await ImportGltfAsync(imported.scene, gltf);
                const node = imported.scene.transformNodes.find((entry) => entry.name === "bounded")!;
                const bounds = GetGLTFBoundingVolume(node)!.getBoundingInfo().boundingBox;
                expect(Vector3.Distance(bounds.minimumWorld, minimum)).toBeLessThan(1e-5);
                expect(Vector3.Distance(bounds.maximumWorld, maximum)).toBeLessThan(1e-5);
            } finally {
                imported.dispose();
            }
        });

        it("rejects bounding volumes for glTF 2.0", async () => {
            await expect(ExportGltfAsync(testScene.scene, { nodeBoundingVolume: () => undefined })).rejects.toThrow(/bounding volume.*2.1/i);
        });

        it("rejects unsupported or degenerate bounding-volume shapes", async () => {
            new TransformNode("bounded", testScene.scene);
            await expect(ExportGltfAsync(testScene.scene, { gltfVersion: "2.1", nodeBoundingVolume: () => ({ shape: { type: "capsule" } }) })).rejects.toThrow(
                /invalid bounding volume shape/i
            );
            await expect(
                ExportGltfAsync(testScene.scene, { gltfVersion: "2.1", nodeBoundingVolume: () => ({ shape: { type: "box", box: { size: [1, 0, 1] } } }) })
            ).rejects.toThrow(/invalid bounding volume shape/i);
            await expect(
                ExportGltfAsync(testScene.scene, { gltfVersion: "2.1", nodeBoundingVolume: () => ({ shape: { type: "box", extensions: { CUSTOM_shape: {} } } }) })
            ).rejects.toThrow(/invalid bounding volume shape/i);
        });

        it("rejects non-finite bounding-volume transforms", async () => {
            new TransformNode("bounded", testScene.scene);
            await expect(
                ExportGltfAsync(testScene.scene, {
                    gltfVersion: "2.1",
                    nodeBoundingVolume: () => ({ shape: { type: "sphere" }, translation: new Vector3(Infinity, 0, 0) }),
                })
            ).rejects.toThrow(/invalid bounding volume transform/i);
        });

        it("exports unique IDs against final node indices and JSON pointers", async () => {
            const node = new TransformNode("identified", testScene.scene);
            node.metadata = { gltf: { extras: { "a/b": { "~name": 7 } } } };
            const gltf = await ExportGltfAsync(testScene.scene, {
                gltfVersion: "2.1",
                uniqueIds: ({ getNodeIndex }) => {
                    const index = getNodeIndex(node);
                    return { target: `/nodes/${index}`, value: `/nodes/${index}/extras/a~1b/~0name`, ["__proto__"]: "/asset" };
                },
            });
            expect(GetGLTFValueByUniqueId(gltf, "target")).toEqual(expect.objectContaining({ name: "identified" }));
            expect(GetGLTFValueByUniqueId(gltf, "value")).toBe(7);
            expect(GetGLTFValueByUniqueId(gltf, "__proto__")).toEqual(gltf.asset);
        });

        it("preserves imported node unique IDs when requested", async () => {
            const source = CreateTriangleGltf();
            source.asset.version = "2.1";
            source.uniqueIds = {
                triangle: { pointer: "/nodes/0" },
                triangleName: { pointer: "/nodes/0/name" },
            };
            await ImportGltfAsync(testScene.scene, source);

            const importedNode = testScene.scene.meshes.find((mesh) => mesh.name === "triangle")!;
            expect(GetGLTFNodeUniqueIds(importedNode)).toEqual({ triangle: "", triangleName: "/name" });

            const gltf = await ExportGltfAsync(testScene.scene, { gltfVersion: "2.1", preserveNodeUniqueIds: true });

            expect(GetGLTFValueByUniqueId(gltf, "triangle")).toEqual(expect.objectContaining({ name: "triangle" }));
            expect(GetGLTFValueByUniqueId(gltf, "triangleName")).toBe("triangle");
        });

        it("rejects preserved node IDs whose target fields were not exported", async () => {
            const source = CreateTriangleGltf();
            source.asset.version = "2.1";
            source.uniqueIds = { missing: { pointer: "/nodes/0/extras/missing" } };
            await ImportGltfAsync(testScene.scene, source);

            await expect(ExportGltfAsync(testScene.scene, { gltfVersion: "2.1", preserveNodeUniqueIds: true })).rejects.toThrow(/unique id 'missing' points to a missing value/i);
        });

        it("rejects invalid unique IDs and unresolved pointers", async () => {
            new TransformNode("node", testScene.scene);
            for (const uniqueIds of [
                { "bad.id": "/asset" },
                { target: "/nodes/999" },
                { target: "/uniqueIds/target" },
                { target: "/asset/missing" },
                { target: "/asset/~2" },
                { target: "/nodes/00" },
                { "bad\u0085id": "/asset" },
            ]) {
                await expect(ExportGltfAsync(testScene.scene, { gltfVersion: "2.1", uniqueIds: () => uniqueIds })).rejects.toThrow(/unique id/i);
            }
            await expect(ExportGltfAsync(testScene.scene, { uniqueIds: () => ({}) })).rejects.toThrow(/unique id.*2.1/i);
        });

        it("validates pointers against the exported document rather than a modified callback snapshot", async () => {
            new TransformNode("node", testScene.scene);
            await expect(
                ExportGltfAsync(testScene.scene, {
                    gltfVersion: "2.1",
                    uniqueIds: ({ gltf }) => {
                        gltf.asset.generator = "changed";
                        (gltf.asset as { invented?: string }).invented = "not exported";
                        return { target: "/asset/invented" };
                    },
                })
            ).rejects.toThrow(/unique id.*missing/i);
        });
    });

    describe("meshes", () => {
        it("round trips vertex data and mode of a primitive", async () => {
            const source = CreateTriangleGltf();
            const sourceUvs = new Float32Array([0, 0, 1, 0, 0, 1]);
            const sourceColors = new Float32Array([1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1]);
            source.buffers[0].uri = ToDataUri(
                new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
                new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
                new Uint16Array([0, 1, 2, 0]),
                sourceUvs,
                sourceColors
            );
            source.buffers[0].byteLength = 80 + sourceUvs.byteLength + sourceColors.byteLength;
            source.bufferViews.push({ buffer: 0, byteOffset: 80, byteLength: 24 }, { buffer: 0, byteOffset: 104, byteLength: 48 });
            source.accessors.push(
                { bufferView: 3, componentType: 5126, count: 3, type: "VEC2" as IAccessor["type"] },
                { bufferView: 4, componentType: 5126, count: 3, type: "VEC4" as IAccessor["type"] }
            );
            source.meshes[0].primitives[0].attributes.TEXCOORD_0 = 3;
            source.meshes[0].primitives[0].attributes.COLOR_0 = 4;

            const { gltf, bin } = await RoundTripGltfWithBinaryAsync(testScene.scene, source, { exportUnusedUVs: true });

            const primitive = gltf.meshes[0].primitives[0];
            expect(primitive.mode ?? 4).toBe(4);
            expect(Object.keys(primitive.attributes).sort()).toEqual(["COLOR_0", "NORMAL", "POSITION", "TEXCOORD_0"]);
            const position = gltf.accessors[primitive.attributes.POSITION];
            expect(position).toMatchObject({ componentType: 5126, count: 3, type: "VEC3", min: [0, 0, 0], max: [1, 1, 0] });
            const positions = ReadAccessor(gltf, bin, primitive.attributes.POSITION);
            const uvs = ReadAccessor(gltf, bin, primitive.attributes.TEXCOORD_0);
            const colors = ReadAccessor(gltf, bin, primitive.attributes.COLOR_0);
            const indices = primitive.indices === undefined ? [0, 1, 2] : ReadAccessor(gltf, bin, primitive.indices);
            const corners = indices
                .map((index) => positions.slice(index * 3, index * 3 + 3))
                .map((position) => position.join(","))
                .sort();
            expect(corners).toEqual(["0,0,0", "0,1,0", "1,0,0"]);

            const rows = positions.reduce<number[][]>((result, _, index) => {
                if (index % 3 === 0) {
                    const vertex = index / 3;
                    result.push([...positions.slice(index, index + 3), ...uvs.slice(vertex * 2, vertex * 2 + 2), ...colors.slice(vertex * 4, vertex * 4 + 4)]);
                }
                return result;
            }, []);
            const expectedRows = [
                [0, 0, 0, 0, 0, 1, 0, 0, 1],
                [1, 0, 0, 1, 0, 0, 1, 0, 1],
                [0, 1, 0, 0, 1, 0, 0, 1, 1],
            ];
            const byPosition = (left: number[], right: number[]) => left.slice(0, 3).join(",").localeCompare(right.slice(0, 3).join(","));
            expect(rows.sort(byPosition)).toEqual(expectedRows.sort(byPosition));
        });

        it("omits texture coordinates that no texture uses unless exportUnusedUVs is set", async () => {
            const box = CreateBox("box", {}, testScene.scene);
            expect(box.isVerticesDataPresent("uv")).toBe(true);

            const defaultExport = await ExportGltfAsync(testScene.scene);
            const withUVs = await ExportGltfAsync(testScene.scene, { exportUnusedUVs: true });

            expect(defaultExport.meshes[0].primitives[0].attributes.TEXCOORD_0).toBeUndefined();
            expect(withUVs.meshes[0].primitives[0].attributes.TEXCOORD_0).toBeDefined();
        });
    });

    // https://github.com/BabylonJS/Babylon.js/pull/4590
    it("exports unit-length normals", async () => {
        const source = CreateTriangleGltf();
        source.buffers[0].uri = ToDataUri(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]), new Uint16Array([0, 1, 2, 0]));
        const { scene } = testScene;
        await ImportGltfAsync(scene, source);
        const mesh = scene.meshes.find((m) => m.getTotalVertices() > 0)!;
        mesh.setVerticesData("normal", [0, 0, 5, 0, 0, 5, 0, 0, 5], true);

        const { gltf, bin } = await ExportGltfWithBinaryAsync(scene);

        const normals = ReadAccessor(gltf, bin, gltf.meshes[0].primitives[0].attributes.NORMAL);
        for (let i = 0; i < normals.length; i += 3) {
            expect(Math.hypot(normals[i], normals[i + 1], normals[i + 2])).toBeCloseTo(1);
        }
    });

    describe("multi-material meshes", () => {
        // https://forum.babylonjs.com/t/multi-materials-cause-strange-glb-export-artifacts/60558
        // https://forum.babylonjs.com/t/exporting-to-gltf-in-new-version-causing-an-error-on-importing-from-blender/62030
        it("preserves geometry and material for sub meshes with offset vertex and index ranges", async () => {
            const { scene } = testScene;
            const mesh = new Mesh("mesh", scene);
            const sourcePositions = [0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 3, 0, 0, 2, 1, 0];
            mesh.setVerticesData("position", sourcePositions);
            mesh.setIndices([0, 1, 2, 3, 4, 5]);
            mesh.subMeshes = [];
            const multiMaterial = new MultiMaterial("multi", scene);
            multiMaterial.subMaterials.push(new PBRMaterial("a", scene), new PBRMaterial("b", scene));
            mesh.material = multiMaterial;
            new SubMesh(0, 0, 3, 0, 3, mesh);
            new SubMesh(1, 3, 3, 3, 3, mesh);

            const { gltf, bin } = await ExportGltfWithBinaryAsync(scene);

            const primitives = gltf.meshes[0].primitives;
            expect(primitives).toHaveLength(2);
            const actual = primitives.map((primitive) => {
                const positions = ReadAccessor(gltf, bin, primitive.attributes.POSITION);
                const indices = primitive.indices === undefined ? Array.from({ length: positions.length / 3 }, (_, index) => index) : ReadAccessor(gltf, bin, primitive.indices);
                const corners = indices.map((index) => positions.slice(index * 3, index * 3 + 3).join(",")).sort();
                const material = primitive.material === undefined ? undefined : gltf.materials[primitive.material];
                if (!material) {
                    throw new Error("Expected a material for every exported sub mesh");
                }
                return { material: material.name, corners };
            });
            expect(actual).toEqual([
                { material: "a", corners: ["0,0,0", "0,1,0", "-1,0,0"].sort() },
                { material: "b", corners: ["-2,0,0", "-2,1,0", "-3,0,0"].sort() },
            ]);
        });

        // https://github.com/BabylonJS/Babylon.js/pull/17244
        it("exports the requested typed index subarray for a sub mesh", async () => {
            const { scene } = testScene;
            const mesh = new Mesh("mesh", scene);
            mesh.setVerticesData("position", [0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 3, 0, 0, 2, 1, 0]);
            mesh.setIndices(new Uint16Array([0, 1, 2, 3, 4, 5]));
            mesh.subMeshes = [];
            const multiMaterial = new MultiMaterial("multi", scene);
            const firstMaterial = new PBRMaterial("a", scene);
            const secondMaterial = new PBRMaterial("b", scene);
            firstMaterial.sideOrientation = Material.CounterClockWiseSideOrientation;
            secondMaterial.sideOrientation = Material.CounterClockWiseSideOrientation;
            multiMaterial.subMaterials.push(firstMaterial, secondMaterial);
            mesh.material = multiMaterial;
            new SubMesh(0, 0, 6, 0, 3, mesh);
            new SubMesh(1, 0, 6, 3, 3, mesh);

            const { gltf, bin } = await ExportGltfWithBinaryAsync(scene);

            const primitive = gltf.meshes[0].primitives[1];
            if (primitive.indices === undefined) {
                throw new Error("Expected exported indices");
            }
            const indices = ReadAccessor(gltf, bin, primitive.indices);
            const positions = ReadAccessor(gltf, bin, primitive.attributes.POSITION);
            const corners = indices.map((index) => positions.slice(index * 3, index * 3 + 3).join(",")).sort();
            expect(indices).toEqual([3, 4, 5]);
            expect(corners).toEqual(["-2,0,0", "-2,1,0", "-3,0,0"].sort());
        });
    });

    describe("cameras", () => {
        it("round trips perspective and orthographic cameras", async () => {
            const source = CreateTriangleGltf();
            source.nodes[0].children = [1, 2];
            source.nodes.push({ name: "persp", camera: 0 }, { name: "ortho", camera: 1 });
            source.cameras = [
                { name: "persp", type: "perspective" as ICamera["type"], perspective: { yfov: 0.8, znear: 0.1, zfar: 50 } },
                { name: "ortho", type: "orthographic" as ICamera["type"], orthographic: { xmag: 4, ymag: 2, znear: 0.5, zfar: 20 } },
            ];

            const gltf = await RoundTripGltfAsync(testScene.scene, source);

            const cameras = gltf.cameras;
            if (!cameras) {
                throw new Error("Expected exported cameras");
            }
            const persp = cameras.find((c) => c.type === "perspective");
            const ortho = cameras.find((c) => c.type === "orthographic");
            if (!persp || !ortho || persp.type !== "perspective" || ortho.type !== "orthographic" || !persp.perspective || !ortho.orthographic) {
                throw new Error("Expected perspective and orthographic cameras");
            }
            expect(persp.perspective.yfov).toBeCloseTo(0.8);
            expect(persp.perspective.znear).toBeCloseTo(0.1);
            expect(persp.perspective.zfar).toBeCloseTo(50);
            expect(ortho.orthographic).toMatchObject({ znear: 0.5, zfar: 20 });
            expect(ortho.orthographic.xmag / ortho.orthographic.ymag).toBeCloseTo(2);
        });
    });
});
