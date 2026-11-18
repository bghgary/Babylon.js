import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { NullEngine } from "core/Engines/nullEngine";
import { Scene } from "core/scene";
import { GLTFFileLoader } from "loaders/glTF/glTFFileLoader.pure";
import "loaders/glTF/2.0/glTFLoader";

describe("glTF 2.1 quantized positions", () => {
    let engine: NullEngine;
    let scene: Scene;

    beforeEach(() => {
        engine = new NullEngine();
        scene = new Scene(engine);
    });

    afterEach(() => {
        scene.dispose();
        engine.dispose();
    });

    it.each([
        { componentType: 5120, min: -128, max: 127, stride: 4 },
        { componentType: 5120, min: -128, max: -128, stride: 4 },
        { componentType: 5122, min: -32768, max: 32767, stride: 8 },
        { componentType: 5122, min: -32768, max: -32768, stride: 8 },
    ])("normalizes $componentType position bounds from $min to $max", async ({ componentType, min, max, stride }) => {
        const bytes = new Uint8Array(stride * 3);
        const view = new DataView(bytes.buffer);
        if (componentType === 5120) {
            view.setInt8(0, min);
            view.setInt8(stride, max);
            view.setInt8(stride * 2, max === min ? min : 0);
        } else {
            view.setInt16(0, min, true);
            view.setInt16(stride, max, true);
            view.setInt16(stride * 2, max === min ? min : 0, true);
        }
        const data = {
            json: {
                asset: { version: "2.1" },
                scene: 0,
                scenes: [{ nodes: [0] }],
                nodes: [{ mesh: 0 }],
                meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
                accessors: [{ bufferView: 0, componentType, normalized: true, count: 3, type: "VEC3", min: [min, 0, 0], max: [max, 0, 0] }],
                bufferViews: [{ buffer: 0, byteLength: bytes.length, byteStride: stride }],
                buffers: [{ byteLength: bytes.length }],
            },
            bin: { byteLength: bytes.length, readAsync: async (offset: number, length: number) => bytes.subarray(offset, offset + length) },
        };
        const result = await new GLTFFileLoader().importMeshAsync(null, scene, data, "");
        const mesh = result.meshes[1];
        expect(mesh.getVerticesData("position")![0]).toBe(-1);
        expect(mesh.getBoundingInfo().boundingBox.minimum.x).toBe(-1);
        expect(mesh.getBoundingInfo().boundingBox.maximum.x).toBe(max === min ? -1 : 1);
    });

    it("loads quantized base attributes and morph differences", async () => {
        const bytes = new Uint8Array(96);
        const view = new DataView(bytes.buffer);
        for (let vertex = 0; vertex < 3; vertex++) {
            const position = vertex * 8;
            view.setUint16(position, 4 + vertex, true);
            view.setUint16(position + 2, 8, true);
            view.setUint16(position + 4, 12, true);
            bytes.set([127, 0, 0, 0], 24 + vertex * 4);
            bytes.set([127, 0, 0, 127], 36 + vertex * 4);
            view.setInt16(48 + vertex * 8, -32768, true);
            bytes.set([128, 127, 0, 0], 72 + vertex * 4);
            bytes.set([0, 127, 0, 0], 84 + vertex * 4);
        }
        const data = {
            json: {
                asset: { version: "2.1", minVersion: "2.1" },
                scene: 0,
                scenes: [{ nodes: [0] }],
                nodes: [{ mesh: 0 }],
                meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 1, TANGENT: 2 }, targets: [{ POSITION: 3, NORMAL: 4, TANGENT: 5 }] }], weights: [1] }],
                accessors: [
                    { bufferView: 0, componentType: 5123, count: 3, type: "VEC3", min: [4, 8, 12], max: [6, 8, 12] },
                    { bufferView: 1, componentType: 5120, normalized: true, count: 3, type: "VEC3" },
                    { bufferView: 2, componentType: 5120, normalized: true, count: 3, type: "VEC4" },
                    { bufferView: 3, componentType: 5122, normalized: true, count: 3, type: "VEC3", min: [-32768, 0, 0], max: [-32768, 0, 0] },
                    { bufferView: 4, componentType: 5120, normalized: true, count: 3, type: "VEC3" },
                    { bufferView: 5, componentType: 5120, normalized: true, count: 3, type: "VEC3" },
                ],
                bufferViews: [
                    { buffer: 0, byteLength: 24, byteStride: 8 },
                    { buffer: 0, byteOffset: 24, byteLength: 12, byteStride: 4 },
                    { buffer: 0, byteOffset: 36, byteLength: 12, byteStride: 4 },
                    { buffer: 0, byteOffset: 48, byteLength: 24, byteStride: 8 },
                    { buffer: 0, byteOffset: 72, byteLength: 12, byteStride: 4 },
                    { buffer: 0, byteOffset: 84, byteLength: 12, byteStride: 4 },
                ],
                buffers: [{ byteLength: bytes.length }],
            },
            bin: { byteLength: bytes.length, readAsync: async (offset: number, length: number) => bytes.subarray(offset, offset + length) },
        };
        const result = await new GLTFFileLoader().importMeshAsync(null, scene, data, "");
        const target = result.meshes[1].morphTargetManager!.getTarget(0);
        expect(Array.from(target.getPositions()!)).toEqual([3, 8, 12, 4, 8, 12, 5, 8, 12]);
        expect(Array.from(target.getNormals()!)).toEqual([0, 1, 0, 0, 1, 0, 0, 1, 0]);
        expect(Array.from(target.getTangents()!).slice(0, 9)).toEqual([1, 1, 0, 1, 1, 0, 1, 1, 0]);
    });
});
