import type {
    IBufferView,
    IAccessor,
    INode,
    IScene,
    IMesh,
    IMaterial,
    ITexture,
    IImage,
    ISampler,
    IAnimation,
    IMeshPrimitive,
    IBuffer,
    IGLTF,
    ITextureInfo,
    ISkin,
    ICamera,
} from "babylonjs-gltf2interface";
import { AccessorComponentType, AccessorType, ImageMimeType, MeshPrimitiveMode } from "babylonjs-gltf2interface";

import type { DataArray, IndicesArray, Nullable } from "core/types";
import { Matrix, TmpVectors } from "core/Maths/math.vector";
import { Vector3, Quaternion } from "core/Maths/math.vector";
import { Tools } from "core/Misc/tools";
import { Buffer, VertexBuffer } from "core/Buffers/buffer";
import type { Node } from "core/node";
import { TransformNode } from "core/Meshes/transformNode";
import type { SubMesh } from "core/Meshes/subMesh";
import { Mesh } from "core/Meshes/mesh";
import { InstancedMesh } from "core/Meshes/instancedMesh";
import type { BaseTexture } from "core/Materials/Textures/baseTexture";
import type { Texture } from "core/Materials/Textures/texture";
import { Material } from "core/Materials/material";
import { Engine } from "core/Engines/engine";
import type { Scene } from "core/scene";
import { EngineStore } from "core/Engines/engineStore";

import type { IGLTFExporterExtensionV2 } from "./glTFExporterExtension";
import { GLTFMaterialExporter } from "./glTFMaterialExporter";
import type { IExportOptions } from "./glTFSerializer";
import { GLTFData } from "./glTFData";
import {
    calculateMinMaxPositions,
    convertToRightHandedPosition,
    convertToRightHandedRotation,
    createAccessor,
    createBufferView,
    getAccessorType,
    getAttributeType,
    getPrimitiveMode,
} from "./glTFUtilities";
import { DataWriter } from "./dataWriter";
import { Camera } from "core/Cameras/camera";
import { MultiMaterial, PBRMaterial, StandardMaterial } from "core/Materials";
import { Logger } from "core/Misc/logger";

// Matrix that converts handedness on the X-axis.
const convertHandednessMatrix = Matrix.Compose(new Vector3(-1, 1, 1), Quaternion.Identity(), Vector3.Zero());

// 180 degrees rotation in Y.
// const rotation180Y = new Quaternion(0, 1, 0, 0);

function isNoopNode(node: Node, useRightHandedSystem: boolean): boolean {
    if (!(node instanceof TransformNode)) {
        return false;
    }

    // Transform
    if (useRightHandedSystem) {
        const matrix = node.getWorldMatrix();
        if (!matrix.isIdentity()) {
            return false;
        }
    } else {
        const matrix = node.getWorldMatrix().multiplyToRef(convertHandednessMatrix, TmpVectors.Matrix[0]);
        if (!matrix.isIdentity()) {
            return false;
        }
    }

    // Geometry
    if ((node instanceof Mesh && node.geometry) || (node instanceof InstancedMesh && node.sourceMesh.geometry)) {
        return false;
    }

    return true;
}

function isIndicesArray32Bits(indices: IndicesArray): boolean {
    return indices instanceof Array ? indices.some((value) => value >= 65536) : indices.BYTES_PER_ELEMENT === 4;
}

function indicesArrayToUint8Array(indices: IndicesArray, is32Bits: boolean): Uint8Array {
    if (indices instanceof Array) {
        indices = is32Bits ? new Uint32Array(indices) : new Uint16Array(indices);
        return new Uint8Array(indices.buffer, indices.byteOffset, indices.byteLength);
    }

    return ArrayBuffer.isView(indices) ? new Uint8Array(indices.buffer, indices.byteOffset, indices.byteLength) : new Uint8Array(indices);
}

function dataArrayToUint8Array(data: DataArray): Uint8Array {
    if (data instanceof Array) {
        const floatData = new Float32Array(data);
        return new Uint8Array(floatData.buffer, floatData.byteOffset, floatData.byteLength);
    }

    return ArrayBuffer.isView(data) ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(data);
}

class ExporterState {
    // Babylon indices array -> glTF buffer view index
    private _indicesBufferViewMap = new Map<IndicesArray, number>();

    // Babylon indices array, index start, index count -> glTF accessor index
    private _indicesAccessorMap = new Map<IndicesArray, Map<number, Map<number, number>>>();

    // Babylon buffer -> glTF buffer view index
    private _attributeBufferViewMap = new Map<Buffer, number>();

    // Babylon vertex buffer, vertex start, vertex count -> glTF accessor index
    private _attributeAccessorMap = new Map<VertexBuffer, Map<number, Map<number, number>>>();

    // Babylon mesh -> glTF mesh index
    private _meshMap = new Map<Mesh, number>();

    public getIndicesBufferView(indices: IndicesArray): number | undefined {
        return this._indicesBufferViewMap.get(indices);
    }

    public setIndicesBufferView(indices: IndicesArray, bufferViewIndex: number): void {
        this._indicesBufferViewMap.set(indices, bufferViewIndex);
    }

    public getIndicesAccessor(indices: IndicesArray, indexStart: number, indexCount: number): number | undefined {
        const map1 = this._indicesAccessorMap.get(indices);
        if (!map1) {
            return undefined;
        }

        const map2 = map1.get(indexStart);
        if (!map2) {
            return undefined;
        }

        return map2.get(indexCount);
    }

    public setIndicesAccessor(indices: IndicesArray, indexStart: number, indexCount: number, accessorIndex: number): void {
        const map1 = this._indicesAccessorMap.get(indices) ?? new Map<number, Map<number, number>>();
        const map2 = map1.get(indexStart) ?? new Map<number, number>();
        map2.set(indexCount, accessorIndex);
    }

    public getAttributeBufferView(buffer: Buffer): number | undefined {
        return this._attributeBufferViewMap.get(buffer);
    }

    public setAttributeBufferView(buffer: Buffer, bufferViewIndex: number): void {
        this._attributeBufferViewMap.set(buffer, bufferViewIndex);
    }

    public getAttributeAccessor(vertexBuffer: VertexBuffer, vertexStart: number, vertexCount: number): number | undefined {
        const map1 = this._attributeAccessorMap.get(vertexBuffer);
        if (!map1) {
            return undefined;
        }

        const map2 = map1.get(vertexStart);
        if (!map2) {
            return undefined;
        }

        return map2.get(vertexCount);
    }

    public setAttributeAccessor(vertexBuffer: VertexBuffer, vertexStart: number, vertexCount: number, accessorIndex: number): void {
        const map1 = this._attributeAccessorMap.get(vertexBuffer) ?? new Map<number, Map<number, number>>();
        const map2 = map1.get(vertexStart) ?? new Map<number, number>();
        map2.set(vertexCount, accessorIndex);
    }

    public getMesh(mesh: Mesh): number | undefined {
        return this._meshMap.get(mesh);
    }

    public setMesh(mesh: Mesh, meshIndex: number): void {
        this._meshMap.set(mesh, meshIndex);
    }
}

/** @internal */
export class GLTFExporter {
    public readonly _glTF: IGLTF = {
        asset: { generator: `Babylon.js v${Engine.Version}`, version: "2.0" },
    };

    public readonly _animations: IAnimation[] = [];
    public readonly _accessors: IAccessor[] = [];
    public readonly _bufferViews: IBufferView[] = [];
    public readonly _cameras: ICamera[] = [];
    public readonly _images: IImage[] = [];
    public readonly _materials: IMaterial[] = [];
    public readonly _meshes: IMesh[] = [];
    public readonly _nodes: INode[] = [];
    public readonly _samplers: ISampler[] = [];
    public readonly _scenes: IScene[] = [];
    public readonly _skins: ISkin[] = [];
    public readonly _textures: ITexture[] = [];

    public readonly _babylonScene: Scene;
    public readonly _imageData: { [fileName: string]: { data: ArrayBuffer; mimeType: ImageMimeType } } = {};
    private readonly _orderedImageData: Array<{ data: ArrayBuffer; mimeType: ImageMimeType }> = [];

    // /**
    //  * Baked animation sample rate
    //  */
    // private _animationSampleRate: number;

    private readonly _options: IExportOptions;

    private readonly _materialExporter = new GLTFMaterialExporter(this);

    private readonly _extensions: { [name: string]: IGLTFExporterExtensionV2 } = {};

    private readonly _dataWriter = new DataWriter(4);
    private readonly _nodesToConvertToRightHanded = new Set<Node>();

    private readonly _originalState = new ExporterState();
    private readonly _convertToRightHandedState = new ExporterState();

    // Babylon node -> glTF node index
    private readonly _nodeMap = new Map<Node, number>();

    // Babylon material -> glTF material index
    public readonly _materialMap = new Map<Material, number>();

    // A material in this set requires UVs
    public readonly _materialNeedsUVsSet = new Set<Material>();

    private static readonly _ExtensionNames = new Array<string>();
    private static readonly _ExtensionFactories: { [name: string]: (exporter: GLTFExporter) => IGLTFExporterExtensionV2 } = {};

    private _applyExtension<T>(
        node: Nullable<T>,
        extensions: IGLTFExporterExtensionV2[],
        index: number,
        actionAsync: (extension: IGLTFExporterExtensionV2, node: Nullable<T>) => Promise<Nullable<T>> | undefined
    ): Promise<Nullable<T>> {
        if (index >= extensions.length) {
            return Promise.resolve(node);
        }

        const currentPromise = actionAsync(extensions[index], node);

        if (!currentPromise) {
            return this._applyExtension(node, extensions, index + 1, actionAsync);
        }

        return currentPromise.then((newNode) => this._applyExtension(newNode, extensions, index + 1, actionAsync));
    }

    private _applyExtensions<T>(
        node: Nullable<T>,
        actionAsync: (extension: IGLTFExporterExtensionV2, node: Nullable<T>) => Promise<Nullable<T>> | undefined
    ): Promise<Nullable<T>> {
        const extensions: IGLTFExporterExtensionV2[] = [];
        for (const name of GLTFExporter._ExtensionNames) {
            extensions.push(this._extensions[name]);
        }

        return this._applyExtension(node, extensions, 0, actionAsync);
    }

    public _extensionsPreExportTextureAsync(context: string, babylonTexture: Nullable<Texture>, mimeType: ImageMimeType): Promise<Nullable<BaseTexture>> {
        return this._applyExtensions(babylonTexture, (extension, node) => extension.preExportTextureAsync && extension.preExportTextureAsync(context, node, mimeType));
    }

    public _extensionsPostExportMeshPrimitiveAsync(context: string, meshPrimitive: IMeshPrimitive, babylonSubMesh: SubMesh): Promise<Nullable<IMeshPrimitive>> {
        return this._applyExtensions(
            meshPrimitive,
            (extension, node) => extension.postExportMeshPrimitiveAsync && extension.postExportMeshPrimitiveAsync(context, node, babylonSubMesh)
        );
    }

    public _extensionsPostExportNodeAsync(context: string, node: Nullable<INode>, babylonNode: Node, nodeMap: { [key: number]: number }): Promise<Nullable<INode>> {
        return this._applyExtensions(node, (extension, node) => extension.postExportNodeAsync && extension.postExportNodeAsync(context, node, babylonNode, nodeMap));
    }

    public _extensionsPostExportMaterialAsync(context: string, material: Nullable<IMaterial>, babylonMaterial: Material): Promise<Nullable<IMaterial>> {
        return this._applyExtensions(material, (extension, node) => extension.postExportMaterialAsync && extension.postExportMaterialAsync(context, node, babylonMaterial));
    }

    public _extensionsPostExportMaterialAdditionalTextures(context: string, material: IMaterial, babylonMaterial: Material): BaseTexture[] {
        const output: BaseTexture[] = [];

        for (const name of GLTFExporter._ExtensionNames) {
            const extension = this._extensions[name];

            if (extension.postExportMaterialAdditionalTextures) {
                output.push(...extension.postExportMaterialAdditionalTextures(context, material, babylonMaterial));
            }
        }

        return output;
    }

    public _extensionsPostExportTextures(context: string, textureInfo: ITextureInfo, babylonTexture: BaseTexture): void {
        for (const name of GLTFExporter._ExtensionNames) {
            const extension = this._extensions[name];

            if (extension.postExportTexture) {
                extension.postExportTexture(context, textureInfo, babylonTexture);
            }
        }
    }

    private _forEachExtensions(action: (extension: IGLTFExporterExtensionV2) => void): void {
        for (const name of GLTFExporter._ExtensionNames) {
            const extension = this._extensions[name];
            if (extension.enabled) {
                action(extension);
            }
        }
    }

    private _extensionsOnExporting(): void {
        this._forEachExtensions((extension) => {
            if (extension.wasUsed) {
                this._glTF.extensionsUsed ||= [];
                if (this._glTF.extensionsUsed.indexOf(extension.name) === -1) {
                    this._glTF.extensionsUsed.push(extension.name);
                }

                if (extension.required) {
                    this._glTF.extensionsRequired ||= [];
                    if (this._glTF.extensionsRequired.indexOf(extension.name) === -1) {
                        this._glTF.extensionsRequired.push(extension.name);
                    }
                }

                if (extension.onExporting) {
                    extension.onExporting();
                }
            }
        });
    }

    private _loadExtensions(): void {
        for (const name of GLTFExporter._ExtensionNames) {
            const extension = GLTFExporter._ExtensionFactories[name](this);
            this._extensions[name] = extension;
        }
    }

    public constructor(babylonScene: Nullable<Scene> = EngineStore.LastCreatedScene, options?: IExportOptions) {
        if (!babylonScene) {
            throw new Error("No scene available to export");
        }

        this._babylonScene = babylonScene;
        this._options = options || {};

        //this._animationSampleRate = this._options.animationSampleRate || 1 / 60;

        this._loadExtensions();
    }

    public dispose() {
        for (const key in this._extensions) {
            const extension = this._extensions[key];
            extension.dispose();
        }
    }

    public get options() {
        return this._options;
    }

    public static RegisterExtension(name: string, factory: (exporter: GLTFExporter) => IGLTFExporterExtensionV2): void {
        if (GLTFExporter.UnregisterExtension(name)) {
            Tools.Warn(`Extension with the name ${name} already exists`);
        }

        GLTFExporter._ExtensionFactories[name] = factory;
        GLTFExporter._ExtensionNames.push(name);
    }

    public static UnregisterExtension(name: string): boolean {
        if (!GLTFExporter._ExtensionFactories[name]) {
            return false;
        }
        delete GLTFExporter._ExtensionFactories[name];

        const index = GLTFExporter._ExtensionNames.indexOf(name);
        if (index !== -1) {
            GLTFExporter._ExtensionNames.splice(index, 1);
        }

        return true;
    }

    // private _reorderIndicesBasedOnPrimitiveMode(submesh: SubMesh, primitiveMode: number, babylonIndices: IndicesArray, byteOffset: number, dataWriter: DataWriter): void {
    //     switch (primitiveMode) {
    //         case Material.TriangleFillMode: {
    //             if (!byteOffset) {
    //                 byteOffset = 0;
    //             }
    //             for (let i = submesh.indexStart, length = submesh.indexStart + submesh.indexCount; i < length; i = i + 3) {
    //                 const index = byteOffset + i * 4;
    //                 // swap the second and third indices
    //                 const secondIndex = dataWriter.getUInt32(index + 4);
    //                 const thirdIndex = dataWriter.getUInt32(index + 8);
    //                 dataWriter.setUInt32(thirdIndex, index + 4);
    //                 dataWriter.setUInt32(secondIndex, index + 8);
    //             }
    //             break;
    //         }
    //         case Material.TriangleFanDrawMode: {
    //             for (let i = submesh.indexStart + submesh.indexCount - 1, start = submesh.indexStart; i >= start; --i) {
    //                 dataWriter.setUInt32(babylonIndices[i], byteOffset);
    //                 byteOffset += 4;
    //             }
    //             break;
    //         }
    //         case Material.TriangleStripDrawMode: {
    //             if (submesh.indexCount >= 3) {
    //                 dataWriter.setUInt32(babylonIndices[submesh.indexStart + 2], byteOffset + 4);
    //                 dataWriter.setUInt32(babylonIndices[submesh.indexStart + 1], byteOffset + 8);
    //             }
    //             break;
    //         }
    //     }
    // }

    // /**
    //  * Reorders the vertex attribute data based on the primitive mode.  This is necessary when indices are not available and the winding order is
    //  * clock-wise during export to glTF
    //  * @param submesh BabylonJS submesh
    //  * @param primitiveMode Primitive mode of the mesh
    //  * @param vertexBufferKind The type of vertex attribute
    //  * @param meshAttributeArray The vertex attribute data
    //  * @param byteOffset The offset to the binary data
    //  * @param dataWriter The binary data for the glTF file
    //  */
    // private _reorderVertexAttributeDataBasedOnPrimitiveMode(
    //     submesh: SubMesh,
    //     primitiveMode: number,
    //     vertexBufferKind: string,
    //     meshAttributeArray: FloatArray,
    //     byteOffset: number,
    //     dataWriter: DataWriter
    // ): void {
    //     switch (primitiveMode) {
    //         case Material.TriangleFillMode: {
    //             this._reorderTriangleFillMode(submesh, vertexBufferKind, meshAttributeArray, byteOffset, dataWriter);
    //             break;
    //         }
    //         case Material.TriangleStripDrawMode: {
    //             this._reorderTriangleStripDrawMode(submesh, vertexBufferKind, meshAttributeArray, byteOffset, dataWriter);
    //             break;
    //         }
    //         case Material.TriangleFanDrawMode: {
    //             this._reorderTriangleFanMode(submesh, vertexBufferKind, meshAttributeArray, byteOffset, dataWriter);
    //             break;
    //         }
    //     }
    // }

    // /**
    //  * Reorders the vertex attributes in the correct triangle mode order .  This is necessary when indices are not available and the winding order is
    //  * clock-wise during export to glTF
    //  * @param submesh BabylonJS submesh
    //  * @param vertexBufferKind The type of vertex attribute
    //  * @param meshAttributeArray The vertex attribute data
    //  * @param byteOffset The offset to the binary data
    //  * @param dataWriter The binary data for the glTF file
    //  */
    // private _reorderTriangleFillMode(submesh: SubMesh, vertexBufferKind: string, meshAttributeArray: FloatArray, byteOffset: number, dataWriter: DataWriter): void {
    //     const vertexBuffer = this._getVertexBuffer(vertexBufferKind, submesh.getMesh());
    //     if (vertexBuffer) {
    //         const stride = vertexBuffer.byteStride / VertexBuffer.GetTypeByteLength(vertexBuffer.type);
    //         if (submesh.verticesCount % 3 !== 0) {
    //             Tools.Error("The submesh vertices for the triangle fill mode is not divisible by 3!");
    //         } else {
    //             const vertexData: Vector2[] | Vector3[] | Vector4[] = [];
    //             let index = 0;
    //             switch (vertexBufferKind) {
    //                 case VertexBuffer.PositionKind:
    //                 case VertexBuffer.NormalKind: {
    //                     for (let x = submesh.verticesStart; x < submesh.verticesStart + submesh.verticesCount; x = x + 3) {
    //                         index = x * stride;
    //                         (vertexData as Vector3[]).push(Vector3.FromArray(meshAttributeArray, index));
    //                         (vertexData as Vector3[]).push(Vector3.FromArray(meshAttributeArray, index + 2 * stride));
    //                         (vertexData as Vector3[]).push(Vector3.FromArray(meshAttributeArray, index + stride));
    //                     }
    //                     break;
    //                 }
    //                 case VertexBuffer.TangentKind: {
    //                     for (let x = submesh.verticesStart; x < submesh.verticesStart + submesh.verticesCount; x = x + 3) {
    //                         index = x * stride;
    //                         (vertexData as Vector4[]).push(Vector4.FromArray(meshAttributeArray, index));
    //                         (vertexData as Vector4[]).push(Vector4.FromArray(meshAttributeArray, index + 2 * stride));
    //                         (vertexData as Vector4[]).push(Vector4.FromArray(meshAttributeArray, index + stride));
    //                     }
    //                     break;
    //                 }
    //                 case VertexBuffer.ColorKind: {
    //                     const size = vertexBuffer.getSize();
    //                     for (let x = submesh.verticesStart; x < submesh.verticesStart + submesh.verticesCount; x = x + size) {
    //                         index = x * stride;
    //                         if (size === 4) {
    //                             (vertexData as Vector4[]).push(Vector4.FromArray(meshAttributeArray, index));
    //                             (vertexData as Vector4[]).push(Vector4.FromArray(meshAttributeArray, index + 2 * stride));
    //                             (vertexData as Vector4[]).push(Vector4.FromArray(meshAttributeArray, index + stride));
    //                         } else {
    //                             (vertexData as Vector3[]).push(Vector3.FromArray(meshAttributeArray, index));
    //                             (vertexData as Vector3[]).push(Vector3.FromArray(meshAttributeArray, index + 2 * stride));
    //                             (vertexData as Vector3[]).push(Vector3.FromArray(meshAttributeArray, index + stride));
    //                         }
    //                     }
    //                     break;
    //                 }
    //                 case VertexBuffer.UVKind:
    //                 case VertexBuffer.UV2Kind: {
    //                     for (let x = submesh.verticesStart; x < submesh.verticesStart + submesh.verticesCount; x = x + 3) {
    //                         index = x * stride;
    //                         (vertexData as Vector2[]).push(Vector2.FromArray(meshAttributeArray, index));
    //                         (vertexData as Vector2[]).push(Vector2.FromArray(meshAttributeArray, index + 2 * stride));
    //                         (vertexData as Vector2[]).push(Vector2.FromArray(meshAttributeArray, index + stride));
    //                     }
    //                     break;
    //                 }
    //                 default: {
    //                     Tools.Error(`Unsupported Vertex Buffer type: ${vertexBufferKind}`);
    //                 }
    //             }
    //             this._writeVertexAttributeData(vertexData, byteOffset, vertexBufferKind, dataWriter);
    //         }
    //     } else {
    //         Tools.Warn(`reorderTriangleFillMode: Vertex Buffer Kind ${vertexBufferKind} not present!`);
    //     }
    // }

    // /**
    //  * Reorders the vertex attributes in the correct triangle strip order.  This is necessary when indices are not available and the winding order is
    //  * clock-wise during export to glTF
    //  * @param submesh BabylonJS submesh
    //  * @param vertexBufferKind The type of vertex attribute
    //  * @param meshAttributeArray The vertex attribute data
    //  * @param byteOffset The offset to the binary data
    //  * @param dataWriter The binary data for the glTF file
    //  */
    // private _reorderTriangleStripDrawMode(submesh: SubMesh, vertexBufferKind: string, meshAttributeArray: FloatArray, byteOffset: number, dataWriter: DataWriter): void {
    //     const vertexBuffer = this._getVertexBuffer(vertexBufferKind, submesh.getMesh());
    //     if (vertexBuffer) {
    //         const stride = vertexBuffer.byteStride / VertexBuffer.GetTypeByteLength(vertexBuffer.type);

    //         const vertexData: Vector2[] | Vector3[] | Vector4[] = [];
    //         let index = 0;
    //         switch (vertexBufferKind) {
    //             case VertexBuffer.PositionKind:
    //             case VertexBuffer.NormalKind: {
    //                 index = submesh.verticesStart;
    //                 (vertexData as Vector3[]).push(Vector3.FromArray(meshAttributeArray, index + 2 * stride));
    //                 (vertexData as Vector3[]).push(Vector3.FromArray(meshAttributeArray, index + stride));
    //                 break;
    //             }
    //             case VertexBuffer.TangentKind: {
    //                 for (let x = submesh.verticesStart + submesh.verticesCount - 1; x >= submesh.verticesStart; --x) {
    //                     index = x * stride;
    //                     (vertexData as Vector4[]).push(Vector4.FromArray(meshAttributeArray, index));
    //                 }
    //                 break;
    //             }
    //             case VertexBuffer.ColorKind: {
    //                 for (let x = submesh.verticesStart + submesh.verticesCount - 1; x >= submesh.verticesStart; --x) {
    //                     index = x * stride;
    //                     vertexBuffer.getSize() === 4
    //                         ? (vertexData as Vector4[]).push(Vector4.FromArray(meshAttributeArray, index))
    //                         : (vertexData as Vector3[]).push(Vector3.FromArray(meshAttributeArray, index));
    //                 }
    //                 break;
    //             }
    //             case VertexBuffer.UVKind:
    //             case VertexBuffer.UV2Kind: {
    //                 for (let x = submesh.verticesStart + submesh.verticesCount - 1; x >= submesh.verticesStart; --x) {
    //                     index = x * stride;
    //                     (vertexData as Vector2[]).push(Vector2.FromArray(meshAttributeArray, index));
    //                 }
    //                 break;
    //             }
    //             default: {
    //                 Tools.Error(`Unsupported Vertex Buffer type: ${vertexBufferKind}`);
    //             }
    //         }
    //         this._writeVertexAttributeData(vertexData, byteOffset + 12, vertexBufferKind, dataWriter);
    //     } else {
    //         Tools.Warn(`reorderTriangleStripDrawMode: Vertex buffer kind ${vertexBufferKind} not present!`);
    //     }
    // }

    // /**
    //  * Reorders the vertex attributes in the correct triangle fan order.  This is necessary when indices are not available and the winding order is
    //  * clock-wise during export to glTF
    //  * @param submesh BabylonJS submesh
    //  * @param vertexBufferKind The type of vertex attribute
    //  * @param meshAttributeArray The vertex attribute data
    //  * @param byteOffset The offset to the binary data
    //  * @param dataWriter The binary data for the glTF file
    //  */
    // private _reorderTriangleFanMode(submesh: SubMesh, vertexBufferKind: string, meshAttributeArray: FloatArray, byteOffset: number, dataWriter: DataWriter): void {
    //     const vertexBuffer = this._getVertexBuffer(vertexBufferKind, submesh.getMesh());
    //     if (vertexBuffer) {
    //         const stride = vertexBuffer.byteStride / VertexBuffer.GetTypeByteLength(vertexBuffer.type);

    //         const vertexData: Vector2[] | Vector3[] | Vector4[] = [];
    //         let index = 0;
    //         switch (vertexBufferKind) {
    //             case VertexBuffer.PositionKind:
    //             case VertexBuffer.NormalKind: {
    //                 for (let x = submesh.verticesStart + submesh.verticesCount - 1; x >= submesh.verticesStart; --x) {
    //                     index = x * stride;
    //                     (vertexData as Vector3[]).push(Vector3.FromArray(meshAttributeArray, index));
    //                 }
    //                 break;
    //             }
    //             case VertexBuffer.TangentKind: {
    //                 for (let x = submesh.verticesStart + submesh.verticesCount - 1; x >= submesh.verticesStart; --x) {
    //                     index = x * stride;
    //                     (vertexData as Vector4[]).push(Vector4.FromArray(meshAttributeArray, index));
    //                 }
    //                 break;
    //             }
    //             case VertexBuffer.ColorKind: {
    //                 for (let x = submesh.verticesStart + submesh.verticesCount - 1; x >= submesh.verticesStart; --x) {
    //                     index = x * stride;
    //                     (vertexData as Vector4[]).push(Vector4.FromArray(meshAttributeArray, index));
    //                     vertexBuffer.getSize() === 4
    //                         ? (vertexData as Vector4[]).push(Vector4.FromArray(meshAttributeArray, index))
    //                         : (vertexData as Vector3[]).push(Vector3.FromArray(meshAttributeArray, index));
    //                 }
    //                 break;
    //             }
    //             case VertexBuffer.UVKind:
    //             case VertexBuffer.UV2Kind: {
    //                 for (let x = submesh.verticesStart + submesh.verticesCount - 1; x >= submesh.verticesStart; --x) {
    //                     index = x * stride;
    //                     (vertexData as Vector2[]).push(Vector2.FromArray(meshAttributeArray, index));
    //                 }
    //                 break;
    //             }
    //             default: {
    //                 Tools.Error(`Unsupported Vertex Buffer type: ${vertexBufferKind}`);
    //             }
    //         }
    //         this._writeVertexAttributeData(vertexData, byteOffset, vertexBufferKind, dataWriter);
    //     } else {
    //         Tools.Warn(`reorderTriangleFanMode: Vertex buffer kind ${vertexBufferKind} not present!`);
    //     }
    // }

    // /**
    //  * Writes the vertex attribute data to binary
    //  * @param vertices The vertices to write to the binary writer
    //  * @param byteOffset The offset into the binary writer to overwrite binary data
    //  * @param vertexAttributeKind The vertex attribute type
    //  * @param dataWriter The writer containing the binary data
    //  */
    // private _writeVertexAttributeData(vertices: Vector2[] | Vector3[] | Vector4[], byteOffset: number, vertexAttributeKind: string, dataWriter: DataWriter) {
    //     for (const vertex of vertices) {
    //         if (vertexAttributeKind === VertexBuffer.NormalKind) {
    //             vertex.normalize();
    //         } else if (vertexAttributeKind === VertexBuffer.TangentKind && vertex instanceof Vector4) {
    //             normalizeTangent(vertex);
    //         }

    //         for (const component of vertex.asArray()) {
    //             dataWriter.writeFloat32(component, byteOffset);
    //             byteOffset += 4;
    //         }
    //     }
    // }

    // /**
    //  * Writes mesh attribute data to a data buffer
    //  * Returns the bytelength of the data
    //  * @param vertexBufferKind Indicates what kind of vertex data is being passed in
    //  * @param attributeComponentKind
    //  * @param meshAttributeArray Array containing the attribute data
    //  * @param stride Specifies the space between data
    //  * @param dataWriter The buffer to write the binary data to
    //  * @param babylonTransformNode
    //  */
    // public _writeAttributeData(
    //     vertexBufferKind: string,
    //     attributeComponentKind: AccessorComponentType,
    //     meshAttributeArray: FloatArray,
    //     stride: number,
    //     dataWriter: DataWriter,
    //     babylonTransformNode: TransformNode
    // ) {
    //     let vertexAttributes: number[][] = [];
    //     let index: number;

    //     switch (vertexBufferKind) {
    //         case VertexBuffer.PositionKind: {
    //             for (let k = 0, length = meshAttributeArray.length / stride; k < length; ++k) {
    //                 index = k * stride;
    //                 const vertexData = Vector3.FromArray(meshAttributeArray, index);
    //                 vertexAttributes.push(vertexData.asArray());
    //             }
    //             break;
    //         }
    //         case VertexBuffer.NormalKind: {
    //             for (let k = 0, length = meshAttributeArray.length / stride; k < length; ++k) {
    //                 index = k * stride;
    //                 const vertexData = Vector3.FromArray(meshAttributeArray, index);
    //                 vertexAttributes.push(vertexData.normalize().asArray());
    //             }
    //             break;
    //         }
    //         case VertexBuffer.TangentKind: {
    //             for (let k = 0, length = meshAttributeArray.length / stride; k < length; ++k) {
    //                 index = k * stride;
    //                 const vertexData = Vector4.FromArray(meshAttributeArray, index);
    //                 normalizeTangent(vertexData);
    //                 vertexAttributes.push(vertexData.asArray());
    //             }
    //             break;
    //         }
    //         case VertexBuffer.ColorKind: {
    //             const meshMaterial = (babylonTransformNode as Mesh).material;
    //             const convertToLinear = meshMaterial ? meshMaterial.getClassName() === "StandardMaterial" : true;
    //             const vertexData: Color3 | Color4 = stride === 3 ? new Color3() : new Color4();
    //             const useExactSrgbConversions = this._babylonScene.getEngine().useExactSrgbConversions;
    //             for (let k = 0, length = meshAttributeArray.length / stride; k < length; ++k) {
    //                 index = k * stride;
    //                 if (stride === 3) {
    //                     Color3.FromArrayToRef(meshAttributeArray, index, vertexData as Color3);
    //                     if (convertToLinear) {
    //                         (vertexData as Color3).toLinearSpaceToRef(vertexData as Color3, useExactSrgbConversions);
    //                     }
    //                 } else {
    //                     Color4.FromArrayToRef(meshAttributeArray, index, vertexData as Color4);
    //                     if (convertToLinear) {
    //                         (vertexData as Color4).toLinearSpaceToRef(vertexData as Color4, useExactSrgbConversions);
    //                     }
    //                 }
    //                 vertexAttributes.push(vertexData.asArray());
    //             }
    //             break;
    //         }
    //         case VertexBuffer.UVKind:
    //         case VertexBuffer.UV2Kind: {
    //             for (let k = 0, length = meshAttributeArray.length / stride; k < length; ++k) {
    //                 index = k * stride;
    //                 const vertexData = Vector2.FromArray(meshAttributeArray, index);
    //                 vertexAttributes.push(vertexData.asArray());
    //             }
    //             break;
    //         }
    //         case VertexBuffer.MatricesIndicesKind:
    //         case VertexBuffer.MatricesIndicesExtraKind: {
    //             for (let k = 0, length = meshAttributeArray.length / stride; k < length; ++k) {
    //                 index = k * stride;
    //                 const vertexData = Vector4.FromArray(meshAttributeArray, index);
    //                 vertexAttributes.push(vertexData.asArray());
    //             }
    //             break;
    //         }
    //         case VertexBuffer.MatricesWeightsKind:
    //         case VertexBuffer.MatricesWeightsExtraKind: {
    //             for (let k = 0, length = meshAttributeArray.length / stride; k < length; ++k) {
    //                 index = k * stride;
    //                 const vertexData = Vector4.FromArray(meshAttributeArray, index);
    //                 vertexAttributes.push(vertexData.asArray());
    //             }
    //             break;
    //         }
    //         default: {
    //             Tools.Warn("Unsupported Vertex Buffer Type: " + vertexBufferKind);
    //             vertexAttributes = [];
    //         }
    //     }

    //     let writeBinaryFunc: (value: number) => void;
    //     switch (attributeComponentKind) {
    //         case AccessorComponentType.UNSIGNED_BYTE: {
    //             writeBinaryFunc = (value) => dataWriter.writeUInt8(value);
    //             break;
    //         }
    //         case AccessorComponentType.UNSIGNED_SHORT: {
    //             writeBinaryFunc = (value) => dataWriter.writeUInt16(value);
    //             break;
    //         }
    //         case AccessorComponentType.UNSIGNED_INT: {
    //             writeBinaryFunc = (value) => dataWriter.writeUInt32(value);
    //             break;
    //         }
    //         case AccessorComponentType.FLOAT: {
    //             writeBinaryFunc = (value) => dataWriter.writeFloat32(value);
    //             break;
    //         }
    //         default: {
    //             Tools.Warn("Unsupported Attribute Component kind: " + attributeComponentKind);
    //             return;
    //         }
    //     }

    //     for (const vertexAttribute of vertexAttributes) {
    //         for (const component of vertexAttribute) {
    //             writeBinaryFunc(component);
    //         }
    //     }
    // }

    // /**
    //  * Writes mesh attribute data to a data buffer
    //  * Returns the bytelength of the data
    //  * @param vertexBufferKind Indicates what kind of vertex data is being passed in
    //  * @param attributeComponentKind
    //  * @param meshPrimitive
    //  * @param morphTarget
    //  * @param meshAttributeArray Array containing the attribute data
    //  * @param morphTargetAttributeArray
    //  * @param stride Specifies the space between data
    //  * @param dataWriter The buffer to write the binary data to
    //  * @param minMax
    //  */
    // public writeMorphTargetAttributeData(
    //     vertexBufferKind: string,
    //     attributeComponentKind: AccessorComponentType,
    //     meshPrimitive: SubMesh,
    //     meshAttributeArray: FloatArray,
    //     morphTargetAttributeArray: FloatArray,
    //     stride: number,
    //     dataWriter: DataWriter,
    //     minMax?: any
    // ) {
    //     let vertexAttributes: number[][] = [];
    //     let index: number;
    //     let difference: Vector3 = new Vector3();
    //     let difference4: Vector4 = new Vector4(0, 0, 0, 0);

    //     switch (vertexBufferKind) {
    //         case VertexBuffer.PositionKind: {
    //             for (let k = meshPrimitive.verticesStart; k < meshPrimitive.verticesCount; ++k) {
    //                 index = meshPrimitive.indexStart + k * stride;
    //                 const vertexData = Vector3.FromArray(meshAttributeArray, index);
    //                 const morphData = Vector3.FromArray(morphTargetAttributeArray, index);
    //                 difference = morphData.subtractToRef(vertexData, difference);
    //                 if (minMax) {
    //                     minMax.min.copyFromFloats(Math.min(difference.x, minMax.min.x), Math.min(difference.y, minMax.min.y), Math.min(difference.z, minMax.min.z));
    //                     minMax.max.copyFromFloats(Math.max(difference.x, minMax.max.x), Math.max(difference.y, minMax.max.y), Math.max(difference.z, minMax.max.z));
    //                 }
    //                 vertexAttributes.push(difference.asArray());
    //             }
    //             break;
    //         }
    //         case VertexBuffer.NormalKind: {
    //             for (let k = meshPrimitive.verticesStart; k < meshPrimitive.verticesCount; ++k) {
    //                 index = meshPrimitive.indexStart + k * stride;
    //                 const vertexData = Vector3.FromArray(meshAttributeArray, index).normalize();
    //                 const morphData = Vector3.FromArray(morphTargetAttributeArray, index).normalize();
    //                 difference = morphData.subtractToRef(vertexData, difference);
    //                 vertexAttributes.push(difference.asArray());
    //             }
    //             break;
    //         }
    //         case VertexBuffer.TangentKind: {
    //             for (let k = meshPrimitive.verticesStart; k < meshPrimitive.verticesCount; ++k) {
    //                 index = meshPrimitive.indexStart + k * (stride + 1);
    //                 const vertexData = Vector4.FromArray(meshAttributeArray, index);
    //                 normalizeTangent(vertexData);
    //                 const morphData = Vector4.FromArray(morphTargetAttributeArray, index);
    //                 normalizeTangent(morphData);
    //                 difference4 = morphData.subtractToRef(vertexData, difference4);
    //                 vertexAttributes.push([difference4.x, difference4.y, difference4.z]);
    //             }
    //             break;
    //         }
    //         default: {
    //             Tools.Warn("Unsupported Vertex Buffer Type: " + vertexBufferKind);
    //             vertexAttributes = [];
    //         }
    //     }

    //     let writeBinaryFunc;
    //     switch (attributeComponentKind) {
    //         case AccessorComponentType.UNSIGNED_BYTE: {
    //             writeBinaryFunc = dataWriter.setUInt8.bind(dataWriter);
    //             break;
    //         }
    //         case AccessorComponentType.UNSIGNED_SHORT: {
    //             writeBinaryFunc = dataWriter.setUInt16.bind(dataWriter);
    //             break;
    //         }
    //         case AccessorComponentType.UNSIGNED_INT: {
    //             writeBinaryFunc = dataWriter.setUInt32.bind(dataWriter);
    //             break;
    //         }
    //         case AccessorComponentType.FLOAT: {
    //             writeBinaryFunc = dataWriter.setFloat32.bind(dataWriter);
    //             break;
    //         }
    //         default: {
    //             Tools.Warn("Unsupported Attribute Component kind: " + attributeComponentKind);
    //             return;
    //         }
    //     }

    //     for (const vertexAttribute of vertexAttributes) {
    //         for (const component of vertexAttribute) {
    //             writeBinaryFunc(component);
    //         }
    //     }
    // }

    private _generateJSON(shouldUseGlb: boolean, bufferByteLength: number, fileName?: string, prettyPrint?: boolean): string {
        const buffer: IBuffer = { byteLength: bufferByteLength };
        let imageName: string;
        let imageData: { data: ArrayBuffer; mimeType: ImageMimeType };
        let bufferView: IBufferView;
        let byteOffset: number = bufferByteLength;

        if (buffer.byteLength) {
            this._glTF.buffers = [buffer];
        }
        if (this._nodes && this._nodes.length) {
            this._glTF.nodes = this._nodes;
        }
        if (this._meshes && this._meshes.length) {
            this._glTF.meshes = this._meshes;
        }
        if (this._scenes && this._scenes.length) {
            this._glTF.scenes = this._scenes;
            this._glTF.scene = 0;
        }
        if (this._cameras && this._cameras.length) {
            this._glTF.cameras = this._cameras;
        }
        if (this._bufferViews && this._bufferViews.length) {
            this._glTF.bufferViews = this._bufferViews;
        }
        if (this._accessors && this._accessors.length) {
            this._glTF.accessors = this._accessors;
        }
        if (this._animations && this._animations.length) {
            this._glTF.animations = this._animations;
        }
        if (this._materials && this._materials.length) {
            this._glTF.materials = this._materials;
        }
        if (this._textures && this._textures.length) {
            this._glTF.textures = this._textures;
        }
        if (this._samplers && this._samplers.length) {
            this._glTF.samplers = this._samplers;
        }
        if (this._skins && this._skins.length) {
            this._glTF.skins = this._skins;
        }
        if (this._images && this._images.length) {
            if (!shouldUseGlb) {
                this._glTF.images = this._images;
            } else {
                this._glTF.images = [];

                this._images.forEach((image) => {
                    if (image.uri) {
                        imageData = this._imageData[image.uri];
                        this._orderedImageData.push(imageData);
                        imageName = image.uri.split(".")[0] + " image";
                        bufferView = createBufferView(0, byteOffset, imageData.data.byteLength, undefined, imageName);
                        byteOffset += imageData.data.byteLength;
                        this._bufferViews.push(bufferView);
                        image.bufferView = this._bufferViews.length - 1;
                        image.name = imageName;
                        image.mimeType = imageData.mimeType;
                        image.uri = undefined;
                        this._glTF.images!.push(image);
                    }
                });

                // Replace uri with bufferview and mime type for glb
                buffer.byteLength = byteOffset;
            }
        }

        if (!shouldUseGlb) {
            buffer.uri = fileName + ".bin";
        }

        return prettyPrint ? JSON.stringify(this._glTF, null, 2) : JSON.stringify(this._glTF);
    }

    /**
     * @internal
     */
    public async generateGLTFAsync(glTFPrefix: string): Promise<GLTFData> {
        const binaryBuffer = await this._generateBinaryAsync();

        this._extensionsOnExporting();
        const jsonText = this._generateJSON(false, binaryBuffer.byteLength, glTFPrefix, true);
        const bin = new Blob([binaryBuffer], { type: "application/octet-stream" });

        const glTFFileName = glTFPrefix + ".gltf";
        const glTFBinFile = glTFPrefix + ".bin";

        const container = new GLTFData();

        container.files[glTFFileName] = jsonText;
        container.files[glTFBinFile] = bin;

        if (this._imageData) {
            for (const image in this._imageData) {
                container.files[image] = new Blob([this._imageData[image].data], { type: this._imageData[image].mimeType });
            }
        }

        return container;
    }

    /**
     * Creates a binary buffer for glTF
     * @returns array buffer for binary data
     */
    private async _generateBinaryAsync(): Promise<Uint8Array> {
        await this._exportSceneAsync();
        return this._dataWriter.getOutputData();
    }

    /**
     * Pads the number to a multiple of 4
     * @param num number to pad
     * @returns padded number
     */
    private _getPadding(num: number): number {
        const remainder = num % 4;
        const padding = remainder === 0 ? remainder : 4 - remainder;

        return padding;
    }

    /**
     * @internal
     */
    public async generateGLBAsync(glTFPrefix: string): Promise<GLTFData> {
        const binaryBuffer = await this._generateBinaryAsync();

        this._extensionsOnExporting();
        const jsonText = this._generateJSON(true, binaryBuffer.byteLength);
        const glbFileName = glTFPrefix + ".glb";
        const headerLength = 12;
        const chunkLengthPrefix = 8;
        let jsonLength = jsonText.length;
        let encodedJsonText;
        let imageByteLength = 0;
        // make use of TextEncoder when available
        if (typeof TextEncoder !== "undefined") {
            const encoder = new TextEncoder();
            encodedJsonText = encoder.encode(jsonText);
            jsonLength = encodedJsonText.length;
        }
        for (let i = 0; i < this._orderedImageData.length; ++i) {
            imageByteLength += this._orderedImageData[i].data.byteLength;
        }
        const jsonPadding = this._getPadding(jsonLength);
        const binPadding = this._getPadding(binaryBuffer.byteLength);
        const imagePadding = this._getPadding(imageByteLength);

        const byteLength = headerLength + 2 * chunkLengthPrefix + jsonLength + jsonPadding + binaryBuffer.byteLength + binPadding + imageByteLength + imagePadding;

        // header
        const headerBuffer = new ArrayBuffer(headerLength);
        const headerBufferView = new DataView(headerBuffer);
        headerBufferView.setUint32(0, 0x46546c67, true); //glTF
        headerBufferView.setUint32(4, 2, true); // version
        headerBufferView.setUint32(8, byteLength, true); // total bytes in file

        // json chunk
        const jsonChunkBuffer = new ArrayBuffer(chunkLengthPrefix + jsonLength + jsonPadding);
        const jsonChunkBufferView = new DataView(jsonChunkBuffer);
        jsonChunkBufferView.setUint32(0, jsonLength + jsonPadding, true);
        jsonChunkBufferView.setUint32(4, 0x4e4f534a, true);

        // json chunk bytes
        const jsonData = new Uint8Array(jsonChunkBuffer, chunkLengthPrefix);
        // if TextEncoder was available, we can simply copy the encoded array
        if (encodedJsonText) {
            jsonData.set(encodedJsonText);
        } else {
            const blankCharCode = "_".charCodeAt(0);
            for (let i = 0; i < jsonLength; ++i) {
                const charCode = jsonText.charCodeAt(i);
                // if the character doesn't fit into a single UTF-16 code unit, just put a blank character
                if (charCode != jsonText.codePointAt(i)) {
                    jsonData[i] = blankCharCode;
                } else {
                    jsonData[i] = charCode;
                }
            }
        }

        // json padding
        const jsonPaddingView = new Uint8Array(jsonChunkBuffer, chunkLengthPrefix + jsonLength);
        for (let i = 0; i < jsonPadding; ++i) {
            jsonPaddingView[i] = 0x20;
        }

        // binary chunk
        const binaryChunkBuffer = new ArrayBuffer(chunkLengthPrefix);
        const binaryChunkBufferView = new DataView(binaryChunkBuffer);
        binaryChunkBufferView.setUint32(0, binaryBuffer.byteLength + imageByteLength + imagePadding, true);
        binaryChunkBufferView.setUint32(4, 0x004e4942, true);

        // binary padding
        const binPaddingBuffer = new ArrayBuffer(binPadding);
        const binPaddingView = new Uint8Array(binPaddingBuffer);
        for (let i = 0; i < binPadding; ++i) {
            binPaddingView[i] = 0;
        }

        const imagePaddingBuffer = new ArrayBuffer(imagePadding);
        const imagePaddingView = new Uint8Array(imagePaddingBuffer);
        for (let i = 0; i < imagePadding; ++i) {
            imagePaddingView[i] = 0;
        }

        const glbData = [headerBuffer, jsonChunkBuffer, binaryChunkBuffer, binaryBuffer];

        // binary data
        for (let i = 0; i < this._orderedImageData.length; ++i) {
            glbData.push(this._orderedImageData[i].data);
        }

        glbData.push(binPaddingBuffer);

        glbData.push(imagePaddingBuffer);

        const glbFile = new Blob(glbData, { type: "application/octet-stream" });

        const container = new GLTFData();
        container.files[glbFileName] = glbFile;

        return container;
    }

    private _setNodeTransformation(node: INode, babylonTransformNode: TransformNode): void {
        if (!babylonTransformNode.getPivotPoint().equalsToFloats(0, 0, 0)) {
            Tools.Warn("Pivot points are not supported in the glTF serializer");
        }

        if (!babylonTransformNode.position.equalsToFloats(0, 0, 0)) {
            const translation = TmpVectors.Vector3[0].copyFrom(babylonTransformNode.position);
            if (this._shouldConvertToRightHanded(babylonTransformNode)) {
                convertToRightHandedPosition(translation);
            }

            node.translation = translation.asArray();
        }

        if (!babylonTransformNode.scaling.equalsToFloats(1, 1, 1)) {
            node.scale = babylonTransformNode.scaling.asArray();
        }

        const rotationQuaternion = Quaternion.FromEulerAngles(babylonTransformNode.rotation.x, babylonTransformNode.rotation.y, babylonTransformNode.rotation.z);
        if (babylonTransformNode.rotationQuaternion) {
            rotationQuaternion.multiplyInPlace(babylonTransformNode.rotationQuaternion);
        }
        if (!Quaternion.IsIdentity(rotationQuaternion)) {
            if (this._shouldConvertToRightHanded(babylonTransformNode)) {
                convertToRightHandedRotation(rotationQuaternion);
            }

            node.rotation = rotationQuaternion.normalize().asArray();
        }
    }

    // private _setCameraTransformation(node: INode, babylonCamera: Camera, convertToRightHanded: boolean): void {
    //     const translation = TmpVectors.Vector3[0];
    //     const rotation = TmpVectors.Quaternion[0];
    //     babylonCamera.getWorldMatrix().decompose(undefined, rotation, translation);

    //     if (!translation.equalsToFloats(0, 0, 0)) {
    //         if (convertToRightHanded) {
    //             convertToRightHandedPosition(translation);
    //         }

    //         node.translation = translation.asArray();
    //     }

    //     // Rotation by 180 as glTF has a different convention than Babylon.
    //     rotation.multiplyInPlace(rotation180Y);

    //     if (!Quaternion.IsIdentity(rotation)) {
    //         if (convertToRightHanded) {
    //             convertToRightHandedRotation(rotation);
    //         }

    //         node.rotation = rotation.asArray();
    //     }
    // }

    // /**
    //  * Creates a bufferview based on the vertices type for the Babylon mesh
    //  * @param babylonSubMesh The Babylon submesh that the morph target is applied to
    //  * @param meshPrimitive
    //  * @param babylonMorphTarget the morph target to be exported
    //  * @param dataWriter The buffer to write the bufferview data to
    //  */
    // private _setMorphTargetAttributes(babylonSubMesh: SubMesh, meshPrimitive: IMeshPrimitive, babylonMorphTarget: MorphTarget, dataWriter: DataWriter) {
    //     if (babylonMorphTarget) {
    //         if (!meshPrimitive.targets) {
    //             meshPrimitive.targets = [];
    //         }
    //         const target: { [attribute: string]: number } = {};
    //         const mesh = babylonSubMesh.getMesh() as Mesh;
    //         if (babylonMorphTarget.hasNormals) {
    //             const vertexNormals = mesh.getVerticesData(VertexBuffer.NormalKind, undefined, undefined, true)!;
    //             const morphNormals = babylonMorphTarget.getNormals()!;
    //             const count = babylonSubMesh.verticesCount;
    //             const byteStride = 12; // 3 x 4 byte floats
    //             const byteLength = count * byteStride;
    //             const bufferView = createBufferView(0, dataWriter.getByteOffset(), byteLength, byteStride, babylonMorphTarget.name + "_NORMAL");
    //             this._bufferViews.push(bufferView);

    //             const bufferViewIndex = this._bufferViews.length - 1;
    //             const accessor = createAccessor(bufferViewIndex, babylonMorphTarget.name + " - " + "NORMAL", AccessorType.VEC3, AccessorComponentType.FLOAT, count, 0, null, null);
    //             this._accessors.push(accessor);
    //             target.NORMAL = this._accessors.length - 1;

    //             this.writeMorphTargetAttributeData(VertexBuffer.NormalKind, AccessorComponentType.FLOAT, babylonSubMesh, vertexNormals, morphNormals, byteStride / 4, dataWriter);
    //         }
    //         if (babylonMorphTarget.hasPositions) {
    //             const vertexPositions = mesh.getVerticesData(VertexBuffer.PositionKind, undefined, undefined, true)!;
    //             const morphPositions = babylonMorphTarget.getPositions()!;
    //             const count = babylonSubMesh.verticesCount;
    //             const byteStride = 12; // 3 x 4 byte floats
    //             const byteLength = count * byteStride;
    //             const bufferView = createBufferView(0, dataWriter.getByteOffset(), byteLength, byteStride, babylonMorphTarget.name + "_POSITION");
    //             this._bufferViews.push(bufferView);

    //             const bufferViewIndex = this._bufferViews.length - 1;
    //             const minMax = { min: new Vector3(Infinity, Infinity, Infinity), max: new Vector3(-Infinity, -Infinity, -Infinity) };
    //             const accessor = createAccessor(
    //                 bufferViewIndex,
    //                 babylonMorphTarget.name + " - " + "POSITION",
    //                 AccessorType.VEC3,
    //                 AccessorComponentType.FLOAT,
    //                 count,
    //                 0,
    //                 null,
    //                 null
    //             );
    //             this._accessors.push(accessor);
    //             target.POSITION = this._accessors.length - 1;

    //             this.writeMorphTargetAttributeData(
    //                 VertexBuffer.PositionKind,
    //                 AccessorComponentType.FLOAT,
    //                 babylonSubMesh,
    //                 vertexPositions,
    //                 morphPositions,
    //                 byteStride / 4,
    //                 dataWriter,
    //                 minMax
    //             );
    //             accessor.min = minMax.min!.asArray();
    //             accessor.max = minMax.max!.asArray();
    //         }
    //         if (babylonMorphTarget.hasTangents) {
    //             const vertexTangents = mesh.getVerticesData(VertexBuffer.TangentKind, undefined, undefined, true)!;
    //             const morphTangents = babylonMorphTarget.getTangents()!;
    //             const count = babylonSubMesh.verticesCount;
    //             const byteStride = 12; // 3 x 4 byte floats
    //             const byteLength = count * byteStride;
    //             const bufferView = createBufferView(0, dataWriter.getByteOffset(), byteLength, byteStride, babylonMorphTarget.name + "_NORMAL");
    //             this._bufferViews.push(bufferView);

    //             const bufferViewIndex = this._bufferViews.length - 1;
    //             const accessor = createAccessor(bufferViewIndex, babylonMorphTarget.name + " - " + "TANGENT", AccessorType.VEC3, AccessorComponentType.FLOAT, count, 0, null, null);
    //             this._accessors.push(accessor);
    //             target.TANGENT = this._accessors.length - 1;

    //             this.writeMorphTargetAttributeData(
    //                 VertexBuffer.TangentKind,
    //                 AccessorComponentType.FLOAT,
    //                 babylonSubMesh,
    //                 vertexTangents,
    //                 morphTangents,
    //                 byteStride / 4,
    //                 dataWriter
    //             );
    //         }
    //         meshPrimitive.targets.push(target);
    //     }
    // }

    // /**
    //  * The primitive mode of the Babylon mesh
    //  * @param babylonMesh The BabylonJS mesh
    //  */
    // private _getMeshPrimitiveMode(babylonMesh: AbstractMesh): number {
    //     if (babylonMesh instanceof LinesMesh) {
    //         return Material.LineListDrawMode;
    //     }
    //     if (babylonMesh instanceof InstancedMesh || babylonMesh instanceof Mesh) {
    //         const baseMesh = babylonMesh instanceof Mesh ? babylonMesh : babylonMesh.sourceMesh;
    //         if (typeof baseMesh.overrideRenderingFillMode === "number") {
    //             return baseMesh.overrideRenderingFillMode;
    //         }
    //     }
    //     return babylonMesh.material ? babylonMesh.material.fillMode : Material.TriangleFillMode;
    // }

    // TODO: move to utilities
    // /**
    //  * Sets the primitive mode of the glTF mesh primitive
    //  * @param meshPrimitive glTF mesh primitive
    //  * @param primitiveMode The primitive mode
    //  */
    // private _setPrimitiveMode(meshPrimitive: IMeshPrimitive, primitiveMode: number) {
    //     switch (primitiveMode) {
    //         case Material.TriangleFillMode: {
    //             // glTF defaults to using Triangle Mode
    //             break;
    //         }
    //         case Material.TriangleStripDrawMode: {
    //             meshPrimitive.mode = MeshPrimitiveMode.TRIANGLE_STRIP;
    //             break;
    //         }
    //         case Material.TriangleFanDrawMode: {
    //             meshPrimitive.mode = MeshPrimitiveMode.TRIANGLE_FAN;
    //             break;
    //         }
    //         case Material.PointListDrawMode: {
    //             meshPrimitive.mode = MeshPrimitiveMode.POINTS;
    //             break;
    //         }
    //         case Material.PointFillMode: {
    //             meshPrimitive.mode = MeshPrimitiveMode.POINTS;
    //             break;
    //         }
    //         case Material.LineLoopDrawMode: {
    //             meshPrimitive.mode = MeshPrimitiveMode.LINE_LOOP;
    //             break;
    //         }
    //         case Material.LineListDrawMode: {
    //             meshPrimitive.mode = MeshPrimitiveMode.LINES;
    //             break;
    //         }
    //         case Material.LineStripDrawMode: {
    //             meshPrimitive.mode = MeshPrimitiveMode.LINE_STRIP;
    //             break;
    //         }
    //     }
    // }

    // /**
    //  * Sets the vertex attribute accessor based of the glTF mesh primitive
    //  * @param meshPrimitive glTF mesh primitive
    //  * @param attributeKind vertex attribute
    //  * @returns boolean specifying if uv coordinates are present
    //  */
    // private _setAttributeKind(meshPrimitive: IMeshPrimitive, attributeKind: string): void {
    //     switch (attributeKind) {
    //         case VertexBuffer.PositionKind: {
    //             meshPrimitive.attributes.POSITION = this._accessors.length - 1;
    //             break;
    //         }
    //         case VertexBuffer.NormalKind: {
    //             meshPrimitive.attributes.NORMAL = this._accessors.length - 1;
    //             break;
    //         }
    //         case VertexBuffer.ColorKind: {
    //             meshPrimitive.attributes.COLOR_0 = this._accessors.length - 1;
    //             break;
    //         }
    //         case VertexBuffer.TangentKind: {
    //             meshPrimitive.attributes.TANGENT = this._accessors.length - 1;
    //             break;
    //         }
    //         case VertexBuffer.UVKind: {
    //             meshPrimitive.attributes.TEXCOORD_0 = this._accessors.length - 1;
    //             break;
    //         }
    //         case VertexBuffer.UV2Kind: {
    //             meshPrimitive.attributes.TEXCOORD_1 = this._accessors.length - 1;
    //             break;
    //         }
    //         case VertexBuffer.MatricesIndicesKind: {
    //             meshPrimitive.attributes.JOINTS_0 = this._accessors.length - 1;
    //             break;
    //         }
    //         case VertexBuffer.MatricesIndicesExtraKind: {
    //             meshPrimitive.attributes.JOINTS_1 = this._accessors.length - 1;
    //             break;
    //         }
    //         case VertexBuffer.MatricesWeightsKind: {
    //             meshPrimitive.attributes.WEIGHTS_0 = this._accessors.length - 1;
    //             break;
    //         }
    //         case VertexBuffer.MatricesWeightsExtraKind: {
    //             meshPrimitive.attributes.WEIGHTS_1 = this._accessors.length - 1;
    //             break;
    //         }
    //         default: {
    //             Tools.Warn("Unsupported Vertex Buffer Type: " + attributeKind);
    //         }
    //     }
    // }

    // /**
    //  * Sets data for the primitive attributes of each submesh
    //  * @param mesh glTF Mesh object to store the primitive attribute information
    //  * @param babylonTransformNode Babylon mesh to get the primitive attribute data from
    //  * @param convertToRightHanded Whether to convert from left-handed to right-handed
    //  * @param dataWriter Buffer to write the attribute data to
    //  */
    // private _setPrimitiveAttributesAsync(mesh: IMesh, babylonTransformNode: TransformNode, convertToRightHanded: boolean, dataWriter: DataWriter): Promise<void> {
    //     const promises: Promise<IMeshPrimitive>[] = [];
    //     let bufferMesh: Nullable<Mesh> = null;
    //     let bufferView: IBufferView;
    //     let minMax: { min: Nullable<number[]>; max: Nullable<number[]> };

    //     if (babylonTransformNode instanceof Mesh) {
    //         bufferMesh = babylonTransformNode as Mesh;
    //     } else if (babylonTransformNode instanceof InstancedMesh) {
    //         bufferMesh = (babylonTransformNode as InstancedMesh).sourceMesh;
    //     }
    //     const attributeData: _IVertexAttributeData[] = [
    //         { kind: VertexBuffer.PositionKind, accessorType: AccessorType.VEC3, accessorComponentType: AccessorComponentType.FLOAT, byteStride: 12 },
    //         { kind: VertexBuffer.NormalKind, accessorType: AccessorType.VEC3, accessorComponentType: AccessorComponentType.FLOAT, byteStride: 12 },
    //         { kind: VertexBuffer.ColorKind, accessorType: AccessorType.VEC4, accessorComponentType: AccessorComponentType.FLOAT, byteStride: 16 },
    //         { kind: VertexBuffer.TangentKind, accessorType: AccessorType.VEC4, accessorComponentType: AccessorComponentType.FLOAT, byteStride: 16 },
    //         { kind: VertexBuffer.UVKind, accessorType: AccessorType.VEC2, accessorComponentType: AccessorComponentType.FLOAT, byteStride: 8 },
    //         { kind: VertexBuffer.UV2Kind, accessorType: AccessorType.VEC2, accessorComponentType: AccessorComponentType.FLOAT, byteStride: 8 },
    //         { kind: VertexBuffer.MatricesIndicesKind, accessorType: AccessorType.VEC4, accessorComponentType: AccessorComponentType.UNSIGNED_SHORT, byteStride: 8 },
    //         { kind: VertexBuffer.MatricesIndicesExtraKind, accessorType: AccessorType.VEC4, accessorComponentType: AccessorComponentType.UNSIGNED_SHORT, byteStride: 8 },
    //         { kind: VertexBuffer.MatricesWeightsKind, accessorType: AccessorType.VEC4, accessorComponentType: AccessorComponentType.FLOAT, byteStride: 16 },
    //         { kind: VertexBuffer.MatricesWeightsExtraKind, accessorType: AccessorType.VEC4, accessorComponentType: AccessorComponentType.FLOAT, byteStride: 16 },
    //     ];

    //     if (bufferMesh) {
    //         let indexBufferViewIndex: Nullable<number> = null;
    //         const primitiveMode = this._getMeshPrimitiveMode(bufferMesh);
    //         const vertexAttributeBufferViews: { [attributeKind: string]: number } = {};
    //         const morphTargetManager = bufferMesh.morphTargetManager;

    //         // For each BabylonMesh, create bufferviews for each 'kind'
    //         for (const attribute of attributeData) {
    //             const attributeKind = attribute.kind;
    //             const attributeComponentKind = attribute.accessorComponentType;
    //             if (bufferMesh.isVerticesDataPresent(attributeKind, true)) {
    //                 const vertexBuffer = this._getVertexBuffer(attributeKind, bufferMesh);
    //                 attribute.byteStride = vertexBuffer
    //                     ? vertexBuffer.getSize() * VertexBuffer.GetTypeByteLength(attribute.accessorComponentType)
    //                     : VertexBuffer.DeduceStride(attributeKind) * 4;
    //                 if (attribute.byteStride === 12) {
    //                     attribute.accessorType = AccessorType.VEC3;
    //                 }

    //                 this._createBufferViewKind(attributeKind, attributeComponentKind, babylonTransformNode, dataWriter, attribute.byteStride);
    //                 attribute.bufferViewIndex = this._bufferViews.length - 1;
    //                 vertexAttributeBufferViews[attributeKind] = attribute.bufferViewIndex;
    //             }
    //         }

    //         if (bufferMesh.getTotalIndices()) {
    //             const indices = bufferMesh.getIndices();
    //             if (indices) {
    //                 const byteLength = indices.length * 4;
    //                 bufferView = createBufferView(0, dataWriter.getByteOffset(), byteLength, undefined, "Indices - " + bufferMesh.name);
    //                 this._bufferViews.push(bufferView);
    //                 indexBufferViewIndex = this._bufferViews.length - 1;

    //                 for (let k = 0, length = indices.length; k < length; ++k) {
    //                     dataWriter.setUInt32(indices[k]);
    //                 }
    //             }
    //         }

    //         if (bufferMesh.subMeshes) {
    //             // go through all mesh primitives (submeshes)
    //             for (const submesh of bufferMesh.subMeshes) {
    //                 let babylonMaterial = submesh.getMaterial() || bufferMesh.getScene().defaultMaterial;

    //                 let materialIndex: Nullable<number> = null;
    //                 if (babylonMaterial) {
    //                     if (bufferMesh instanceof LinesMesh) {
    //                         // get the color from the lines mesh and set it in the material
    //                         const material: IMaterial = {
    //                             name: bufferMesh.name + " material",
    //                         };
    //                         if (!bufferMesh.color.equals(Color3.White()) || bufferMesh.alpha < 1) {
    //                             material.pbrMetallicRoughness = {
    //                                 baseColorFactor: bufferMesh.color.asArray().concat([bufferMesh.alpha]),
    //                             };
    //                         }
    //                         this._materials.push(material);
    //                         materialIndex = this._materials.length - 1;
    //                     } else if (babylonMaterial instanceof MultiMaterial) {
    //                         const subMaterial = babylonMaterial.subMaterials[submesh.materialIndex];
    //                         if (subMaterial) {
    //                             babylonMaterial = subMaterial;
    //                             materialIndex = this._materialMap[babylonMaterial.uniqueId];
    //                         }
    //                     } else {
    //                         materialIndex = this._materialMap[babylonMaterial.uniqueId];
    //                     }
    //                 }

    //                 const glTFMaterial: Nullable<IMaterial> = materialIndex != null ? this._materials[materialIndex] : null;

    //                 const meshPrimitive: IMeshPrimitive = { attributes: {} };
    //                 this._setPrimitiveMode(meshPrimitive, primitiveMode);

    //                 for (const attribute of attributeData) {
    //                     const attributeKind = attribute.kind;
    //                     if ((attributeKind === VertexBuffer.UVKind || attributeKind === VertexBuffer.UV2Kind) && !this._options.exportUnusedUVs) {
    //                         if (!glTFMaterial || !this._glTFMaterialExporter._hasTexturesPresent(glTFMaterial)) {
    //                             continue;
    //                         }
    //                     }
    //                     const vertexData = bufferMesh.getVerticesData(attributeKind, undefined, undefined, true);
    //                     if (vertexData) {
    //                         const vertexBuffer = this._getVertexBuffer(attributeKind, bufferMesh);
    //                         if (vertexBuffer) {
    //                             const stride = vertexBuffer.getSize();
    //                             const bufferViewIndex = attribute.bufferViewIndex;
    //                             if (bufferViewIndex != undefined) {
    //                                 // check to see if bufferviewindex has a numeric value assigned.
    //                                 minMax = { min: null, max: null };
    //                                 if (attributeKind == VertexBuffer.PositionKind) {
    //                                     minMax = calculateMinMaxPositions(vertexData, 0, vertexData.length / stride);
    //                                 }
    //                                 const accessor = createAccessor(
    //                                     bufferViewIndex,
    //                                     attributeKind + " - " + babylonTransformNode.name,
    //                                     attribute.accessorType,
    //                                     attribute.accessorComponentType,
    //                                     vertexData.length / stride,
    //                                     0,
    //                                     minMax.min,
    //                                     minMax.max
    //                                 );
    //                                 this._accessors.push(accessor);
    //                                 this._setAttributeKind(meshPrimitive, attributeKind);
    //                             }
    //                         }
    //                     }
    //                 }

    //                 if (indexBufferViewIndex) {
    //                     // Create accessor
    //                     const accessor = createAccessor(
    //                         indexBufferViewIndex,
    //                         "indices - " + babylonTransformNode.name,
    //                         AccessorType.SCALAR,
    //                         AccessorComponentType.UNSIGNED_INT,
    //                         submesh.indexCount,
    //                         submesh.indexStart * 4,
    //                         null,
    //                         null
    //                     );
    //                     this._accessors.push(accessor);
    //                     meshPrimitive.indices = this._accessors.length - 1;
    //                 }

    //                 if (Object.keys(meshPrimitive.attributes).length > 0) {
    //                     const sideOrientation = bufferMesh.overrideMaterialSideOrientation !== null ? bufferMesh.overrideMaterialSideOrientation : babylonMaterial.sideOrientation;

    //                     if (sideOrientation === (this._babylonScene.useRightHandedSystem ? Material.ClockWiseSideOrientation : Material.CounterClockWiseSideOrientation)) {
    //                         let byteOffset = indexBufferViewIndex != null ? this._bufferViews[indexBufferViewIndex].byteOffset : null;
    //                         if (byteOffset == null) {
    //                             byteOffset = 0;
    //                         }
    //                         let babylonIndices: Nullable<IndicesArray> = null;
    //                         if (indexBufferViewIndex != null) {
    //                             babylonIndices = bufferMesh.getIndices();
    //                         }
    //                         if (babylonIndices) {
    //                             this._reorderIndicesBasedOnPrimitiveMode(submesh, primitiveMode, babylonIndices, byteOffset, dataWriter);
    //                         } else {
    //                             for (const attribute of attributeData) {
    //                                 const vertexData = bufferMesh.getVerticesData(attribute.kind, undefined, undefined, true);
    //                                 if (vertexData) {
    //                                     const byteOffset = this._bufferViews[vertexAttributeBufferViews[attribute.kind]].byteOffset || 0;
    //                                     this._reorderVertexAttributeDataBasedOnPrimitiveMode(submesh, primitiveMode, attribute.kind, vertexData, byteOffset, dataWriter);
    //                                 }
    //                             }
    //                         }
    //                     }

    //                     if (materialIndex != null) {
    //                         meshPrimitive.material = materialIndex;
    //                     }
    //                 }
    //                 if (morphTargetManager) {
    //                     // By convention, morph target names are stored in the mesh extras.
    //                     if (!mesh.extras) {
    //                         mesh.extras = {};
    //                     }
    //                     mesh.extras.targetNames = [];

    //                     for (let i = 0; i < morphTargetManager.numTargets; ++i) {
    //                         const target = morphTargetManager.getTarget(i);
    //                         this._setMorphTargetAttributes(submesh, meshPrimitive, target, dataWriter);
    //                         mesh.extras.targetNames.push(target.name);
    //                     }
    //                 }

    //                 mesh.primitives.push(meshPrimitive);

    //                 this._extensionsPostExportMeshPrimitiveAsync("postExport", meshPrimitive, submesh, dataWriter);
    //                 promises.push();
    //             }
    //         }
    //     }
    //     return Promise.all(promises).then(() => {
    //         /* do nothing */
    //     });
    // }

    private async _exportSceneAsync(): Promise<void> {
        const scene: IScene = { nodes: [] };

        const shouldExportNode = this._options.shouldExportNode || (() => true);
        const transformNodes = this._babylonScene.transformNodes.filter(shouldExportNode);
        const meshes = this._babylonScene.meshes.filter(shouldExportNode);
        const lights = this._babylonScene.lights.filter(shouldExportNode);
        const cameras = this._babylonScene.cameras.filter(shouldExportNode);
        const nodes: Node[] = [...transformNodes, ...meshes, ...lights, ...cameras];
        const removedRootNodes = new Set<Node>();

        // Scene metadata
        if (this._babylonScene.metadata) {
            if (this._options.metadataSelector) {
                scene.extras = this._options.metadataSelector(this._babylonScene.metadata);
            } else if (this._babylonScene.metadata.gltf) {
                scene.extras = this._babylonScene.metadata.gltf.extras;
            }
        }

        // Assume all nodes must be converted to right-handed when scene is using left-handed system initially.
        if (!this._babylonScene.useRightHandedSystem) {
            for (const node of nodes) {
                this._nodesToConvertToRightHanded.add(node);
            }
        }

        // Remove no-op root nodes
        if ((this._options.removeNoopRootNodes ?? true) && !this._options.includeCoordinateSystemConversionNodes) {
            for (const rootNode of this._babylonScene.rootNodes) {
                if (isNoopNode(rootNode, this._babylonScene.useRightHandedSystem)) {
                    removedRootNodes.add(rootNode);

                    // Exclude the node from list of nodes to export
                    nodes.splice(nodes.indexOf(rootNode), 1);

                    if (!this._babylonScene.useRightHandedSystem) {
                        // Cancel conversion to right handed system
                        this._nodesToConvertToRightHanded.delete(rootNode);
                        rootNode.getDescendants().forEach((descendant) => {
                            this._nodesToConvertToRightHanded.delete(descendant);
                        });
                    }
                }
            }
        }

        // // Export babylon cameras to glTF cameras
        // const cameraMap = new Map<Camera, number>();
        // for (const camera of cameras) {
        //     const glTFCamera: ICamera = {
        //         type: camera.mode === Camera.PERSPECTIVE_CAMERA ? CameraType.PERSPECTIVE : CameraType.ORTHOGRAPHIC,
        //     };

        //     if (camera.name) {
        //         glTFCamera.name = camera.name;
        //     }

        //     if (glTFCamera.type === CameraType.PERSPECTIVE) {
        //         glTFCamera.perspective = {
        //             aspectRatio: camera.getEngine().getAspectRatio(camera),
        //             yfov: camera.fovMode === Camera.FOVMODE_VERTICAL_FIXED ? camera.fov : camera.fov * camera.getEngine().getAspectRatio(camera),
        //             znear: camera.minZ,
        //             zfar: camera.maxZ,
        //         };
        //     } else if (glTFCamera.type === CameraType.ORTHOGRAPHIC) {
        //         const halfWidth = camera.orthoLeft && camera.orthoRight ? 0.5 * (camera.orthoRight - camera.orthoLeft) : camera.getEngine().getRenderWidth() * 0.5;
        //         const halfHeight = camera.orthoBottom && camera.orthoTop ? 0.5 * (camera.orthoTop - camera.orthoBottom) : camera.getEngine().getRenderHeight() * 0.5;
        //         glTFCamera.orthographic = {
        //             xmag: halfWidth,
        //             ymag: halfHeight,
        //             znear: camera.minZ,
        //             zfar: camera.maxZ,
        //         };
        //     }

        //     cameraMap.set(camera, this._cameras.length);
        //     this._cameras.push(glTFCamera);
        // }

        // await this._materialExporter.convertMaterialsToGLTFAsync(this._getMaterials(nodes));

        await this._exportNodesAsync(nodes);

        for (const babylonNode of nodes) {
            const nodeIndex = this._nodeMap.get(babylonNode)!;

            const parentNodeIndex = babylonNode.parent && this._nodeMap.get(babylonNode.parent);
            if (parentNodeIndex != null) {
                const parentNode = this._nodes[parentNodeIndex];
                parentNode.children ||= [];
                parentNode.children.push(nodeIndex);
            } else {
                scene.nodes.push(nodeIndex);
            }
        }

        this._scenes.push(scene);

        //     return this._exportNodesAndAnimationsAsync(nodes, convertToRightHandedMap, dataWriter).then((nodeMap) => {
        //         return this._createSkinsAsync(nodeMap, dataWriter).then((skinMap) => {
        //             for (const babylonNode of nodes) {
        //                 const glTFNodeIndex = nodeMap[babylonNode.uniqueId];
        //                 if (glTFNodeIndex !== undefined) {
        //                     const glTFNode = this._nodes[glTFNodeIndex];

        //                     if (babylonNode.metadata) {
        //                         if (this._options.metadataSelector) {
        //                             glTFNode.extras = this._options.metadataSelector(babylonNode.metadata);
        //                         } else if (babylonNode.metadata.gltf) {
        //                             glTFNode.extras = babylonNode.metadata.gltf.extras;
        //                         }
        //                     }

        //                     if (babylonNode instanceof Camera) {
        //                         glTFNode.camera = cameraMap.get(babylonNode);
        //                     }

        //                     if (!babylonNode.parent || removedRootNodes.has(babylonNode.parent)) {
        //                         scene.nodes.push(glTFNodeIndex);
        //                     }

        //                     if (babylonNode instanceof Mesh) {
        //                         if (babylonNode.skeleton) {
        //                             glTFNode.skin = skinMap[babylonNode.skeleton.uniqueId];
        //                         }
        //                     }

        //                     const directDescendents = babylonNode.getDescendants(true);
        //                     if (!glTFNode.children && directDescendents && directDescendents.length) {
        //                         const children: number[] = [];
        //                         for (const descendent of directDescendents) {
        //                             if (nodeMap[descendent.uniqueId] != null) {
        //                                 children.push(nodeMap[descendent.uniqueId]);
        //                             }
        //                         }
        //                         if (children.length) {
        //                             glTFNode.children = children;
        //                         }
        //                     }
        //                 }
        //             }

        //             if (scene.nodes.length) {
        //                 this._scenes.push(scene);
        //             }
        //         });
        //     });
        // });
    }

    private _shouldConvertToRightHanded(babylonNode: Node): boolean {
        return this._nodesToConvertToRightHanded.has(babylonNode);
    }

    private _getState(babylonNode: Node) {
        return this._shouldConvertToRightHanded(babylonNode) ? this._convertToRightHandedState : this._originalState;
    }

    // private _getMaterials(nodes: Node[]): Material[] {
    //     const materials = new Set<Material>();

    //     for (const babylonNode of nodes) {
    //         const babylonMesh = babylonNode as AbstractMesh;
    //         if (babylonMesh.subMeshes && babylonMesh.subMeshes.length > 0) {
    //             const material = babylonMesh.material || babylonMesh.getScene().defaultMaterial;
    //             if (material instanceof MultiMaterial) {
    //                 for (const subMaterial of material.subMaterials) {
    //                     if (subMaterial) {
    //                         materials.add(subMaterial);
    //                     }
    //                 }
    //             } else {
    //                 materials.add(material);
    //             }
    //         }
    //     }

    //     return Array.from(materials);
    // }

    private _exportIndicesData(indices: IndicesArray, is32Bits: boolean, convertToRightHanded: boolean): number {
        const byteOffset = this._dataWriter.byteOffset;

        const bytes = indicesArrayToUint8Array(indices, is32Bits);

        // TODO: left-hand to right-hand conversion
        convertToRightHanded;

        this._dataWriter.writeUint8Array(bytes);

        this._bufferViews.push(createBufferView(0, byteOffset, bytes.length));
        return this._bufferViews.length - 1;
    }

    private _exportIndices(bufferViewIndex: number, is32Bits: boolean, start: number, count: number): number {
        const byteStride = is32Bits ? 4 : 2;
        const componentType = is32Bits ? AccessorComponentType.UNSIGNED_INT : AccessorComponentType.UNSIGNED_SHORT;
        const byteOffset = start * byteStride;
        this._accessors.push(createAccessor(bufferViewIndex, AccessorType.SCALAR, componentType, count, byteOffset, null, null));
        return this._accessors.length - 1;
    }

    private _exportBuffer(buffer: Buffer, byteStride: number, convertToRightHanded: boolean): number {
        const data = buffer.getData();
        if (!data) {
            throw new Error("Buffer data is not available");
        }

        // TODO: check buffer.instanced and buffer.divisor to see if they will just work

        const byteOffset = this._dataWriter.byteOffset;

        const bytes = dataArrayToUint8Array(data);

        // TODO: left-hand to right-hand conversion
        convertToRightHanded;

        this._dataWriter.writeUint8Array(bytes);

        this._bufferViews.push(createBufferView(0, byteOffset, bytes.length, byteStride));
        return this._bufferViews.length - 1;
    }

    private _exportVertexBuffer(vertexBuffer: VertexBuffer, start: number, count: number, bufferViewIndex: number): number {
        const kind = vertexBuffer.getKind();

        let min: Nullable<number[]> = null;
        let max: Nullable<number[]> = null;

        if (kind === VertexBuffer.PositionKind) {
            const positionsData = vertexBuffer.getFloatData();
            if (!positionsData) {
                throw new Error("Missing position data");
            }

            [min, max] = calculateMinMaxPositions(positionsData, start, count);
        }

        const byteOffset = vertexBuffer.byteOffset + start * vertexBuffer.byteStride;
        this._accessors.push(createAccessor(bufferViewIndex, getAccessorType(kind), vertexBuffer.type, count, byteOffset, min, max));
        return this._accessors.length - 1;
    }

    private async _exportMaterialAsync(babylonMaterial: Material, hasUVs: boolean): Promise<number> {
        if (babylonMaterial instanceof PBRMaterial) {
            return await this._materialExporter.exportPBRMaterialAsync(babylonMaterial, ImageMimeType.PNG, hasUVs);
        } else if (babylonMaterial instanceof StandardMaterial) {
            return await this._materialExporter.exportStandardMaterialAsync(babylonMaterial, ImageMimeType.PNG, hasUVs);
        }

        Logger.Warn(`Unsupported material '${babylonMaterial.name}' with type ${babylonMaterial.getClassName()}`);
        return -1;
    }

    //     if (babylonMesh instanceof LinesMesh) {
    //         const material: IMaterial = {
    //             name: babylonMaterial.name,
    //         };

    //         if (!babylonMesh.color.equals(Color3.White()) || babylonMesh.alpha < 1) {
    //             material.pbrMetallicRoughness = {
    //                 baseColorFactor: [...babylonMesh.color.asArray(), babylonMesh.alpha],
    //             };
    //         }

    //         this._materials.push(material);
    //         materialIndex = this._materials.length - 1;
    //     } else if (babylonMaterial instanceof MultiMaterial) {
    //         const subMaterial = babylonMaterial.subMaterials[submesh.materialIndex];
    //         if (subMaterial) {
    //             babylonMaterial = subMaterial;
    //             materialIndex = this._materialMap[babylonMaterial.uniqueId];
    //         }
    //     } else {
    //         materialIndex = this._materialMap[babylonMaterial.uniqueId];
    //     }
    // }

    private async _exportMeshAsync(babylonMesh: Mesh): Promise<number> {
        const state = this._getState(babylonMesh);
        const convertToRightHanded = this._shouldConvertToRightHanded(babylonMesh);

        const mesh: IMesh = { primitives: [] };

        const indices = babylonMesh.isUnIndexed ? null : babylonMesh.getIndices();
        const vertexBuffers = babylonMesh.geometry?.getVertexBuffers();

        const subMeshes = babylonMesh.subMeshes;
        if (vertexBuffers && subMeshes && subMeshes.length > 0) {
            for (let subMeshIndex = 0; subMeshIndex < subMeshes.length; ++subMeshIndex) {
                const subMesh = subMeshes[subMeshIndex];

                const primitive: IMeshPrimitive = { attributes: {} };
                const babylonMaterial = subMesh.getMaterial(true);

                if (babylonMaterial) {
                    let materialIndex = this._materialMap.get(babylonMaterial) ?? -1;
                    if (materialIndex === -1) {
                        // TODO: Handle LinesMesh

                        const hasUVs = vertexBuffers && Object.keys(vertexBuffers).some((kind) => kind.startsWith("uv"));
                        if (babylonMaterial instanceof MultiMaterial) {
                            const subMaterial = babylonMaterial.subMaterials[subMesh.materialIndex];
                            if (subMaterial) {
                                materialIndex = await this._exportMaterialAsync(subMaterial, hasUVs);
                            }
                        } else {
                            materialIndex = await this._exportMaterialAsync(babylonMaterial, hasUVs);
                        }
                    }

                    if (materialIndex !== -1) {
                        this._materialMap.set(babylonMaterial, materialIndex);
                        primitive.material = materialIndex;
                    }
                }

                const mode = getPrimitiveMode(babylonMesh.overrideRenderingFillMode ?? babylonMaterial?.fillMode ?? Material.TriangleFillMode);
                if (mode !== MeshPrimitiveMode.TRIANGLES) {
                    primitive.mode = mode;
                }

                if (indices) {
                    const is32Bits = isIndicesArray32Bits(indices);

                    const bufferViewIndex = state.getIndicesBufferView(indices) ?? this._exportIndicesData(indices, is32Bits, convertToRightHanded);
                    state.setIndicesBufferView(indices, bufferViewIndex);

                    const accessorIndex =
                        state.getIndicesAccessor(indices, subMesh.indexStart, subMesh.indexCount) ??
                        this._exportIndices(bufferViewIndex, is32Bits, subMesh.indexStart, subMesh.indexCount);
                    state.setIndicesAccessor(indices, subMesh.indexStart, subMesh.indexCount, accessorIndex);

                    primitive.indices = accessorIndex;
                }

                for (const kind of Object.keys(vertexBuffers)) {
                    if (kind.startsWith("uv") && !this._options.exportUnusedUVs) {
                        if (!babylonMaterial || !this._materialNeedsUVsSet.has(babylonMaterial)) {
                            continue;
                        }
                    }

                    const vertexBuffer = vertexBuffers[kind];
                    const buffer = vertexBuffer._buffer;

                    const bufferViewIndex = state.getAttributeBufferView(buffer) ?? this._exportBuffer(buffer, vertexBuffer.byteStride, convertToRightHanded);
                    state.setAttributeBufferView(buffer, bufferViewIndex);

                    const accessorIndex =
                        state.getAttributeAccessor(vertexBuffer, subMesh.verticesStart, subMesh.verticesCount) ??
                        this._exportVertexBuffer(vertexBuffer, subMesh.verticesStart, subMesh.verticesCount, bufferViewIndex);
                    state.setAttributeAccessor(vertexBuffer, subMesh.verticesStart, subMesh.verticesCount, accessorIndex);

                    primitive.attributes[getAttributeType(kind)] = accessorIndex;
                }

                mesh.primitives.push(primitive);
            }
        }

        // TODO: handle morph targets
        // TODO: handle skeleton

        this._meshes.push(mesh);
        return this._meshes.length - 1;
    }

    private async _exportNodeAsync(babylonNode: Node): Promise<number> {
        const state = this._getState(babylonNode);

        const node: INode = {};

        if (babylonNode.name) {
            node.name = babylonNode.name;
        }

        if (babylonNode instanceof TransformNode) {
            this._setNodeTransformation(node, babylonNode);

            if (babylonNode instanceof Mesh || babylonNode instanceof InstancedMesh) {
                const babylonMesh = babylonNode instanceof Mesh ? babylonNode : babylonNode.sourceMesh;
                const meshIndex = state.getMesh(babylonMesh) || await this._exportMeshAsync(babylonMesh);
                state.setMesh(babylonMesh, meshIndex);

                node.mesh = meshIndex;
            } else if (babylonNode instanceof Camera) {
                // TODO: handle camera
            } else {
                // TODO: handle other Babylon node types
            }
        }

        this._nodes.push(node);
        return this._nodes.length - 1;
    }

    private async _exportNodesAsync(babylonNodes: Node[]): Promise<void> {
        // const nodeMap: { [key: number]: number } = {};

        // const runtimeGLTFAnimation: IAnimation = {
        //     name: "runtime animations",
        //     channels: [],
        //     samplers: [],
        // };
        // const idleGLTFAnimations: IAnimation[] = [];

        for (const babylonNode of babylonNodes) {
            const nodeIndex = this._nodeMap.get(babylonNode) || await this._exportNodeAsync(babylonNode);
            this._nodeMap.set(babylonNode, nodeIndex);
        }

        // const promise = this._extensionsPostExportNodeAsync("createNodeAsync", node, babylonNode, nodeMap);
        //         if (promise == null) {
        //             Tools.Warn(`Not exporting node ${babylonNode.name}`);
        //             return Promise.resolve();
        //         } else {
        //             return promise.then((node) => {
        //                 if (!node) {
        //                     return;
        //                 }

        // if (!this._babylonScene.animationGroups.length) {
        //     _GLTFAnimation._CreateMorphTargetAnimationFromMorphTargetAnimations(
        //         babylonNode,
        //         runtimeGLTFAnimation,
        //         idleGLTFAnimations,
        //         nodeMap,
        //         this._nodes,
        //         dataWriter,
        //         this._bufferViews,
        //         this._accessors,
        //         this._animationSampleRate,
        //         this._options.shouldExportAnimation
        //     );
        //     if (babylonNode.animations.length) {
        //         _GLTFAnimation._CreateNodeAnimationFromNodeAnimations(
        //             babylonNode,
        //             runtimeGLTFAnimation,
        //             idleGLTFAnimations,
        //             nodeMap,
        //             this._nodes,
        //             dataWriter,
        //             this._bufferViews,
        //             this._accessors,
        //             this._animationSampleRate,
        //             this._options.shouldExportAnimation
        //         );
        //     }
        // }
        //                 });
        //             }
        //         });
        //     });
        // }

        // return promise.then(() => {
        //     if (runtimeGLTFAnimation.channels.length && runtimeGLTFAnimation.samplers.length) {
        //         this._animations.push(runtimeGLTFAnimation);
        //     }
        //     idleGLTFAnimations.forEach((idleGLTFAnimation) => {
        //         if (idleGLTFAnimation.channels.length && idleGLTFAnimation.samplers.length) {
        //             this._animations.push(idleGLTFAnimation);
        //         }
        //     });

        //     if (this._babylonScene.animationGroups.length) {
        //         _GLTFAnimation._CreateNodeAndMorphAnimationFromAnimationGroups(
        //             this._babylonScene,
        //             this._animations,
        //             nodeMap,
        //             dataWriter,
        //             this._bufferViews,
        //             this._accessors,
        //             this._animationSampleRate,
        //             this._options.shouldExportAnimation
        //         );
        //     }

        //     return nodeMap;
        // });

        //     return nodeMap;
    }

    // private _createNode(babylonNode: Node): INode {
    //     const node: INode = {};

    //     if (babylonNode.name) {
    //         node.name = babylonNode.name;
    //     }

    //     if (babylonNode instanceof TransformNode) {
    //         this._setNodeTransformation(node, babylonNode);

    //         if (babylonNode instanceof AbstractMesh) {
    //             // TODO: handle instancing

    //             const mesh = this._exportMesh(babylonNode);
    //             this._meshes.push(mesh);
    //             node.mesh = this._meshes.length - 1;
    //         }

    //         // if (babylonNode instanceof Mesh) {
    //         //     const morphTargetManager = babylonNode.morphTargetManager;
    //         //     if (morphTargetManager && morphTargetManager.numTargets > 0) {
    //         //         mesh.weights = [];
    //         //         for (let i = 0; i < morphTargetManager.numTargets; ++i) {
    //         //             mesh.weights.push(morphTargetManager.getTarget(i).influence);
    //         //         }
    //         //     }
    //         // }

    //         // return this._setPrimitiveAttributesAsync(mesh, babylonNode, convertToRightHanded, dataWriter).then(() => {
    //         //     if (mesh.primitives.length) {
    //         //         this._meshes.push(mesh);
    //         //         node.mesh = this._meshes.length - 1;
    //         //     }
    //         //     return node;
    //         // });
    //     }

    //     // if (babylonNode instanceof Camera) {
    //     //     this._setCameraTransformation(node, babylonNode, convertToRightHanded);
    //     // }

    //     return node;
    // }

    // /**
    //  * Creates a glTF skin from a Babylon skeleton
    //  * @param babylonScene Babylon Scene
    //  * @param nodeMap Babylon transform nodes
    //  * @param dataWriter Buffer to write binary data to
    //  * @returns Node mapping of unique id to index
    //  */
    // private _createSkinsAsync(nodeMap: { [key: number]: number }, dataWriter: DataWriter): Promise<{ [key: number]: number }> {
    //     const promiseChain = Promise.resolve();
    //     const skinMap: { [key: number]: number } = {};
    //     for (const skeleton of this._babylonScene.skeletons) {
    //         if (skeleton.bones.length <= 0) {
    //             continue;
    //         }
    //         // create skin
    //         const skin: ISkin = { joints: [] };
    //         const inverseBindMatrices: Matrix[] = [];

    //         const boneIndexMap: { [index: number]: Bone } = {};
    //         let maxBoneIndex = -1;
    //         for (let i = 0; i < skeleton.bones.length; ++i) {
    //             const bone = skeleton.bones[i];
    //             const boneIndex = bone.getIndex() ?? i;
    //             if (boneIndex !== -1) {
    //                 boneIndexMap[boneIndex] = bone;
    //                 if (boneIndex > maxBoneIndex) {
    //                     maxBoneIndex = boneIndex;
    //                 }
    //             }
    //         }

    //         for (let boneIndex = 0; boneIndex <= maxBoneIndex; ++boneIndex) {
    //             const bone = boneIndexMap[boneIndex];
    //             inverseBindMatrices.push(bone.getAbsoluteInverseBindMatrix());

    //             const transformNode = bone.getTransformNode();
    //             if (transformNode && nodeMap[transformNode.uniqueId] !== null && nodeMap[transformNode.uniqueId] !== undefined) {
    //                 skin.joints.push(nodeMap[transformNode.uniqueId]);
    //             } else {
    //                 Tools.Warn("Exporting a bone without a linked transform node is currently unsupported");
    //             }
    //         }

    //         if (skin.joints.length > 0) {
    //             // create buffer view for inverse bind matrices
    //             const byteStride = 64; // 4 x 4 matrix of 32 bit float
    //             const byteLength = inverseBindMatrices.length * byteStride;
    //             const bufferViewOffset = dataWriter.getByteOffset();
    //             const bufferView = createBufferView(0, bufferViewOffset, byteLength, undefined, "InverseBindMatrices" + " - " + skeleton.name);
    //             this._bufferViews.push(bufferView);
    //             const bufferViewIndex = this._bufferViews.length - 1;
    //             const bindMatrixAccessor = createAccessor(
    //                 bufferViewIndex,
    //                 "InverseBindMatrices" + " - " + skeleton.name,
    //                 AccessorType.MAT4,
    //                 AccessorComponentType.FLOAT,
    //                 inverseBindMatrices.length,
    //                 null,
    //                 null,
    //                 null
    //             );
    //             const inverseBindAccessorIndex = this._accessors.push(bindMatrixAccessor) - 1;
    //             skin.inverseBindMatrices = inverseBindAccessorIndex;
    //             this._skins.push(skin);
    //             skinMap[skeleton.uniqueId] = this._skins.length - 1;

    //             inverseBindMatrices.forEach((mat) => {
    //                 mat.m.forEach((cell: number) => {
    //                     dataWriter.setFloat32(cell);
    //                 });
    //             });
    //         }
    //     }
    //     return promiseChain.then(() => {
    //         return skinMap;
    //     });
    // }
}
