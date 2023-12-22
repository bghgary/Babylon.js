import type { IBufferView, AccessorComponentType, IAccessor } from "babylonjs-gltf2interface";
import { AccessorType, MeshPrimitiveMode } from "babylonjs-gltf2interface";

import type { FloatArray, Nullable } from "core/types";
import type { Quaternion, Vector4 } from "core/Maths/math.vector";
import { Vector3 } from "core/Maths/math.vector";
import { VertexBuffer } from "core/Buffers/buffer";
import { Material } from "core/Materials/material";

/**
 * Creates a buffer view based on the supplied arguments
 * @param bufferIndex index value of the specified buffer
 * @param byteOffset byte offset value
 * @param byteLength byte length of the bufferView
 * @param byteStride byte distance between conequential elements
 * @param name name of the buffer view
 * @returns bufferView for glTF
 */
export function createBufferView(bufferIndex: number, byteOffset: number, byteLength: number, byteStride?: number, name?: string): IBufferView {
    const bufferview: IBufferView = { buffer: bufferIndex, byteLength: byteLength };

    if (byteOffset) {
        bufferview.byteOffset = byteOffset;
    }

    if (name) {
        bufferview.name = name;
    }

    if (byteStride) {
        bufferview.byteStride = byteStride;
    }

    return bufferview;
}

/**
 * Creates an accessor based on the supplied arguments
 * @param bufferViewIndex The index of the bufferview referenced by this accessor
 * @param type The type of the accessor
 * @param componentType The datatype of components in the attribute
 * @param count The number of attributes referenced by this accessor
 * @param byteOffset The offset relative to the start of the bufferView in bytes
 * @param min Minimum value of each component in this attribute
 * @param max Maximum value of each component in this attribute
 * @returns accessor for glTF
 */
export function createAccessor(
    bufferViewIndex: number,
    type: AccessorType,
    componentType: AccessorComponentType,
    count: number,
    byteOffset: Nullable<number>,
    min: Nullable<number[]>,
    max: Nullable<number[]>
): IAccessor {
    const accessor: IAccessor = { bufferView: bufferViewIndex, componentType: componentType, count: count, type: type };

    if (min != null) {
        accessor.min = min;
    }

    if (max != null) {
        accessor.max = max;
    }

    if (byteOffset != null) {
        accessor.byteOffset = byteOffset;
    }

    return accessor;
}

/**
 * Calculates the minimum and maximum values of an array of position floats
 * @param positions Positions array of a mesh
 * @param vertexStart Starting vertex offset to calculate min and max values
 * @param vertexCount Number of vertices to check for min and max values
 * @returns min number array and max number array
 */
export function calculateMinMaxPositions(positions: FloatArray, vertexStart: number, vertexCount: number): [number[], number[]] {
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    const positionStrideSize = 3;
    let indexOffset: number;
    let position: Vector3;
    let vector: number[];

    if (vertexCount) {
        for (let i = vertexStart, length = vertexStart + vertexCount; i < length; ++i) {
            indexOffset = positionStrideSize * i;

            position = Vector3.FromArray(positions, indexOffset);
            vector = position.asArray();

            for (let j = 0; j < positionStrideSize; ++j) {
                const num = vector[j];
                if (num < min[j]) {
                    min[j] = num;
                }
                if (num > max[j]) {
                    max[j] = num;
                }
                ++indexOffset;
            }
        }
    }

    return [min, max];
}

export function getAccessorElementCount(accessorType: AccessorType): number {
    switch (accessorType) {
        case AccessorType.MAT2:
            return 4;
        case AccessorType.MAT3:
            return 9;
        case AccessorType.MAT4:
            return 16;
        case AccessorType.SCALAR:
            return 1;
        case AccessorType.VEC2:
            return 2;
        case AccessorType.VEC3:
            return 3;
        case AccessorType.VEC4:
            return 4;
    }
}

export function getAccessorType(kind: string): AccessorType {
    switch (kind) {
        case VertexBuffer.PositionKind:
        case VertexBuffer.NormalKind:
            return AccessorType.VEC3;
        case VertexBuffer.ColorKind:
        case VertexBuffer.TangentKind:
        case VertexBuffer.MatricesIndicesKind:
        case VertexBuffer.MatricesIndicesExtraKind:
        case VertexBuffer.MatricesWeightsKind:
        case VertexBuffer.MatricesWeightsExtraKind:
            return AccessorType.VEC4;
        case VertexBuffer.UVKind:
        case VertexBuffer.UV2Kind:
        case VertexBuffer.UV3Kind:
        case VertexBuffer.UV4Kind:
        case VertexBuffer.UV5Kind:
        case VertexBuffer.UV6Kind:
            return AccessorType.VEC2;
    }

    throw new Error(`Unknown kind ${kind}`);
}

export function getAttributeType(kind: string): string {
    switch (kind) {
        case VertexBuffer.PositionKind:
            return "POSITION";
        case VertexBuffer.NormalKind:
            return "NORMAL";
        case VertexBuffer.TangentKind:
            return "TANGENT";
        case VertexBuffer.ColorKind:
            return "COLOR_0";
        case VertexBuffer.UVKind:
            return "TEXCOORD_0";
        case VertexBuffer.UV2Kind:
            return "TEXCOORD_1";
        case VertexBuffer.UV3Kind:
            return "TEXCOORD_2";
        case VertexBuffer.UV4Kind:
            return "TEXCOORD_3";
        case VertexBuffer.UV5Kind:
            return "TEXCOORD_4";
        case VertexBuffer.UV6Kind:
            return "TEXCOORD_5";
        case VertexBuffer.MatricesIndicesKind:
            return "JOINTS_0";
        case VertexBuffer.MatricesIndicesExtraKind:
            return "JOINTS_1";
        case VertexBuffer.MatricesWeightsKind:
            return "WEIGHTS_0";
        case VertexBuffer.MatricesWeightsExtraKind:
            return "WEIGHTS_1";
    }

    throw new Error(`Unknown kind: ${kind}`);
}

export function getPrimitiveMode(fillMode: number): MeshPrimitiveMode {
    switch (fillMode) {
        case Material.TriangleFillMode:
            return MeshPrimitiveMode.TRIANGLES;
        case Material.TriangleStripDrawMode:
            return MeshPrimitiveMode.TRIANGLE_STRIP;
        case Material.TriangleFanDrawMode:
            return MeshPrimitiveMode.TRIANGLE_FAN;
        case Material.PointListDrawMode:
            return MeshPrimitiveMode.POINTS;
        case Material.PointFillMode:
            return MeshPrimitiveMode.POINTS;
        case Material.LineLoopDrawMode:
            return MeshPrimitiveMode.LINE_LOOP;
        case Material.LineListDrawMode:
            return MeshPrimitiveMode.LINES;
        case Material.LineStripDrawMode:
            return MeshPrimitiveMode.LINE_STRIP;
    }

    throw new Error(`Unknown fill mode: ${fillMode}`);
}

export function normalizeTangent(tangent: Vector4) {
    const length = Math.sqrt(tangent.x * tangent.x + tangent.y * tangent.y + tangent.z * tangent.z);
    if (length > 0) {
        tangent.x /= length;
        tangent.y /= length;
        tangent.z /= length;
    }
}

/**
 * Converts a Vector3 to right-handed
 * @param value value to convert to right-handed
 */
export function convertToRightHandedPosition(value: Vector3): Vector3 {
    value.z *= -1;
    return value;
}

/**
 * Converts a Quaternion to right-handed
 * @param value value to convert to right-handed
 */
export function convertToRightHandedRotation(value: Quaternion): Quaternion {
    value.x *= -1;
    value.y *= -1;
    return value;
}

// /**
//  * Converts a new right-handed Vector3
//  * @param vector vector3 array
//  * @returns right-handed Vector3
//  */
// public static _GetRightHandedNormalVector3(vector: Vector3): Vector3 {
//     return new Vector3(vector.x, vector.y, -vector.z);
// }

// /**
//  * Converts a Vector3 to right-handed
//  * @param vector Vector3 to convert to right-handed
//  */
// public static _GetRightHandedNormalVector3FromRef(vector: Vector3) {
//     vector.z *= -1;
// }

// /**
//  * Converts a three element number array to right-handed
//  * @param vector number array to convert to right-handed
//  */
// public static _GetRightHandedNormalArray3FromRef(vector: number[]) {
//     vector[2] *= -1;
// }

// /**
//  * Converts a Vector4 to right-handed
//  * @param vector Vector4 to convert to right-handed
//  */
// public static _GetRightHandedVector4FromRef(vector: Vector4) {
//     vector.z *= -1;
//     vector.w *= -1;
// }

// /**
//  * Converts a Vector4 to right-handed
//  * @param vector Vector4 to convert to right-handed
//  */
// public static _GetRightHandedArray4FromRef(vector: number[]) {
//     vector[2] *= -1;
//     vector[3] *= -1;
// }

// /**
//  * Converts a Quaternion to right-handed
//  * @param quaternion Source quaternion to convert to right-handed
//  */
// public static _GetRightHandedQuaternionFromRef(quaternion: Quaternion) {
//     quaternion.x *= -1;
//     quaternion.y *= -1;
// }

// /**
//  * Converts a Quaternion to right-handed
//  * @param quaternion Source quaternion to convert to right-handed
//  */
// public static _GetRightHandedQuaternionArrayFromRef(quaternion: number[]) {
//     quaternion[0] *= -1;
//     quaternion[1] *= -1;
// }
