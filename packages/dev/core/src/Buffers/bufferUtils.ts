import { DataArray, FloatArray } from "../types";
import { VertexBuffer } from "./buffer";

function getFloatValue(dataView: DataView, type: number, byteOffset: number, normalized: boolean): number {
    switch (type) {
        case VertexBuffer.BYTE: {
            let value = dataView.getInt8(byteOffset);
            if (normalized) {
                value = Math.max(value / 127, -1);
            }
            return value;
        }
        case VertexBuffer.UNSIGNED_BYTE: {
            let value = dataView.getUint8(byteOffset);
            if (normalized) {
                value = value / 255;
            }
            return value;
        }
        case VertexBuffer.SHORT: {
            let value = dataView.getInt16(byteOffset, true);
            if (normalized) {
                value = Math.max(value / 32767, -1);
            }
            return value;
        }
        case VertexBuffer.UNSIGNED_SHORT: {
            let value = dataView.getUint16(byteOffset, true);
            if (normalized) {
                value = value / 65535;
            }
            return value;
        }
        case VertexBuffer.INT: {
            return dataView.getInt32(byteOffset, true);
        }
        case VertexBuffer.UNSIGNED_INT: {
            return dataView.getUint32(byteOffset, true);
        }
        case VertexBuffer.FLOAT: {
            return dataView.getFloat32(byteOffset, true);
        }
        default: {
            throw new Error(`Invalid component type ${type}`);
        }
    }
}

function setFloatValue(dataView: DataView, type: number, byteOffset: number, normalized: boolean, value: number): void {
    switch (type) {
        case VertexBuffer.BYTE: {
            if (normalized) {
                value = Math.round(value * 127.0);
            }
            dataView.setInt8(byteOffset, value);
            break;
        }
        case VertexBuffer.UNSIGNED_BYTE: {
            if (normalized) {
                value = Math.round(value * 255);
            }
            dataView.setUint8(byteOffset, value);
            break;
        }
        case VertexBuffer.SHORT: {
            if (normalized) {
                value = Math.round(value * 32767);
            }
            dataView.setInt16(byteOffset, value, true);
            break;
        }
        case VertexBuffer.UNSIGNED_SHORT: {
            if (normalized) {
                value = Math.round(value * 65535);
            }
            dataView.setUint16(byteOffset, value, true);
            break;
        }
        case VertexBuffer.INT: {
            dataView.setInt32(byteOffset, value, true);
            break;
        }
        case VertexBuffer.UNSIGNED_INT: {
            dataView.setUint32(byteOffset, value, true);
            break;
        }
        case VertexBuffer.FLOAT: {
            dataView.setFloat32(byteOffset, value, true);
            break;
        }
        default: {
            throw new Error(`Invalid component type ${type}`);
        }
    }
}

/**
 * Enumerates each value of the data array and calls the given callback.
 * @param data the data to enumerate
 * @param byteOffset the byte offset of the data
 * @param byteStride the byte stride of the data
 * @param componentCount the number of components per element
 * @param componentType the type of the component
 * @param count the number of values to enumerate
 * @param normalized whether the data is normalized
 * @param callback the callback function called for each group of component values
 */
export function enumerateFloatValues(
    data: DataArray,
    byteOffset: number,
    byteStride: number,
    componentCount: number,
    componentType: number,
    count: number,
    normalized: boolean,
    callback: (values: number[], index: number) => void
): void {
    const oldValues = new Array<number>(componentCount);
    const newValues = new Array<number>(componentCount);

    if (data instanceof Array) {
        let offset = byteOffset / 4;
        const stride = byteStride / 4;
        for (let index = 0; index < count; index += componentCount) {
            for (let componentIndex = 0; componentIndex < componentCount; componentIndex++) {
                oldValues[componentIndex] = newValues[componentIndex] = data[offset + componentIndex];
            }

            callback(newValues, index);

            for (let componentIndex = 0; componentIndex < componentCount; componentIndex++) {
                if (oldValues[componentIndex] !== newValues[componentIndex]) {
                    data[offset + componentIndex] = newValues[componentIndex];
                }
            }

            offset += stride;
        }
    } else {
        const dataView = data instanceof ArrayBuffer ? new DataView(data) : new DataView(data.buffer, data.byteOffset, data.byteLength);
        const componentByteLength = VertexBuffer.GetTypeByteLength(componentType);
        for (let index = 0; index < count; index += componentCount) {
            for (let componentIndex = 0, componentByteOffset = byteOffset; componentIndex < componentCount; componentIndex++, componentByteOffset += componentByteLength) {
                oldValues[componentIndex] = newValues[componentIndex] = getFloatValue(dataView, componentType, componentByteOffset, normalized);
            }

            callback(newValues, index);

            for (let componentIndex = 0, componentByteOffset = byteOffset; componentIndex < componentCount; componentIndex++, componentByteOffset += componentByteLength) {
                if (oldValues[componentIndex] !== newValues[componentIndex]) {
                    setFloatValue(dataView, componentType, componentByteOffset, normalized, newValues[componentIndex]);
                }
            }

            byteOffset += byteStride;
        }
    }
}

/**
 * Gets the given data array as a float array. Float data is constructed if the data array cannot be returned directly.
 * @param data the input data array
 * @param size the number of components
 * @param type the component type
 * @param byteOffset the byte offset of the data
 * @param byteStride the byte stride of the data
 * @param normalized whether the data is normalized
 * @param totalVertices number of vertices in the buffer to take into account
 * @param forceCopy defines a boolean indicating that the returned array must be cloned upon returning it
 * @returns a float array containing vertex data
 */
export function getFloatData(
    data: DataArray,
    size: number,
    type: number,
    byteOffset: number,
    byteStride: number,
    normalized: boolean,
    totalVertices: number,
    forceCopy?: boolean
): FloatArray {
    const tightlyPackedByteStride = size * VertexBuffer.GetTypeByteLength(type);
    const count = totalVertices * size;

    if (type !== VertexBuffer.FLOAT || byteStride !== tightlyPackedByteStride) {
        const copy = new Float32Array(count);
        enumerateFloatValues(data, byteOffset, byteStride, size, type, count, normalized, (values, index) => {
            for (let i = 0; i < size; i++) {
                copy[index + i] = values[i];
            }
        });
        return copy;
    }

    if (!(data instanceof Array || data instanceof Float32Array) || byteOffset !== 0 || data.length !== count) {
        if (data instanceof Array) {
            const offset = byteOffset / 4;
            return data.slice(offset, offset + count);
        } else if (data instanceof ArrayBuffer) {
            return new Float32Array(data, byteOffset, count);
        } else {
            let offset = data.byteOffset + byteOffset;
            if (forceCopy) {
                const result = new Float32Array(count);
                const source = new Float32Array(data.buffer, offset, count);

                result.set(source);

                return result;
            }

            // Protect against bad data
            const remainder = offset % 4;
            if (remainder) {
                offset = Math.max(0, offset - remainder);
            }

            return new Float32Array(data.buffer, offset, count);
        }
    }

    if (forceCopy) {
        return data.slice();
    }

    return data;
}
