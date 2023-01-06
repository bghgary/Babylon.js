import type { IGLTFLoaderExtension } from "../glTFLoaderExtension";
import { GLTFLoader, ArrayItem } from "../glTFLoader";
import type { ITexture } from "../glTFLoaderInterfaces";
import type { BaseTexture } from "core/Materials/Textures/baseTexture";
import type { Nullable } from "core/types";
import type { IKHRTextureBasisU } from "babylonjs-gltf2interface";
import type { IKTX2DecoderOptions } from "../../../../../../tools/ktx2Decoder/src/ktx2Decoder";

const NAME = "KHR_texture_basisu";

/**
 * [Specification](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_texture_basisu/README.md)
 */
// eslint-disable-next-line @typescript-eslint/naming-convention
export class KHR_texture_basisu implements IGLTFLoaderExtension {
    /** The name of this extension. */
    public readonly name = NAME;

    /** Defines whether this extension is enabled. */
    public enabled: boolean;

    private _loader: GLTFLoader;

    public decoderOptions: IKTX2DecoderOptions = {};

    /**
     * @internal
     */
    constructor(loader: GLTFLoader) {
        this._loader = loader;
        this.enabled = loader.isExtensionUsed(NAME);
    }

    /** @internal */
    public dispose() {
        (this._loader as any) = null;
    }

    /**
     * @internal
     */
    public _loadTextureAsync(context: string, texture: ITexture, assign: (babylonTexture: BaseTexture) => void): Nullable<Promise<BaseTexture>> {
        return GLTFLoader.LoadExtensionAsync<IKHRTextureBasisU, BaseTexture>(context, texture, this.name, (extensionContext, extension) => {
            const sampler = texture.sampler == undefined ? GLTFLoader.DefaultSampler : ArrayItem.Get(`${context}/sampler`, this._loader.gltf.samplers, texture.sampler);
            const image = ArrayItem.Get(`${extensionContext}/source`, this._loader.gltf.images, extension.source);
            const textureLoadOptions = texture._textureInfo.nonColorData ? { ...this.decoderOptions, useRGBAIfASTCBC7NotAvailableWhenUASTC: true } : this.decoderOptions;
            const useSRGBBuffer = !texture._textureInfo.nonColorData;
            return this._loader._createTextureAsync(context, sampler, image, assign, textureLoadOptions, useSRGBBuffer);
        });
    }
}

GLTFLoader.RegisterExtension(NAME, (loader) => new KHR_texture_basisu(loader));
