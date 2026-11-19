import { type Node } from "core/node";
import { type Scene } from "core/scene";
import { type Animation } from "core/Animations/animation";
import { type AnimationGroup } from "core/Animations/animationGroup";
import { type Camera } from "core/Cameras/camera";
import { type Material } from "core/Materials/material";
import { type IGLTF, type IKHRInteractivity, type IShape } from "babylonjs-gltf2interface";
import { type Quaternion, type Vector3 } from "core/Maths/math.vector";
import { type GLTFData } from "./glTFData";
import { GLTFExporter } from "./glTFExporter";

/**
 * Indexed glTF root collections that KHR_interactivity references can target.
 */
export type KhrInteractivityRootCollection = "nodes" | "animations" | "cameras" | "materials" | "meshes" | "textures" | "images" | "samplers" | "skins" | "scenes";

/**
 * Mesh compression methods.
 */
export type MeshCompressionMethod = "None" | "Draco";

/**
 * Final entity remapping context exposed to a KHR_interactivity export provider.
 */
export interface IKHRInteractivityExportContext {
    /**
     * Gets the final number of glTF nodes.
     * @returns final glTF node count
     */
    getNodeCount(): number;
    /**
     * Gets the final glTF node index for a Babylon node.
     * @param node Babylon node to resolve
     * @returns final glTF node index, or undefined when the node was not exported
     */
    getNodeIndex(node: Node): number | undefined;
    /**
     * Gets the final glTF animation index for a Babylon animation group.
     * @param animation Babylon animation group to resolve
     * @returns final glTF animation index, or undefined when the animation was not exported
     */
    getAnimationIndex(animation: AnimationGroup): number | undefined;
    /**
     * Gets the final glTF camera index for a Babylon camera.
     * @param camera Babylon camera to resolve
     * @returns final glTF camera index, or undefined when the camera was not exported
     */
    getCameraIndex(camera: Camera): number | undefined;
    /**
     * Gets the final glTF material index for a Babylon material.
     * @param material Babylon material to resolve
     * @returns final glTF material index, or undefined when the material was not exported
     */
    getMaterialIndex(material: Material): number | undefined;
    /**
     * Gets the final glTF index for an imported Babylon entity in a root collection.
     * @param collection target glTF root collection
     * @param entity imported Babylon entity associated with the source entry
     * @returns final glTF index, or undefined when the entity was not exported uniquely
     */
    getRootIndex?(collection: KhrInteractivityRootCollection, entity: object): number | undefined;
    /**
     * Writes a companion extension on an already-exported glTF node.
     * @param nodeIndex final glTF node index
     * @param extensionName companion extension name
     * @param value companion extension payload
     */
    setNodeExtension(nodeIndex: number, extensionName: string, value: unknown): void;
}

/**
 * Supplies a detached canonical KHR_interactivity document to the glTF serializer.
 */
export interface IKHRInteractivityExportProvider {
    /** Whether KHR_interactivity must be listed in extensionsRequired. */
    readonly required: boolean;
    /** Additional operation or companion extensions referenced by the graph. */
    readonly additionalExtensionsUsed: readonly string[];
    /** Additional extensions that must be listed in extensionsRequired. */
    readonly additionalExtensionsRequired: readonly string[];
    /**
     * Builds the extension after final glTF entity indices are available.
     * @param context final serializer remapping context
     * @returns canonical KHR_interactivity extension payload
     */
    build(context: IKHRInteractivityExportContext): IKHRInteractivity;
}

/** A bounding volume in a Babylon node's local coordinate system. */
export interface IGLTFBoundingVolumeExport {
    /** The box or sphere defining the volume. Shape extensions are not supported. */
    shape: IShape;
    /** Translation relative to the node. */
    translation?: Vector3;
    /** Rotation relative to the node. */
    rotation?: Quaternion;
    /** Scale relative to the node. */
    scale?: Vector3;
}

/** Final indices and document available when assigning glTF unique IDs. */
export interface IGLTFUniqueIdExportContext {
    /** The exported document before unique IDs are added. */
    readonly gltf: Readonly<IGLTF>;
    /**
     * Gets the exported index of a Babylon node.
     * @param node source Babylon node
     * @returns final node index, or undefined if the node was not exported
     */
    getNodeIndex(node: Node): number | undefined;
}

/**
 * Holds a collection of exporter options and parameters
 */
export interface IExportOptions {
    /**
     * The glTF version to export. Defaults to "2.0".
     */
    gltfVersion?: "2.0" | "2.1";

    /**
     * The binary glTF container version. Defaults to 2.
     */
    glbVersion?: 2 | 3;

    /**
     * An encoded PNG, JPEG, or WebP thumbnail image. Requires glTF 2.1.
     */
    thumbnail?: Blob;

    /**
     * Supplies authored glTF visibility for exported nodes. False hides a node's visual
     * features and those of all descendants, but not cameras. Undefined uses the default.
     * Babylon's `isVisible` does not have the same hierarchy semantics; for loaded glTF
     * nodes, `GetGLTFNodeVisibility` from the loaders package returns the current local value.
     * Requires glTF 2.1. Imported skinned meshes cannot be reparented before export.
     * @param node source Babylon node
     * @returns authored visibility, or undefined for the glTF default
     */
    nodeVisibility?(node: Node): boolean | undefined;

    /**
     * Supplies an authored bounding volume for a node. Requires glTF 2.1.
     * `GetGLTFBoundingVolume` from the loaders package can provide loaded volumes for transform nodes.
     * Imported skinned meshes cannot be reparented before export.
     * @param node source Babylon node
     * @returns the bounding volume, or undefined if the node has none
     */
    nodeBoundingVolume?(node: Node): IGLTFBoundingVolumeExport | undefined;

    /**
     * Supplies unique IDs and their JSON pointers after export indices are assigned.
     * Requires glTF 2.1.
     * @param context final exported document and node indices
     * @returns a map from IDs to JSON pointers
     */
    uniqueIds?(context: IGLTFUniqueIdExportContext): Readonly<Record<string, string>>;

    /**
     * Function which indicates whether a babylon node should be exported or not
     * @param node source Babylon node. It is used to check whether it should be exported to glTF or not
     * @returns boolean, which indicates whether the node should be exported (true) or not (false)
     */
    shouldExportNode?(node: Node): boolean;

    /**
     * Function which indicates whether an animation on the scene should be exported or not
     * @param animation source animation
     * @returns boolean, which indicates whether the animation should be exported (true) or not (false)
     */
    shouldExportAnimation?(animation: Animation): boolean;

    /**
     * Function to extract the part of the scene or node's `metadata` that will populate the corresponding
     * glTF object's `extras` field. If not defined, `node.metadata.gltf.extras` will be used.
     * @param metadata source metadata to read from
     * @returns the data to store into the glTF extras field
     */
    metadataSelector?(metadata: any): any;

    /**
     * The sample rate to bake animation curves. Defaults to 1 / 60.
     */
    animationSampleRate?: number;

    /**
     * Begin serialization without waiting for the scene to be ready. Defaults to false.
     */
    exportWithoutWaitingForScene?: boolean;

    /**
     * Indicates if unused vertex uv attributes should be included in export. Defaults to false.
     */
    exportUnusedUVs?: boolean;

    /**
     * Remove no-op root nodes when possible. Defaults to true. No-op roots are preserved when
     * {@link khrInteractivity}, {@link nodeVisibility}, {@link nodeBoundingVolume}, or {@link uniqueIds} is provided.
     */
    removeNoopRootNodes?: boolean;

    /**
     * Indicates if coordinate system swapping root nodes should be included in export. Defaults to false.
     * @deprecated Please use removeNoopRootNodes instead
     */
    includeCoordinateSystemConversionNodes?: boolean;

    /**
     * Indicates what compression method to apply to mesh data.
     */
    meshCompressionMethod?: MeshCompressionMethod;

    /**
     * Canonical KHR_interactivity export provider. The provider is evaluated only after scene
     * nodes and animations have their final glTF indices.
     */
    khrInteractivity?: IKHRInteractivityExportProvider;
}

/**
 * Class for generating glTF data from a Babylon scene.
 */
export class GLTF2Export {
    /**
     * Exports the scene to .gltf file format
     * @param scene Babylon scene
     * @param fileName Name to use for the .gltf file
     * @param options Exporter options
     * @returns Returns the exported data
     */
    public static async GLTFAsync(scene: Scene, fileName: string, options?: IExportOptions): Promise<GLTFData> {
        if (!options || !options.exportWithoutWaitingForScene) {
            await scene.whenReadyAsync();
        }

        const exporter = new GLTFExporter(scene, options);
        const data = await exporter.generateGLTFAsync(fileName.replace(/\.[^/.]+$/, ""));
        exporter.dispose();

        return data;
    }

    /**
     * Exports the scene to .glb file format
     * @param scene Babylon scene
     * @param fileName Name to use for the .glb file
     * @param options Exporter options
     * @returns Returns the exported data
     */
    public static async GLBAsync(scene: Scene, fileName: string, options?: IExportOptions): Promise<GLTFData> {
        if (!options || !options.exportWithoutWaitingForScene) {
            await scene.whenReadyAsync();
        }

        const exporter = new GLTFExporter(scene, options);
        const data = await exporter.generateGLBAsync(fileName.replace(/\.[^/.]+$/, ""));
        exporter.dispose();

        return data;
    }
}
