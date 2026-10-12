import { CreateTypeScriptTranspiler } from "@tools/snippet-loader";

let transpilerPromise: Promise<ReturnType<typeof CreateTypeScriptTranspiler>> | undefined;

export async function transpileSnippet(source: string, fileName: string): Promise<string> {
    if (!/\.tsx?$/i.test(fileName)) {
        return source;
    }
    transpilerPromise ??= import("typescript").then(CreateTypeScriptTranspiler);
    return (await transpilerPromise)(source, fileName);
}
