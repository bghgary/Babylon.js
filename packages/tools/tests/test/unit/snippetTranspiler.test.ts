import { vi } from "vitest";
import { CreateTypeScriptTranspiler } from "@tools/snippet-loader";
import { transpileSnippet } from "../playwright/snippetTranspiler";

vi.mock("@tools/snippet-loader", async (importOriginal) => {
    const original = await importOriginal<typeof import("@tools/snippet-loader")>();
    return { ...original, CreateTypeScriptTranspiler: vi.fn(original.CreateTypeScriptTranspiler) };
});

describe("visualization snippet transpilation", () => {
    it("passes JavaScript through without loading TypeScript", async () => {
        const source = "export const value = 42;";
        expect(await transpileSnippet(source, "index.js")).toBe(source);
        expect(CreateTypeScriptTranspiler).not.toHaveBeenCalled();
    });

    it("shares the compiler across concurrent TypeScript snippets", async () => {
        const results = await Promise.all([
            transpileSnippet("export const value: number = 42;", "index.ts"),
            transpileSnippet("export const value: string = 'hello';", "other.ts"),
        ]);
        expect(results[0]).toContain("export const value = 42;");
        expect(results[1]).toContain("export const value = 'hello';");
        expect(CreateTypeScriptTranspiler).toHaveBeenCalledOnce();
    });

    it("preserves ES modules and handles TSX", async () => {
        const result = await transpileSnippet("export const element = <div />;", "component.tsx");
        expect(result).toContain("react/jsx-runtime");
        expect(result).toContain("export const element");
        expect(CreateTypeScriptTranspiler).toHaveBeenCalledOnce();
    });
});
