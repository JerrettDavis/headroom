import { describe, it, expect, afterEach, vi } from "vitest";
import {
  ProxyManager,
  normalizeAndValidateProxyUrl,
  isLocalProxyUrl,
  probeHeadroomProxy,
} from "../src/proxy-manager.js";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("normalizeAndValidateProxyUrl", () => {
  it("accepts localhost origins", () => {
    expect(normalizeAndValidateProxyUrl("http://127.0.0.1:8787")).toBe("http://127.0.0.1:8787");
    expect(normalizeAndValidateProxyUrl("http://localhost:8787")).toBe("http://localhost:8787");
  });

  it("accepts remote URLs", () => {
    expect(normalizeAndValidateProxyUrl("http://example.com:8787")).toBe("http://example.com:8787");
    expect(normalizeAndValidateProxyUrl("https://headroom.example.com")).toBe("https://headroom.example.com");
    expect(normalizeAndValidateProxyUrl("https://headroom.example.com:9090")).toBe("https://headroom.example.com:9090");
  });

  it("rejects malformed URLs", () => {
    expect(() => normalizeAndValidateProxyUrl("ftp://localhost:8787")).toThrow(
      /must use http/,
    );
    expect(() => normalizeAndValidateProxyUrl("http://localhost:8787/path")).toThrow(
      /must not include a path/,
    );
  });
});

describe("isLocalProxyUrl", () => {
  it("returns true for localhost addresses", () => {
    expect(isLocalProxyUrl("http://127.0.0.1:8787")).toBe(true);
    expect(isLocalProxyUrl("http://localhost:8787")).toBe(true);
  });

  it("returns false for remote addresses", () => {
    expect(isLocalProxyUrl("http://example.com:8787")).toBe(false);
    expect(isLocalProxyUrl("https://headroom.example.com")).toBe(false);
  });

  it("returns false for invalid URLs", () => {
    expect(isLocalProxyUrl("not-a-url")).toBe(false);
  });
});

describe("probeHeadroomProxy", () => {
  it("returns reachable+isHeadroom when both endpoints succeed", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: true, status: 200 })
      .mockResolvedValueOnce({ ok: true, status: 200 });
    vi.stubGlobal("fetch", fetchMock);

    const result = await probeHeadroomProxy("http://127.0.0.1:8787");
    expect(result).toEqual({ reachable: true, isHeadroom: true });
  });

  it("returns reachable but non-headroom when retrieve endpoint fails", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: true, status: 200 })
      .mockResolvedValueOnce({ ok: false, status: 404 });
    vi.stubGlobal("fetch", fetchMock);

    const result = await probeHeadroomProxy("http://127.0.0.1:8787");
    expect(result.reachable).toBe(true);
    expect(result.isHeadroom).toBe(false);
    expect(result.reason).toMatch(/retrieve stats HTTP 404/);
  });

  it("returns unreachable when health check fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("boom")));
    const result = await probeHeadroomProxy("http://127.0.0.1:8787");
    expect(result.reachable).toBe(false);
    expect(result.isHeadroom).toBe(false);
  });
});

describe("ProxyManager.start", () => {
  it("auto-detects running proxy on default candidates", async () => {
    const manager = new ProxyManager({});

    // Candidate 1: health fail. Candidate 2: health+retrieve succeed.
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new Error("down"))
      .mockResolvedValueOnce({ ok: true, status: 200 })
      .mockResolvedValueOnce({ ok: true, status: 200 });
    vi.stubGlobal("fetch", fetchMock);

    const startSpy = vi.spyOn(manager as any, "startHeadroomProxy");
    const url = await manager.start();
    expect(url).toBe("http://localhost:8787");
    expect(startSpy).not.toHaveBeenCalled();
  });

  it("uses proxyPort for auto-detect candidates", async () => {
    const manager = new ProxyManager({ proxyPort: 9797, autoStart: false });

    const fetchMock = vi.fn().mockRejectedValue(new Error("down"));
    vi.stubGlobal("fetch", fetchMock);

    await expect(manager.start()).rejects.toThrow(/127\.0\.0\.1:9797.*localhost:9797/);
  });

  it("rejects invalid proxyPort", async () => {
    const manager = new ProxyManager({ proxyPort: 0 });
    await expect(manager.start()).rejects.toThrow(/proxyPort must be an integer between 1 and 65535/);
  });

  it("fails when explicit URL is reachable but not a headroom proxy", async () => {
    const manager = new ProxyManager({ proxyUrl: "http://127.0.0.1:8787" });
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: true, status: 200 })
      .mockResolvedValueOnce({ ok: false, status: 404 });
    vi.stubGlobal("fetch", fetchMock);

    await expect(manager.start()).rejects.toThrow(/does not appear to be a Headroom proxy/);
  });

  it("applies default proxyPort when explicit proxyUrl omits port", async () => {
    const manager = new ProxyManager({ proxyUrl: "http://127.0.0.1", autoStart: true });
    const startSpy = vi.spyOn(manager as any, "startHeadroomProxy").mockResolvedValue(undefined);

    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new Error("down"))
      .mockResolvedValueOnce({ ok: true, status: 200 })
      .mockResolvedValueOnce({ ok: true, status: 200 });
    vi.stubGlobal("fetch", fetchMock);

    const url = await manager.start();
    expect(url).toBe("http://127.0.0.1:8787");
    expect(startSpy).toHaveBeenCalledWith("http://127.0.0.1:8787", 8787);
  });

  it("connects to remote proxy without auto-start", async () => {
    const manager = new ProxyManager({ proxyUrl: "http://headroom.remote.example:8787", autoStart: true });
    const startSpy = vi.spyOn(manager as any, "startHeadroomProxy").mockResolvedValue(undefined);

    // Remote probe succeeds
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: true, status: 200 })   // /health
      .mockResolvedValueOnce({ ok: true, status: 200 });   // /v1/retrieve/stats
    vi.stubGlobal("fetch", fetchMock);

    const url = await manager.start();
    expect(url).toBe("http://headroom.remote.example:8787");
    expect(startSpy).not.toHaveBeenCalled();
  });

  it("fails fast for unreachable remote proxy without attempting auto-start", async () => {
    const manager = new ProxyManager({ proxyUrl: "https://headroom.remote.example:8787", autoStart: true });
    const startSpy = vi.spyOn(manager as any, "startHeadroomProxy").mockResolvedValue(undefined);

    const fetchMock = vi.fn().mockRejectedValue(new Error("ECONNREFUSED"));
    vi.stubGlobal("fetch", fetchMock);

    await expect(manager.start()).rejects.toThrow(/Remote Headroom proxy not reachable/);
    expect(startSpy).not.toHaveBeenCalled();
  });

  it("auto-starts when nothing is detected", async () => {
    const manager = new ProxyManager({ autoStart: true });
    const startSpy = vi.spyOn(manager as any, "startHeadroomProxy").mockResolvedValue(undefined);

    // First two candidate probes fail (health only), then waitForHealthy probe succeeds.
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new Error("down"))
      .mockRejectedValueOnce(new Error("down"))
      .mockResolvedValueOnce({ ok: true, status: 200 })
      .mockResolvedValueOnce({ ok: true, status: 200 });
    vi.stubGlobal("fetch", fetchMock);

    const url = await manager.start();
    expect(url).toBe("http://127.0.0.1:8787");
    expect(startSpy).toHaveBeenCalledWith("http://127.0.0.1:8787", 8787);
  });
});

describe("ProxyManager launch internals", () => {
  it("prefers configured pythonPath in fallback order", () => {
    const manager = new ProxyManager({ pythonPath: "C:\\Python311\\python.exe" });
    const commands = (manager as any).getPythonCommands() as string[];
    expect(commands[0]).toBe("C:\\Python311\\python.exe");
    expect(commands).toContain("python");
    expect(commands).toContain("python3");
    expect(commands).toContain("py");
  });

  it("uses first available launcher from provided specs", async () => {
    const manager = new ProxyManager({});
    (manager as any).buildLaunchSpecs = () => [
      {
        label: "first",
        command: "first-missing-command",
        args: ["proxy"],
        checkCommand: "first-missing-command",
        checkArgs: ["--version"],
      },
      {
        label: "second-node",
        command: "node",
        args: ["-e", ""],
        checkCommand: "node",
        checkArgs: ["--version"],
      },
    ];
    const infoSpy = vi.spyOn((manager as any).logger, "info");

    await (manager as any).startHeadroomProxy("http://127.0.0.1:8787");
    expect(infoSpy).toHaveBeenCalledWith(expect.stringContaining("Auto-start launcher selected"));
    expect(infoSpy).toHaveBeenCalledWith(expect.stringContaining("second-node"));
  });

  it("throws when no launcher is executable", async () => {
    const manager = new ProxyManager({});
    (manager as any).buildLaunchSpecs = () => [
      {
        label: "none",
        command: "none",
        args: ["proxy"],
        checkCommand: "none",
        checkArgs: ["--version"],
      },
    ];
    (manager as any).canExecute = () => false;

    await expect((manager as any).startHeadroomProxy("http://127.0.0.1:8787")).rejects.toThrow(
      /No usable Headroom launcher found/,
    );
  });
});
