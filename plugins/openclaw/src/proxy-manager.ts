/**
 * Manages connectivity to an externally managed Headroom proxy.
 *
 * Security model:
 * - No process execution
 * - No environment variable access
 * - Localhost-only network access (127.0.0.1 / localhost)
 */

export interface ProxyManagerConfig {
  proxyUrl?: string;
}

export interface ProxyManagerLogger {
  info(message: string): void;
  warn(message: string): void;
  error(message: string): void;
  debug(message: string): void;
}

export interface ProxyProbeResult {
  reachable: boolean;
  isHeadroom: boolean;
  reason?: string;
}

const defaultLogger: ProxyManagerLogger = {
  info: (m) => console.log(`[headroom] ${m}`),
  warn: (m) => console.warn(`[headroom] ${m}`),
  error: (m) => console.error(`[headroom] ${m}`),
  debug: () => {},
};

export class ProxyManager {
  private config: ProxyManagerConfig;
  private logger: ProxyManagerLogger;
  private proxyUrl: string | null = null;

  constructor(config: ProxyManagerConfig = {}, logger?: ProxyManagerLogger) {
    this.config = config;
    this.logger = logger ?? defaultLogger;
  }

  /**
   * Ensure a proxy is available. Returns the normalized URL origin.
   */
  async start(): Promise<string> {
    if (!this.config.proxyUrl) {
      throw new Error(
        "Headroom proxy URL is required. Configure plugins.entries.headroom.config.proxyUrl " +
          '(example: "http://127.0.0.1:8787").',
      );
    }

    const url = normalizeAndValidateProxyUrl(this.config.proxyUrl);
    const probe = await probeHeadroomProxy(url);

    if (probe.reachable && probe.isHeadroom) {
      this.proxyUrl = url;
      this.logger.info(`Headroom proxy already running at ${url}`);
      return url;
    }

    if (probe.reachable && !probe.isHeadroom) {
      throw new Error(
        `Service reachable at ${url}, but it does not appear to be a Headroom proxy (${probe.reason ?? "unknown service"}).`,
      );
    }

    throw new Error(`Headroom proxy not reachable at ${url}. Ensure the proxy is running first.`);
  }

  /**
   * No-op: plugin never starts or manages external processes.
   */
  async stop(): Promise<void> {
    this.proxyUrl = null;
  }

  getUrl(): string | null {
    return this.proxyUrl;
  }

  // --- Internal ---
}

export function normalizeAndValidateProxyUrl(proxyUrl: string): string {
  let parsed: URL;
  try {
    parsed = new URL(proxyUrl);
  } catch {
    throw new Error(`Invalid proxyUrl: "${proxyUrl}"`);
  }

  if (parsed.protocol !== "http:") {
    throw new Error("proxyUrl must use http://");
  }
  if (parsed.hostname !== "127.0.0.1" && parsed.hostname !== "localhost") {
    throw new Error("proxyUrl host must be localhost or 127.0.0.1");
  }

  if (parsed.pathname !== "/" || parsed.search || parsed.hash) {
    throw new Error("proxyUrl must not include a path, query, or hash");
  }

  return parsed.origin;
}

/**
 * Probe a configured URL and verify whether it is a running Headroom proxy.
 */
export async function probeHeadroomProxy(proxyUrl: string): Promise<ProxyProbeResult> {
  const origin = normalizeAndValidateProxyUrl(proxyUrl);

  try {
    const health = await fetch(`${origin}/health`, {
      signal: AbortSignal.timeout(3_000),
    });
    if (!health.ok) {
      return { reachable: false, isHeadroom: false, reason: `health HTTP ${health.status}` };
    }
  } catch {
    return { reachable: false, isHeadroom: false, reason: "health check failed" };
  }

  try {
    const retrieveStats = await fetch(`${origin}/v1/retrieve/stats`, {
      signal: AbortSignal.timeout(3_000),
    });
    if (retrieveStats.ok) {
      return { reachable: true, isHeadroom: true };
    }
    return {
      reachable: true,
      isHeadroom: false,
      reason: `retrieve stats HTTP ${retrieveStats.status}`,
    };
  } catch {
    return {
      reachable: true,
      isHeadroom: false,
      reason: "retrieve stats endpoint unavailable",
    };
  }
}
