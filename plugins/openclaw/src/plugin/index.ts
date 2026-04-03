/**
 * Headroom OpenClaw Plugin — register ContextEngine + CCR retrieval tool.
 *
 * Usage:
 *   openclaw plugins install headroom-ai/openclaw
 *
 * Configuration (in ~/.openclaw/config.json or ~/.clawdbot/clawdbot.json):
 *   {
 *     "plugins": {
 *       "slots": { "contextEngine": "headroom" },
 *       "entries": { "headroom": { "enabled": true } }
 *     }
 *   }
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

import { HeadroomContextEngine } from "../engine.js";
import { normalizeAndValidateProxyUrl } from "../proxy-manager.js";
import { createHeadroomRetrieveTool } from "../tools/headroom-retrieve.js";

export default function headroomPlugin(api: any) {
  const config = api.config?.plugins?.entries?.headroom?.config ?? {};
  const logger = api.logger ?? console;
  const rawProxyUrl = config.proxyUrl;
  if (!rawProxyUrl || typeof rawProxyUrl !== "string") {
    throw new Error(
      '[headroom] Missing required config: plugins.entries.headroom.config.proxyUrl (example: "http://127.0.0.1:8787")',
    );
  }
  const proxyUrl = normalizeAndValidateProxyUrl(rawProxyUrl);

  const engine = new HeadroomContextEngine({ ...config, proxyUrl }, {
    info: (m: string) => logger.info(m),
    warn: (m: string) => logger.warn(m),
    error: (m: string) => logger.error(m),
    debug: (m: string) => logger.debug?.(m),
  });

  // Register as context engine
  api.registerContextEngine("headroom", () => engine);

  // Register CCR retrieval tool (active once proxy is running)
  api.registerTool((ctx: any) => {
    const activeProxyUrl = engine.getProxyUrl() ?? proxyUrl;
    return createHeadroomRetrieveTool({ proxyUrl: activeProxyUrl });
  });

  logger.info("[headroom] Plugin registered");
}
