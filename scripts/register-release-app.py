"""Register a GitHub App from a manifest without persisting its private key."""

import argparse
import html
import json
import re
import secrets
import subprocess
import threading
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def registration_url(owner, organization, state):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*", owner):
        raise ValueError("Invalid GitHub owner")
    path = f"organizations/{owner}/settings/apps/new" if organization else "settings/apps/new"
    return f"https://github.com/{path}?state={urllib.parse.quote(state)}"


def callback_code(query, state):
    params = urllib.parse.parse_qs(query)
    if params.get("state") != [state] or len(params.get("code", [])) != 1:
        raise ValueError("Invalid registration callback")
    code = params["code"][0]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", code):
        raise ValueError("Invalid registration code")
    return code


def configure_repository(repo, app):
    # No token, private key, client secret, or credential response goes to logs.
    for command, value in [
        (["gh", "secret", "set", "RELEASE_APP_PRIVATE_KEY", "--repo", repo], app["pem"]),
        (["gh", "variable", "set", "RELEASE_APP_ID", "--repo", repo], str(app["id"])),
    ]:
        result = subprocess.run(command, input=value, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError("Repository configuration failed; recover in GitHub App settings")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default="JerrettDavis")
    parser.add_argument("--repo", default="JerrettDavis/headroom")
    parser.add_argument("--organization", action="store_true")
    parser.add_argument("--name", help="Globally unique App name (34 characters maximum)")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9-]+/[A-Za-z0-9_.-]+", args.repo):
        parser.error("--repo must be OWNER/REPOSITORY")
    profile = subprocess.run(
        ["gh", "api", "user", "--jq", ".login"], capture_output=True, text=True, check=True
    )
    if not args.organization and profile.stdout.strip().lower() != args.owner.lower():
        parser.error("Authenticate gh as the personal App owner first")
    state = secrets.token_urlsafe(32)
    action = registration_url(args.owner, args.organization, state)
    manifest = json.loads(Path(__file__).with_name("release-app-manifest.json").read_text())
    manifest["name"] = args.name or f"Headroom Release {args.owner}"[:34]
    if not manifest["name"].strip() or len(manifest["name"]) > 34:
        parser.error("App name must contain 1 to 34 characters")
    manifest["url"] = f"https://github.com/{args.repo}"
    outcome = {"success": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Callback URLs contain a one-time credential.

        def reply(self, status, body):
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body.encode())

        def do_GET(self):
            parsed = urllib.parse.urlsplit(self.path)
            if parsed.path == "/":
                encoded = html.escape(json.dumps(manifest), quote=True)
                self.reply(
                    200,
                    f'<h1>Register Headroom release App</h1><form method="post" action="{html.escape(action, quote=True)}"><input type="hidden" name="manifest" value="{encoded}"><button>Register on GitHub</button></form>',
                )
                return
            if parsed.path != "/callback":
                self.reply(404, "Not found")
                return
            try:
                code = callback_code(parsed.query, state)
            except ValueError:
                self.reply(400, "Invalid callback; registration remains open.")
                return
            try:
                request = urllib.request.Request(
                    f"https://api.github.com/app-manifests/{code}/conversions",
                    data=b"",
                    method="POST",
                    headers={
                        "Accept": "application/vnd.github+json",
                        "User-Agent": "headroom-release-app",
                        "X-GitHub-Api-Version": "2022-11-28",
                    },
                )
                with urllib.request.urlopen(request, timeout=30) as response:
                    app = json.load(response)
                configure_repository(args.repo, app)
                install = f"https://github.com/apps/{urllib.parse.quote(app['slug'], safe='')}/installations/new"
                self.reply(
                    200,
                    f'<h1>App credentials configured</h1><a href="{install}">Install App on selected repositories</a>',
                )
                print(f"Configured {args.repo}. Install the App: {install}")
                webbrowser.open(install)
                outcome["success"] = True
            except Exception:
                self.reply(
                    500,
                    "Registration or repository configuration failed. Recover in GitHub App settings; no credentials were logged.",
                )
                print(
                    "Registration or configuration failed. Check GitHub App settings and follow the recovery instructions."
                )
            finally:
                threading.Thread(target=self.server.shutdown, daemon=True).start()

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        base = f"http://127.0.0.1:{server.server_port}"
        manifest["redirect_url"] = f"{base}/callback"
        print(f"Open {base} to register the App. Ctrl+C cancels.")
        webbrowser.open(base)
        server.serve_forever()
    if not outcome["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
