#!/usr/bin/env bash

set -euo pipefail

IMAGE_DEFAULT="ghcr.io/chopratejas/headroom:latest"
INSTALL_DIR="${HOME}/.local/bin"
if [[ ! -d "${HOME}/.local" ]]; then
  INSTALL_DIR="${HOME}/bin"
fi

info() {
  printf '==> %s\n' "$*"
}

warn() {
  printf 'WARN: %s\n' "$*" >&2
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

append_path_block() {
  local target_file="$1"
  local marker_start="# >>> headroom docker-native >>>"
  local marker_end="# <<< headroom docker-native <<<"
  local block="${marker_start}
export PATH=\"${INSTALL_DIR}:\$PATH\"
${marker_end}"

  touch "${target_file}"
  if grep -Fq "${marker_start}" "${target_file}"; then
    return
  fi

  {
    printf '\n%s\n' "${block}"
  } >>"${target_file}"
}

write_wrapper() {
  local wrapper_path="${INSTALL_DIR}/headroom"

  cat >"${wrapper_path}" <<'WRAPPER'
#!/usr/bin/env bash

set -euo pipefail

HEADROOM_IMAGE="${HEADROOM_DOCKER_IMAGE:-ghcr.io/chopratejas/headroom:latest}"
HEADROOM_CONTAINER_HOME="${HEADROOM_CONTAINER_HOME:-/tmp/headroom-home}"
HEADROOM_HOST_HOME="${HOME:?}"

warn() {
  printf 'WARN: %s\n' "$*" >&2
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

detect_rtk_target() {
  local system
  local machine
  system="$(uname -s)"
  machine="$(uname -m)"

  case "${system}" in
    Darwin)
      if [[ "${machine}" == "arm64" ]]; then
        printf 'aarch64-apple-darwin'
      else
        printf 'x86_64-apple-darwin'
      fi
      ;;
    Linux)
      if [[ "${machine}" == "aarch64" ]]; then
        printf 'aarch64-unknown-linux-gnu'
      else
        printf 'x86_64-unknown-linux-musl'
      fi
      ;;
    *)
      die "Unsupported host platform for Docker-native wrapper: ${system}/${machine}"
      ;;
  esac
}

ensure_host_dirs() {
  mkdir -p \
    "${HEADROOM_HOST_HOME}/.headroom" \
    "${HEADROOM_HOST_HOME}/.claude" \
    "${HEADROOM_HOST_HOME}/.codex" \
    "${HEADROOM_HOST_HOME}/.gemini"
}

append_passthrough_envs() {
  local -n ref=$1
  local name

  for name in $(compgen -e); do
    case "${name}" in
      HEADROOM_*|ANTHROPIC_*|OPENAI_*|GEMINI_*|AWS_*|AZURE_*|VERTEX_*|GOOGLE_*|GOOGLE_CLOUD_*|MISTRAL_*|GROQ_*|OPENROUTER_*|XAI_*|TOGETHER_*|COHERE_*|OLLAMA_*|LITELLM_*|OTEL_*|SUPABASE_*|QDRANT_*|NEO4J_*|LANGSMITH_*)
        ref+=(--env "${name}")
        ;;
    esac
  done
}

append_common_container_args() {
  local -n ref=$1

  ensure_host_dirs
  ref+=(-w /workspace)
  ref+=(--env "HOME=${HEADROOM_CONTAINER_HOME}")
  ref+=(--env "PYTHONUNBUFFERED=1")
  ref+=(-v "${PWD}:/workspace")
  ref+=(-v "${HEADROOM_HOST_HOME}/.headroom:${HEADROOM_CONTAINER_HOME}/.headroom")
  ref+=(-v "${HEADROOM_HOST_HOME}/.claude:${HEADROOM_CONTAINER_HOME}/.claude")
  ref+=(-v "${HEADROOM_HOST_HOME}/.codex:${HEADROOM_CONTAINER_HOME}/.codex")
  ref+=(-v "${HEADROOM_HOST_HOME}/.gemini:${HEADROOM_CONTAINER_HOME}/.gemini")

  if command -v id >/dev/null 2>&1; then
    ref+=(--user "$(id -u):$(id -g)")
  fi

  append_passthrough_envs ref
}

append_tty_args() {
  local -n ref=$1

  if [[ -t 0 && -t 1 ]]; then
    ref+=(-it)
  elif [[ -t 0 ]]; then
    ref+=(-i)
  elif [[ -t 1 ]]; then
    ref+=(-t)
  fi
}

run_headroom() {
  local args=()
  args=(docker run --rm)
  append_tty_args args
  append_common_container_args args
  args+=(--entrypoint headroom "${HEADROOM_IMAGE}" "$@")
  "${args[@]}"
}

docker_container_exists() {
  local name="$1"
  docker ps --format '{{.Names}}' | grep -Fxq "${name}"
}

wait_for_proxy() {
  local container_name="$1"
  local port="$2"
  local attempt

  for attempt in $(seq 1 45); do
    if (echo >/dev/tcp/127.0.0.1/"${port}") >/dev/null 2>&1; then
      return 0
    fi

    if ! docker_container_exists "${container_name}"; then
      break
    fi

    sleep 1
  done

  docker logs "${container_name}" >&2 || true
  return 1
}

start_proxy_container() {
  local port="$1"
  shift

  local container_name="headroom-proxy-${port}-$$"
  local args=()
  args=(docker run -d --rm --name "${container_name}" -p "${port}:${port}")
  append_common_container_args args
  args+=("${HEADROOM_IMAGE}" --host 0.0.0.0 --port "${port}" "$@")
  "${args[@]}" >/dev/null

  if ! wait_for_proxy "${container_name}" "${port}"; then
    docker stop "${container_name}" >/dev/null 2>&1 || true
    die "Headroom proxy failed to start on port ${port}"
  fi

  printf '%s\n' "${container_name}"
}

stop_proxy_container() {
  local container_name="${1:-}"
  if [[ -n "${container_name}" ]]; then
    docker stop "${container_name}" >/dev/null 2>&1 || true
  fi
}

run_claude_rtk_init() {
  local rtk_bin="${HEADROOM_HOST_HOME}/.headroom/bin/rtk"
  if [[ ! -x "${rtk_bin}" ]]; then
    warn "rtk was not installed at ${rtk_bin}; Claude hooks were not registered"
    return
  fi

  if ! "${rtk_bin}" init --global --auto-patch >/dev/null 2>&1; then
    warn "Failed to register Claude hooks with rtk; continuing without hook registration"
  fi
}

parse_wrap_args() {
  local -n out_known=$1
  local -n out_host=$2
  local -n out_port=$3
  local -n out_no_rtk=$4
  local -n out_no_proxy=$5
  local -n out_learn=$6
  local -n out_backend=$7
  local -n out_anyllm=$8
  local -n out_region=$9
  shift 9

  out_known=()
  out_host=()
  out_port=8787
  out_no_rtk=0
  out_no_proxy=0
  out_learn=0
  out_backend=""
  out_anyllm=""
  out_region=""

  while (($#)); do
    case "$1" in
      --)
        shift
        out_host+=("$@")
        break
        ;;
      --port|-p)
        out_port="$2"
        out_known+=("$1" "$2")
        shift 2
        ;;
      --port=*)
        out_port="${1#*=}"
        out_known+=("$1")
        shift
        ;;
      --no-rtk)
        out_no_rtk=1
        out_known+=("$1")
        shift
        ;;
      --no-proxy)
        out_no_proxy=1
        out_known+=("$1")
        shift
        ;;
      --learn)
        out_learn=1
        out_known+=("$1")
        shift
        ;;
      --verbose|-v)
        out_known+=("$1")
        shift
        ;;
      --backend)
        out_backend="$2"
        out_known+=("$1" "$2")
        shift 2
        ;;
      --backend=*)
        out_backend="${1#*=}"
        out_known+=("$1")
        shift
        ;;
      --anyllm-provider)
        out_anyllm="$2"
        out_known+=("$1" "$2")
        shift 2
        ;;
      --anyllm-provider=*)
        out_anyllm="${1#*=}"
        out_known+=("$1")
        shift
        ;;
      --region)
        out_region="$2"
        out_known+=("$1" "$2")
        shift 2
        ;;
      --region=*)
        out_region="${1#*=}"
        out_known+=("$1")
        shift
        ;;
      *)
        out_host+=("$@")
        break
        ;;
    esac
  done
}

run_prepare_only() {
  local tool="$1"
  shift

  local args=()
  args=(docker run --rm)
  append_tty_args args
  append_common_container_args args
  args+=(--env "HEADROOM_RTK_TARGET=$(detect_rtk_target)")
  args+=(--entrypoint headroom "${HEADROOM_IMAGE}" wrap "${tool}" --prepare-only "$@")
  "${args[@]}"
}

run_host_tool() {
  local binary="$1"
  shift

  command -v "${binary}" >/dev/null 2>&1 || die "'${binary}' not found in PATH"
  "${binary}" "$@"
}

contains_help_flag() {
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == "--help" || "${arg}" == "-?" ]]; then
      return 0
    fi
  done

  return 1
}

parse_openclaw_wrap_args() {
  local -n out_plugin_path=$1
  local -n out_plugin_spec=$2
  local -n out_skip_build=$3
  local -n out_copy=$4
  local -n out_proxy_port=$5
  local -n out_startup_timeout_ms=$6
  local -n out_gateway_provider_ids=$7
  local -n out_python_path=$8
  local -n out_no_auto_start=$9
  local -n out_no_restart=${10}
  local -n out_verbose=${11}
  shift 11

  out_plugin_path=""
  out_plugin_spec="headroom-ai/openclaw"
  out_skip_build=0
  out_copy=0
  out_proxy_port=8787
  out_startup_timeout_ms=20000
  out_gateway_provider_ids=()
  out_python_path=""
  out_no_auto_start=0
  out_no_restart=0
  out_verbose=0

  while (($#)); do
    case "$1" in
      --plugin-path)
        out_plugin_path="$2"
        shift 2
        ;;
      --plugin-path=*)
        out_plugin_path="${1#*=}"
        shift
        ;;
      --plugin-spec)
        out_plugin_spec="$2"
        shift 2
        ;;
      --plugin-spec=*)
        out_plugin_spec="${1#*=}"
        shift
        ;;
      --skip-build)
        out_skip_build=1
        shift
        ;;
      --copy)
        out_copy=1
        shift
        ;;
      --proxy-port)
        out_proxy_port="$2"
        shift 2
        ;;
      --proxy-port=*)
        out_proxy_port="${1#*=}"
        shift
        ;;
      --startup-timeout-ms)
        out_startup_timeout_ms="$2"
        shift 2
        ;;
      --startup-timeout-ms=*)
        out_startup_timeout_ms="${1#*=}"
        shift
        ;;
      --gateway-provider-id)
        out_gateway_provider_ids+=("$2")
        shift 2
        ;;
      --gateway-provider-id=*)
        out_gateway_provider_ids+=("${1#*=}")
        shift
        ;;
      --python-path)
        out_python_path="$2"
        shift 2
        ;;
      --python-path=*)
        out_python_path="${1#*=}"
        shift
        ;;
      --no-auto-start)
        out_no_auto_start=1
        shift
        ;;
      --no-restart)
        out_no_restart=1
        shift
        ;;
      --verbose|-v)
        out_verbose=1
        shift
        ;;
      *)
        die "Unsupported option for 'headroom wrap openclaw': $1"
        ;;
    esac
  done
}

parse_openclaw_unwrap_args() {
  local -n out_no_restart=$1
  local -n out_verbose=$2
  shift 2

  out_no_restart=0
  out_verbose=0

  while (($#)); do
    case "$1" in
      --no-restart)
        out_no_restart=1
        shift
        ;;
      --verbose|-v)
        out_verbose=1
        shift
        ;;
      *)
        die "Unsupported option for 'headroom unwrap openclaw': $1"
        ;;
    esac
  done
}

get_openclaw_existing_entry_json() {
  local output=""
  if output="$(openclaw config get plugins.entries.headroom 2>/dev/null)"; then
    printf '%s' "${output}"
  fi
}

prepare_openclaw_entry_json() {
  local existing_entry_json="$1"
  local proxy_port="$2"
  local startup_timeout_ms="$3"
  local python_path="$4"
  local no_auto_start="$5"
  shift 5
  local gateway_provider_ids=("$@")
  local args=()
  args=(docker run --rm)
  append_common_container_args args
  args+=(--entrypoint headroom "${HEADROOM_IMAGE}" wrap openclaw --prepare-only)
  args+=(--proxy-port "${proxy_port}" --startup-timeout-ms "${startup_timeout_ms}")

  if [[ -n "${existing_entry_json}" ]]; then
    args+=(--existing-entry-json "${existing_entry_json}")
  fi
  if [[ -n "${python_path}" ]]; then
    args+=(--python-path "${python_path}")
  fi
  if [[ "${no_auto_start}" -eq 1 ]]; then
    args+=(--no-auto-start)
  fi

  local provider_id
  for provider_id in "${gateway_provider_ids[@]}"; do
    args+=(--gateway-provider-id "${provider_id}")
  done

  "${args[@]}"
}

prepare_openclaw_unwrap_entry_json() {
  local existing_entry_json="$1"
  local args=()
  args=(docker run --rm)
  append_common_container_args args
  args+=(--entrypoint headroom "${HEADROOM_IMAGE}" unwrap openclaw --prepare-only)
  if [[ -n "${existing_entry_json}" ]]; then
    args+=(--existing-entry-json "${existing_entry_json}")
  fi
  "${args[@]}"
}

run_openclaw_checked() {
  local action="$1"
  shift
  local output=""

  if ! output="$("$@" 2>&1)"; then
    output="${output//$'\r'/}"
    die "${action} failed: ${output:-unknown error}"
  fi

  printf '%s' "${output//$'\r'/}"
}

run_openclaw_checked_in_dir() {
  local action="$1"
  local cwd="$2"
  shift 2
  local output=""

  if ! output="$(cd "${cwd}" && "$@" 2>&1)"; then
    output="${output//$'\r'/}"
    die "${action} failed: ${output:-unknown error}"
  fi

  printf '%s' "${output//$'\r'/}"
}

resolve_openclaw_extensions_dir() {
  local config_output
  config_output="$(run_openclaw_checked "openclaw config file" openclaw config file)"
  local config_path
  config_path="$(printf '%s\n' "${config_output}" | tail -n 1)"
  [[ -n "${config_path}" ]] || die "Unable to resolve OpenClaw config path."
  printf '%s\n' "$(dirname "${config_path}")/extensions"
}

copy_openclaw_plugin_into_extensions() {
  local plugin_dir="$1"
  local dist_dir="${plugin_dir}/dist"
  local hook_shim_dir="${plugin_dir}/hook-shim"
  [[ -d "${dist_dir}" ]] || die "Plugin dist folder missing at ${dist_dir}. Build the plugin first."
  [[ -d "${hook_shim_dir}" ]] || die "Plugin hook-shim folder missing at ${hook_shim_dir}. Build the plugin first."

  local extensions_dir
  extensions_dir="$(resolve_openclaw_extensions_dir)"
  local target_dir="${extensions_dir}/headroom"
  mkdir -p "${target_dir}"
  rm -rf "${target_dir}/dist" "${target_dir}/hook-shim"
  cp -R "${dist_dir}" "${target_dir}/dist"
  cp -R "${hook_shim_dir}" "${target_dir}/hook-shim"

  local filename
  for filename in openclaw.plugin.json package.json README.md; do
    if [[ -f "${plugin_dir}/${filename}" ]]; then
      cp "${plugin_dir}/${filename}" "${target_dir}/${filename}"
    fi
  done

  printf '%s\n' "${target_dir}"
}

install_openclaw_plugin() {
  local plugin_path="$1"
  local plugin_spec="$2"
  local skip_build="$3"
  local copy_mode="$4"
  local verbose="$5"

  local local_source_mode=0
  if [[ -n "${plugin_path}" ]]; then
    local_source_mode=1
    [[ -d "${plugin_path}" ]] || die "Plugin path not found: ${plugin_path}."
    [[ -f "${plugin_path}/package.json" ]] || die "Invalid plugin path (missing package.json): ${plugin_path}"
    [[ -f "${plugin_path}/openclaw.plugin.json" ]] || die "Invalid plugin path (missing openclaw.plugin.json): ${plugin_path}"
  fi

  if [[ "${local_source_mode}" -eq 1 && "${skip_build}" -eq 0 ]]; then
    require_cmd npm
    info "Building OpenClaw plugin (npm install + npm run build)..."
    run_openclaw_checked_in_dir "npm install" "${plugin_path}" npm install >/dev/null
    run_openclaw_checked_in_dir "npm run build" "${plugin_path}" npm run build >/dev/null
  fi

  local install_output=""
  local install_status=0
  set +e
  if [[ "${local_source_mode}" -eq 1 ]]; then
    if [[ "${copy_mode}" -eq 1 ]]; then
      install_output="$(openclaw plugins install --dangerously-force-unsafe-install "${plugin_path}" 2>&1)"
      install_status=$?
    else
      install_output="$(cd "${plugin_path}" && openclaw plugins install --dangerously-force-unsafe-install --link . 2>&1)"
      install_status=$?
    fi
  else
    install_output="$(openclaw plugins install --dangerously-force-unsafe-install "${plugin_spec}" 2>&1)"
    install_status=$?
  fi
  set -e
  install_output="${install_output//$'\r'/}"

  if [[ "${install_status}" -eq 0 ]]; then
    if [[ "${verbose}" -eq 1 && -n "${install_output}" ]]; then
      printf '%s\n' "${install_output}"
    fi
    return
  fi

  local lower_output="${install_output,,}"
  if [[ "${lower_output}" == *"plugin already exists"* ]]; then
    info "Plugin already installed; continuing with configuration/update steps."
    return
  fi

  if [[ "${lower_output}" == *"also not a valid hook pack"* && "${local_source_mode}" -eq 1 && "${copy_mode}" -eq 0 ]]; then
    info "OpenClaw linked-path install bug detected; applying extension-path fallback..."
    local target_dir
    target_dir="$(copy_openclaw_plugin_into_extensions "${plugin_path}")"
    info "Fallback plugin copy completed: ${target_dir}"
    return
  fi

  die "openclaw plugins install failed: ${install_output:-exit code ${install_status}}"
}

restart_or_start_openclaw_gateway() {
  local output=""
  if output="$(openclaw gateway restart 2>&1)"; then
    OPENCLAW_GATEWAY_ACTION="restarted"
    OPENCLAW_GATEWAY_OUTPUT="${output//$'\r'/}"
    return
  fi

  OPENCLAW_GATEWAY_OUTPUT="$(run_openclaw_checked "openclaw gateway start" openclaw gateway start)"
  OPENCLAW_GATEWAY_ACTION="started"
}

wrap_openclaw_host() {
  local plugin_path plugin_spec skip_build copy_mode proxy_port startup_timeout_ms python_path
  local no_auto_start no_restart verbose
  local gateway_provider_ids=()

  parse_openclaw_wrap_args \
    plugin_path \
    plugin_spec \
    skip_build \
    copy_mode \
    proxy_port \
    startup_timeout_ms \
    gateway_provider_ids \
    python_path \
    no_auto_start \
    no_restart \
    verbose \
    "$@"

  require_cmd openclaw
  local existing_entry_json=""
  existing_entry_json="$(get_openclaw_existing_entry_json)"
  local entry_json
  entry_json="$(prepare_openclaw_entry_json "${existing_entry_json}" "${proxy_port}" "${startup_timeout_ms}" "${python_path}" "${no_auto_start}" "${gateway_provider_ids[@]}")"

  printf '\n  ╔═══════════════════════════════════════════════╗\n'
  printf '  ║           HEADROOM WRAP: OPENCLAW             ║\n'
  printf '  ╚═══════════════════════════════════════════════╝\n\n'
  if [[ -n "${plugin_path}" ]]; then
    printf '  Plugin source: local (%s)\n' "${plugin_path}"
  else
    printf '  Plugin source: npm (%s)\n' "${plugin_spec}"
  fi

  printf '  Writing plugin configuration...\n'
  run_openclaw_checked \
    "openclaw config set plugins.entries.headroom" \
    openclaw config set plugins.entries.headroom "${entry_json}" --strict-json >/dev/null

  printf '  Installing OpenClaw plugin with required unsafe-install flag...\n'
  install_openclaw_plugin "${plugin_path}" "${plugin_spec}" "${skip_build}" "${copy_mode}" "${verbose}"

  run_openclaw_checked \
    "openclaw config set plugins.slots.contextEngine" \
    openclaw config set plugins.slots.contextEngine '"headroom"' --strict-json >/dev/null
  run_openclaw_checked "openclaw config validate" openclaw config validate >/dev/null

  if [[ "${no_restart}" -eq 1 ]]; then
    printf '  Skipping gateway restart (--no-restart).\n'
    printf '  Run `openclaw gateway restart` (or `openclaw gateway start`) to apply plugin changes.\n'
  else
    printf '  Applying plugin changes to OpenClaw gateway...\n'
    restart_or_start_openclaw_gateway
    printf '  Gateway %s.\n' "${OPENCLAW_GATEWAY_ACTION}"
    if [[ "${verbose}" -eq 1 && -n "${OPENCLAW_GATEWAY_OUTPUT}" ]]; then
      printf '%s\n' "${OPENCLAW_GATEWAY_OUTPUT}"
    fi
  fi

  local inspect_output=""
  inspect_output="$(run_openclaw_checked "openclaw plugins inspect headroom" openclaw plugins inspect headroom)"
  if [[ "${verbose}" -eq 1 && -n "${inspect_output}" ]]; then
    printf '%s\n' "${inspect_output}"
  fi

  printf '\n✓ OpenClaw is configured to use Headroom context compression.\n'
  printf '  Plugin: headroom\n'
  printf '  Slot:   plugins.slots.contextEngine = headroom\n\n'
}

unwrap_openclaw_host() {
  local no_restart verbose
  parse_openclaw_unwrap_args no_restart verbose "$@"

  require_cmd openclaw
  local existing_entry_json=""
  existing_entry_json="$(get_openclaw_existing_entry_json)"
  local entry_json
  entry_json="$(prepare_openclaw_unwrap_entry_json "${existing_entry_json}")"

  printf '\n  ╔═══════════════════════════════════════════════╗\n'
  printf '  ║          HEADROOM UNWRAP: OPENCLAW            ║\n'
  printf '  ╚═══════════════════════════════════════════════╝\n\n'
  printf '  Disabling Headroom plugin and removing engine mapping...\n'

  run_openclaw_checked \
    "openclaw config set plugins.entries.headroom" \
    openclaw config set plugins.entries.headroom "${entry_json}" --strict-json >/dev/null
  run_openclaw_checked \
    "openclaw config set plugins.slots.contextEngine" \
    openclaw config set plugins.slots.contextEngine '"legacy"' --strict-json >/dev/null
  run_openclaw_checked "openclaw config validate" openclaw config validate >/dev/null

  if [[ "${no_restart}" -eq 1 ]]; then
    printf '  Skipping gateway restart (--no-restart).\n'
    printf '  Run `openclaw gateway restart` (or `openclaw gateway start`) to apply unwrap changes.\n'
  else
    printf '  Applying unwrap changes to OpenClaw gateway...\n'
    restart_or_start_openclaw_gateway
    printf '  Gateway %s.\n' "${OPENCLAW_GATEWAY_ACTION}"
    if [[ "${verbose}" -eq 1 && -n "${OPENCLAW_GATEWAY_OUTPUT}" ]]; then
      printf '%s\n' "${OPENCLAW_GATEWAY_OUTPUT}"
    fi
  fi

  if [[ "${verbose}" -eq 1 ]]; then
    local inspect_output=""
    inspect_output="$(run_openclaw_checked "openclaw plugins inspect headroom" openclaw plugins inspect headroom)"
    if [[ -n "${inspect_output}" ]]; then
      printf '%s\n' "${inspect_output}"
    fi
  fi

  printf '\n✓ OpenClaw Headroom wrap removed.\n'
  printf '  Plugin: headroom (installed, disabled)\n'
  printf '  Slot:   plugins.slots.contextEngine = legacy\n\n'
}

main() {
  require_cmd docker

  if (($# == 0)); then
    run_headroom --help
    return
  fi

  case "$1" in
    wrap)
      (($# >= 2)) || die "Usage: headroom wrap <claude|codex|aider|cursor|openclaw> [...]"
      local tool="$2"
      shift 2

      if [[ "${tool}" == "openclaw" ]]; then
        if contains_help_flag "$@"; then
          run_headroom wrap openclaw "$@"
          return
        fi
        wrap_openclaw_host "$@"
        return
      fi

      local known_args host_args port no_rtk no_proxy learn backend anyllm region
      parse_wrap_args known_args host_args port no_rtk no_proxy learn backend anyllm region "$@"

      local proxy_args=()
      if [[ "${learn}" -eq 1 ]]; then
        proxy_args+=(--learn)
      fi
      if [[ -n "${backend}" ]]; then
        proxy_args+=(--backend "${backend}")
      fi
      if [[ -n "${anyllm}" ]]; then
        proxy_args+=(--anyllm-provider "${anyllm}")
      fi
      if [[ -n "${region}" ]]; then
        proxy_args+=(--region "${region}")
      fi

      case "${tool}" in
        claude|codex|aider|cursor)
          ;;
        *)
          die "Unsupported wrap target: ${tool}"
          ;;
      esac

      local container_name=""
      if [[ "${no_proxy}" -eq 0 ]]; then
        container_name="$(start_proxy_container "${port}" "${proxy_args[@]}")"
      fi
      trap 'stop_proxy_container "${container_name}"' EXIT INT TERM

      local prep_args=("${known_args[@]}")
      if [[ "${no_proxy}" -eq 0 ]]; then
        prep_args+=(--no-proxy)
      fi
      run_prepare_only "${tool}" "${prep_args[@]}"

      case "${tool}" in
        claude)
          if [[ "${no_rtk}" -eq 0 ]]; then
            run_claude_rtk_init
          fi
          ANTHROPIC_BASE_URL="http://127.0.0.1:${port}" run_host_tool claude "${host_args[@]}"
          ;;
        codex)
          OPENAI_BASE_URL="http://127.0.0.1:${port}/v1" run_host_tool codex "${host_args[@]}"
          ;;
        aider)
          OPENAI_API_BASE="http://127.0.0.1:${port}/v1" \
          ANTHROPIC_BASE_URL="http://127.0.0.1:${port}" \
          run_host_tool aider "${host_args[@]}"
          ;;
        cursor)
          cat <<EOF
Headroom proxy is running for Cursor.

OpenAI base URL:     http://127.0.0.1:${port}/v1
Anthropic base URL:  http://127.0.0.1:${port}

Press Ctrl+C to stop the proxy.
EOF
          while true; do
            sleep 1
          done
          ;;
      esac
      ;;
    unwrap)
      if (($# >= 2)) && [[ "$2" == "openclaw" ]]; then
        shift 2
        if contains_help_flag "$@"; then
          run_headroom unwrap openclaw "$@"
          return
        fi
        unwrap_openclaw_host "$@"
        return
      fi
      run_headroom "$@"
      ;;
    proxy)
      shift
      local port=8787
      local args=()
      args=(proxy)
      while (($#)); do
        case "$1" in
          --port|-p)
            port="$2"
            args+=("$1" "$2")
            shift 2
            ;;
          --port=*)
            port="${1#*=}"
            args+=("$1")
            shift
            ;;
          *)
            args+=("$1")
            shift
            ;;
        esac
      done
      local run_args=()
      run_args=(docker run --rm)
      append_tty_args run_args
      append_common_container_args run_args
      run_args+=(-p "${port}:${port}")
      run_args+=(--entrypoint headroom "${HEADROOM_IMAGE}" "${args[@]}")
      "${run_args[@]}"
      ;;
    *)
      run_headroom "$@"
      ;;
  esac
}

main "$@"
WRAPPER

  chmod +x "${wrapper_path}"
}

main() {
  require_cmd docker
  docker version >/dev/null 2>&1 || die "Docker is installed but not available to the current user"

  mkdir -p "${INSTALL_DIR}"
  write_wrapper

  append_path_block "${HOME}/.bashrc"
  append_path_block "${HOME}/.zshrc"
  append_path_block "${HOME}/.profile"

  info "Pulling ${IMAGE_DEFAULT}"
  docker pull "${IMAGE_DEFAULT}" >/dev/null

  cat <<EOF

Headroom Docker-native install complete.

Installed wrapper:
  ${INSTALL_DIR}/headroom

Next steps:
  1. Restart your shell or run: export PATH="${INSTALL_DIR}:\$PATH"
  2. Try: headroom proxy
  3. Docs: https://github.com/chopratejas/headroom/blob/main/docs/docker-install.md
EOF
}

main "$@"
