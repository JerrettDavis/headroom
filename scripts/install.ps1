$ErrorActionPreference = 'Stop'

$ImageDefault = 'ghcr.io/chopratejas/headroom:latest'
$InstallDir = Join-Path $HOME '.local\bin'
if (-not (Test-Path (Join-Path $HOME '.local'))) {
    $InstallDir = Join-Path $HOME 'bin'
}

function Write-Info {
    param([string]$Message)
    Write-Host "==> $Message"
}

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Missing required command: $Name"
    }
}

function Ensure-PathEntry {
    param([string]$PathEntry)

    $currentPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $parts = @()
    if ($currentPath) {
        $parts = $currentPath -split ';' | Where-Object { $_ }
    }
    if ($parts -notcontains $PathEntry) {
        $newPath = @($PathEntry) + $parts
        [Environment]::SetEnvironmentVariable('Path', ($newPath -join ';'), 'User')
    }
}

function Ensure-ProfileBlock {
    param([string]$PathEntry)

    $markerStart = '# >>> headroom docker-native >>>'
    $markerEnd = '# <<< headroom docker-native <<<'
    $block = @"
$markerStart
if (-not ((`$env:Path -split ';') -contains '$PathEntry')) {
    `$env:Path = '$PathEntry;' + `$env:Path
}
$markerEnd
"@

    $profileDir = Split-Path -Parent $PROFILE
    if (-not (Test-Path $profileDir)) {
        New-Item -ItemType Directory -Force -Path $profileDir | Out-Null
    }
    if (-not (Test-Path $PROFILE)) {
        New-Item -ItemType File -Force -Path $PROFILE | Out-Null
    }

    $existing = Get-Content -Raw -Path $PROFILE
    if ($existing -notmatch [regex]::Escape($markerStart)) {
        Add-Content -Path $PROFILE -Value "`n$block"
    }
}

function Write-Wrapper {
    param([string]$TargetDir)

    $wrapperPath = Join-Path $TargetDir 'headroom.ps1'
    $cmdPath = Join-Path $TargetDir 'headroom.cmd'

    $wrapper = @'
$ErrorActionPreference = 'Stop'

$HeadroomImage = if ($env:HEADROOM_DOCKER_IMAGE) { $env:HEADROOM_DOCKER_IMAGE } else { 'ghcr.io/chopratejas/headroom:latest' }
$ContainerHome = if ($env:HEADROOM_CONTAINER_HOME) { $env:HEADROOM_CONTAINER_HOME } else { '/tmp/headroom-home' }
$HostHome = $HOME

function Fail {
    param([string]$Message)
    throw $Message
}

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Fail "Missing required command: $Name"
    }
}

function Get-RtkTarget {
    $arch = if ($env:PROCESSOR_ARCHITECTURE -match 'ARM64') { 'aarch64' } else { 'x86_64' }
    return "${arch}-pc-windows-msvc"
}

function Ensure-HostDirs {
    foreach ($dir in @(
        (Join-Path $HostHome '.headroom'),
        (Join-Path $HostHome '.claude'),
        (Join-Path $HostHome '.codex'),
        (Join-Path $HostHome '.gemini')
    )) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Force -Path $dir | Out-Null
        }
    }
}

function Get-PassthroughEnvArgs {
    $args = New-Object System.Collections.Generic.List[string]
    $prefixes = @(
        'HEADROOM_','ANTHROPIC_','OPENAI_','GEMINI_','AWS_','AZURE_','VERTEX_',
        'GOOGLE_','GOOGLE_CLOUD_','MISTRAL_','GROQ_','OPENROUTER_','XAI_',
        'TOGETHER_','COHERE_','OLLAMA_','LITELLM_','OTEL_','SUPABASE_',
        'QDRANT_','NEO4J_','LANGSMITH_'
    )

    foreach ($item in Get-ChildItem Env:) {
        foreach ($prefix in $prefixes) {
            if ($item.Name.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
                $args.Add('--env')
                $args.Add($item.Name)
                break
            }
        }
    }

    return ,$args.ToArray()
}

function Get-SharedDockerArgs {
    Ensure-HostDirs
    $args = New-Object System.Collections.Generic.List[string]
    $args.Add('--workdir')
    $args.Add('/workspace')
    $args.Add('--env')
    $args.Add("HOME=$ContainerHome")
    $args.Add('--env')
    $args.Add('PYTHONUNBUFFERED=1')
    $args.Add('--volume')
    $args.Add("${PWD}:/workspace")
    $args.Add('--volume')
    $args.Add((Join-Path $HostHome '.headroom') + ":$ContainerHome/.headroom")
    $args.Add('--volume')
    $args.Add((Join-Path $HostHome '.claude') + ":$ContainerHome/.claude")
    $args.Add('--volume')
    $args.Add((Join-Path $HostHome '.codex') + ":$ContainerHome/.codex")
    $args.Add('--volume')
    $args.Add((Join-Path $HostHome '.gemini') + ":$ContainerHome/.gemini")

    foreach ($entry in (Get-PassthroughEnvArgs)) {
        $args.Add($entry)
    }

    return ,$args.ToArray()
}

function Invoke-HeadroomDocker {
    param([string[]]$Arguments)

    $dockerArgs = New-Object System.Collections.Generic.List[string]
    $dockerArgs.AddRange(@('run','--rm','-it'))
    $dockerArgs.AddRange((Get-SharedDockerArgs))
    $dockerArgs.Add('--entrypoint')
    $dockerArgs.Add('headroom')
    $dockerArgs.Add($HeadroomImage)
    foreach ($arg in $Arguments) {
        $dockerArgs.Add($arg)
    }

    & docker @dockerArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

function Wait-Proxy {
    param(
        [string]$ContainerName,
        [int]$Port
    )

    for ($attempt = 0; $attempt -lt 45; $attempt++) {
        try {
            Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/readyz" | Out-Null
            return
        } catch {
            $running = docker ps --format '{{.Names}}'
            if ($running -notcontains $ContainerName) {
                break
            }
            Start-Sleep -Seconds 1
        }
    }

    docker logs $ContainerName | Write-Error
    throw "Headroom proxy failed to start on port $Port"
}

function Start-ProxyContainer {
    param(
        [int]$Port,
        [string[]]$ProxyArgs
    )

    $containerName = "headroom-proxy-$Port-$PID"
    $dockerArgs = New-Object System.Collections.Generic.List[string]
    $dockerArgs.AddRange(@('run','-d','--rm','--name',$containerName,'-p',"$Port`:$Port"))
    $dockerArgs.AddRange((Get-SharedDockerArgs))
    $dockerArgs.Add($HeadroomImage)
    $dockerArgs.Add('--host')
    $dockerArgs.Add('0.0.0.0')
    $dockerArgs.Add('--port')
    $dockerArgs.Add("$Port")
    foreach ($arg in $ProxyArgs) {
        $dockerArgs.Add($arg)
    }

    & docker @dockerArgs | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to start Headroom proxy container"
    }

    Wait-Proxy -ContainerName $containerName -Port $Port
    return $containerName
}

function Stop-ProxyContainer {
    param([string]$ContainerName)
    if ($ContainerName) {
        docker stop $ContainerName | Out-Null
    }
}

function Invoke-ClaudeRtkInit {
    $rtkPath = Join-Path $HostHome '.headroom\bin\rtk.exe'
    if (-not (Test-Path $rtkPath)) {
        Write-Warning "rtk was not installed at $rtkPath; Claude hooks were not registered"
        return
    }

    try {
        & $rtkPath init --global --auto-patch | Out-Null
    } catch {
        Write-Warning "Failed to register Claude hooks with rtk; continuing without hook registration"
    }
}

function Invoke-WithTemporaryEnv {
    param(
        [hashtable]$Environment,
        [string]$Command,
        [string[]]$Arguments
    )

    $previous = @{}
    foreach ($pair in $Environment.GetEnumerator()) {
        $previous[$pair.Key] = [Environment]::GetEnvironmentVariable($pair.Key, 'Process')
        [Environment]::SetEnvironmentVariable($pair.Key, $pair.Value, 'Process')
    }

    try {
        & $Command @Arguments
        return $LASTEXITCODE
    } finally {
        foreach ($pair in $Environment.GetEnumerator()) {
            [Environment]::SetEnvironmentVariable($pair.Key, $previous[$pair.Key], 'Process')
        }
    }
}

function Test-HelpFlag {
    param([string[]]$Arguments)

    foreach ($arg in $Arguments) {
        if ($arg -eq '--help' -or $arg -eq '-?') {
            return $true
        }
    }

    return $false
}

function Parse-OpenClawWrapArgs {
    param([string[]]$Arguments)

    $gatewayProviderIds = New-Object System.Collections.Generic.List[string]
    $pluginPath = $null
    $pluginSpec = 'headroom-ai/openclaw'
    $skipBuild = $false
    $copy = $false
    $proxyPort = 8787
    $startupTimeoutMs = 20000
    $pythonPath = $null
    $noAutoStart = $false
    $noRestart = $false
    $verbose = $false

    $i = 0
    while ($i -lt $Arguments.Count) {
        $arg = $Arguments[$i]
        switch -Regex ($arg) {
            '^--plugin-path$' {
                $pluginPath = $Arguments[$i + 1]
                $i += 2
                continue
            }
            '^--plugin-path=' {
                $pluginPath = $arg -replace '^--plugin-path=', ''
                $i += 1
                continue
            }
            '^--plugin-spec$' {
                $pluginSpec = $Arguments[$i + 1]
                $i += 2
                continue
            }
            '^--plugin-spec=' {
                $pluginSpec = $arg -replace '^--plugin-spec=', ''
                $i += 1
                continue
            }
            '^--skip-build$' {
                $skipBuild = $true
                $i += 1
                continue
            }
            '^--copy$' {
                $copy = $true
                $i += 1
                continue
            }
            '^--proxy-port$' {
                $proxyPort = [int]$Arguments[$i + 1]
                $i += 2
                continue
            }
            '^--proxy-port=' {
                $proxyPort = [int]($arg -replace '^--proxy-port=', '')
                $i += 1
                continue
            }
            '^--startup-timeout-ms$' {
                $startupTimeoutMs = [int]$Arguments[$i + 1]
                $i += 2
                continue
            }
            '^--startup-timeout-ms=' {
                $startupTimeoutMs = [int]($arg -replace '^--startup-timeout-ms=', '')
                $i += 1
                continue
            }
            '^--gateway-provider-id$' {
                $gatewayProviderIds.Add($Arguments[$i + 1])
                $i += 2
                continue
            }
            '^--gateway-provider-id=' {
                $gatewayProviderIds.Add($arg -replace '^--gateway-provider-id=', '')
                $i += 1
                continue
            }
            '^--python-path$' {
                $pythonPath = $Arguments[$i + 1]
                $i += 2
                continue
            }
            '^--python-path=' {
                $pythonPath = $arg -replace '^--python-path=', ''
                $i += 1
                continue
            }
            '^--no-auto-start$' {
                $noAutoStart = $true
                $i += 1
                continue
            }
            '^--no-restart$' {
                $noRestart = $true
                $i += 1
                continue
            }
            '^--verbose$|^-v$' {
                $verbose = $true
                $i += 1
                continue
            }
            default {
                Fail "Unsupported option for 'headroom wrap openclaw': $arg"
            }
        }
    }

    [pscustomobject]@{
        PluginPath = $pluginPath
        PluginSpec = $pluginSpec
        SkipBuild = $skipBuild
        Copy = $copy
        ProxyPort = $proxyPort
        StartupTimeoutMs = $startupTimeoutMs
        GatewayProviderIds = $gatewayProviderIds.ToArray()
        PythonPath = $pythonPath
        NoAutoStart = $noAutoStart
        NoRestart = $noRestart
        Verbose = $verbose
    }
}

function Parse-OpenClawUnwrapArgs {
    param([string[]]$Arguments)

    $noRestart = $false
    $verbose = $false
    $i = 0
    while ($i -lt $Arguments.Count) {
        $arg = $Arguments[$i]
        switch -Regex ($arg) {
            '^--no-restart$' {
                $noRestart = $true
                $i += 1
                continue
            }
            '^--verbose$|^-v$' {
                $verbose = $true
                $i += 1
                continue
            }
            default {
                Fail "Unsupported option for 'headroom unwrap openclaw': $arg"
            }
        }
    }

    [pscustomobject]@{
        NoRestart = $noRestart
        Verbose = $verbose
    }
}

function Invoke-CapturedCommand {
    param(
        [string]$Action,
        [string]$Command,
        [string[]]$Arguments,
        [string]$WorkingDirectory
    )

    $previous = $null
    try {
        if ($WorkingDirectory) {
            $previous = Get-Location
            Set-Location $WorkingDirectory
        }

        $output = (& $Command @Arguments 2>&1 | Out-String).Trim()
        $exitCode = $LASTEXITCODE
    } finally {
        if ($previous) {
            Set-Location $previous
        }
    }

    if ($exitCode -ne 0) {
        if (-not $output) {
            $output = "exit code $exitCode"
        }
        Fail "$Action failed: $output"
    }

    return $output
}

function Get-OpenClawExistingEntryJson {
    $output = (& openclaw config get plugins.entries.headroom 2>$null | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        return $null
    }
    return $output
}

function Invoke-OpenClawPrepareEntryJson {
    param(
        [string]$ExistingEntryJson,
        [pscustomobject]$Parsed
    )

    $dockerArgs = New-Object System.Collections.Generic.List[string]
    $dockerArgs.AddRange(@('run','--rm'))
    $dockerArgs.AddRange((Get-SharedDockerArgs))
    $dockerArgs.Add('--entrypoint')
    $dockerArgs.Add('headroom')
    $dockerArgs.Add($HeadroomImage)
    $dockerArgs.AddRange(@('wrap','openclaw','--prepare-only','--proxy-port',"$($Parsed.ProxyPort)",'--startup-timeout-ms',"$($Parsed.StartupTimeoutMs)"))
    if ($ExistingEntryJson) {
        $dockerArgs.Add('--existing-entry-json')
        $dockerArgs.Add($ExistingEntryJson)
    }
    if ($Parsed.PythonPath) {
        $dockerArgs.Add('--python-path')
        $dockerArgs.Add($Parsed.PythonPath)
    }
    if ($Parsed.NoAutoStart) {
        $dockerArgs.Add('--no-auto-start')
    }
    foreach ($providerId in $Parsed.GatewayProviderIds) {
        $dockerArgs.Add('--gateway-provider-id')
        $dockerArgs.Add($providerId)
    }

    $output = (& docker @dockerArgs 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        Fail "Failed to prepare docker-native OpenClaw config: $output"
    }

    return $output
}

function Invoke-OpenClawPrepareUnwrapEntryJson {
    param([string]$ExistingEntryJson)

    $dockerArgs = New-Object System.Collections.Generic.List[string]
    $dockerArgs.AddRange(@('run','--rm'))
    $dockerArgs.AddRange((Get-SharedDockerArgs))
    $dockerArgs.Add('--entrypoint')
    $dockerArgs.Add('headroom')
    $dockerArgs.Add($HeadroomImage)
    $dockerArgs.AddRange(@('unwrap','openclaw','--prepare-only'))
    if ($ExistingEntryJson) {
        $dockerArgs.Add('--existing-entry-json')
        $dockerArgs.Add($ExistingEntryJson)
    }

    $output = (& docker @dockerArgs 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        Fail "Failed to prepare docker-native OpenClaw unwrap config: $output"
    }

    return $output
}

function Resolve-OpenClawExtensionsDir {
    $configOutput = Invoke-CapturedCommand -Action 'openclaw config file' -Command 'openclaw' -Arguments @('config','file')
    $configPath = ($configOutput -split "`r?`n")[-1].Trim()
    if (-not $configPath) {
        Fail 'Unable to resolve OpenClaw config path.'
    }
    return (Join-Path (Split-Path -Parent $configPath) 'extensions')
}

function Copy-OpenClawPluginIntoExtensions {
    param([string]$PluginPath)

    $distDir = Join-Path $PluginPath 'dist'
    $hookShimDir = Join-Path $PluginPath 'hook-shim'
    if (-not (Test-Path $distDir)) {
        Fail "Plugin dist folder missing at $distDir. Build the plugin first."
    }
    if (-not (Test-Path $hookShimDir)) {
        Fail "Plugin hook-shim folder missing at $hookShimDir. Build the plugin first."
    }

    $extensionsDir = Resolve-OpenClawExtensionsDir
    $targetDir = Join-Path $extensionsDir 'headroom'
    $targetDist = Join-Path $targetDir 'dist'
    $targetHookShim = Join-Path $targetDir 'hook-shim'
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
    if (Test-Path $targetDist) { Remove-Item -Recurse -Force $targetDist }
    if (Test-Path $targetHookShim) { Remove-Item -Recurse -Force $targetHookShim }
    Copy-Item -Recurse -Force $distDir $targetDist
    Copy-Item -Recurse -Force $hookShimDir $targetHookShim

    foreach ($fileName in @('openclaw.plugin.json','package.json','README.md')) {
        $source = Join-Path $PluginPath $fileName
        if (Test-Path $source) {
            Copy-Item -Force $source (Join-Path $targetDir $fileName)
        }
    }

    return $targetDir
}

function Install-OpenClawPlugin {
    param([pscustomobject]$Parsed)

    if ($Parsed.PluginPath) {
        if (-not (Test-Path $Parsed.PluginPath)) {
            Fail "Plugin path not found: $($Parsed.PluginPath)."
        }
        if (-not (Test-Path (Join-Path $Parsed.PluginPath 'package.json'))) {
            Fail "Invalid plugin path (missing package.json): $($Parsed.PluginPath)"
        }
        if (-not (Test-Path (Join-Path $Parsed.PluginPath 'openclaw.plugin.json'))) {
            Fail "Invalid plugin path (missing openclaw.plugin.json): $($Parsed.PluginPath)"
        }
    }

    if ($Parsed.PluginPath -and -not $Parsed.SkipBuild) {
        Require-Command npm
        Write-Host '  Building OpenClaw plugin (npm install + npm run build)...'
        [void](Invoke-CapturedCommand -Action 'npm install' -Command 'npm' -Arguments @('install') -WorkingDirectory $Parsed.PluginPath)
        [void](Invoke-CapturedCommand -Action 'npm run build' -Command 'npm' -Arguments @('run','build') -WorkingDirectory $Parsed.PluginPath)
    }

    if ($Parsed.PluginPath) {
        if ($Parsed.Copy) {
            $arguments = @('plugins','install','--dangerously-force-unsafe-install',$Parsed.PluginPath)
            $workingDirectory = $null
        } else {
            $arguments = @('plugins','install','--dangerously-force-unsafe-install','--link','.')
            $workingDirectory = $Parsed.PluginPath
        }
    } else {
        $arguments = @('plugins','install','--dangerously-force-unsafe-install',$Parsed.PluginSpec)
        $workingDirectory = $null
    }

    $previous = $null
    try {
        if ($workingDirectory) {
            $previous = Get-Location
            Set-Location $workingDirectory
        }
        $installOutput = (& openclaw @arguments 2>&1 | Out-String).Trim()
        $installExitCode = $LASTEXITCODE
    } finally {
        if ($previous) {
            Set-Location $previous
        }
    }

    if ($installExitCode -eq 0) {
        if ($Parsed.Verbose -and $installOutput) {
            Write-Host $installOutput
        }
        return
    }

    $lowerOutput = $installOutput.ToLowerInvariant()
    if ($lowerOutput.Contains('plugin already exists')) {
        Write-Host '  Plugin already installed; continuing with configuration/update steps.'
        return
    }

    if ($Parsed.PluginPath -and -not $Parsed.Copy -and $lowerOutput.Contains('also not a valid hook pack')) {
        Write-Host '  OpenClaw linked-path install bug detected; applying extension-path fallback...'
        $targetDir = Copy-OpenClawPluginIntoExtensions -PluginPath $Parsed.PluginPath
        Write-Host "  Fallback plugin copy completed: $targetDir"
        return
    }

    if (-not $installOutput) {
        $installOutput = "exit code $installExitCode"
    }
    Fail "openclaw plugins install failed: $installOutput"
}

function Restart-OrStartOpenClawGateway {
    $restartOutput = (& openclaw gateway restart 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -eq 0) {
        return [pscustomobject]@{ Action = 'restarted'; Output = $restartOutput }
    }

    $startOutput = Invoke-CapturedCommand -Action 'openclaw gateway start' -Command 'openclaw' -Arguments @('gateway','start')
    return [pscustomobject]@{ Action = 'started'; Output = $startOutput }
}

function Invoke-OpenClawWrap {
    param([string[]]$Arguments)

    Require-Command openclaw
    $parsed = Parse-OpenClawWrapArgs -Arguments $Arguments
    $existingEntryJson = Get-OpenClawExistingEntryJson
    $entryJson = Invoke-OpenClawPrepareEntryJson -ExistingEntryJson $existingEntryJson -Parsed $parsed

    Write-Host ""
    Write-Host "  ╔═══════════════════════════════════════════════╗"
    Write-Host "  ║           HEADROOM WRAP: OPENCLAW             ║"
    Write-Host "  ╚═══════════════════════════════════════════════╝"
    Write-Host ""
    if ($parsed.PluginPath) {
        Write-Host "  Plugin source: local ($($parsed.PluginPath))"
    } else {
        Write-Host "  Plugin source: npm ($($parsed.PluginSpec))"
    }

    Write-Host '  Writing plugin configuration...'
    [void](Invoke-CapturedCommand -Action 'openclaw config set plugins.entries.headroom' -Command 'openclaw' -Arguments @('config','set','plugins.entries.headroom',$entryJson,'--strict-json'))
    Write-Host '  Installing OpenClaw plugin with required unsafe-install flag...'
    Install-OpenClawPlugin -Parsed $parsed
    [void](Invoke-CapturedCommand -Action 'openclaw config set plugins.slots.contextEngine' -Command 'openclaw' -Arguments @('config','set','plugins.slots.contextEngine','"headroom"','--strict-json'))
    [void](Invoke-CapturedCommand -Action 'openclaw config validate' -Command 'openclaw' -Arguments @('config','validate'))

    if ($parsed.NoRestart) {
        Write-Host '  Skipping gateway restart (--no-restart).'
        Write-Host '  Run `openclaw gateway restart` (or `openclaw gateway start`) to apply plugin changes.'
    } else {
        Write-Host '  Applying plugin changes to OpenClaw gateway...'
        $gatewayResult = Restart-OrStartOpenClawGateway
        Write-Host "  Gateway $($gatewayResult.Action)."
        if ($parsed.Verbose -and $gatewayResult.Output) {
            Write-Host $gatewayResult.Output
        }
    }

    $inspectOutput = Invoke-CapturedCommand -Action 'openclaw plugins inspect headroom' -Command 'openclaw' -Arguments @('plugins','inspect','headroom')
    if ($parsed.Verbose -and $inspectOutput) {
        Write-Host $inspectOutput
    }

    Write-Host ""
    Write-Host "✓ OpenClaw is configured to use Headroom context compression."
    Write-Host "  Plugin: headroom"
    Write-Host "  Slot:   plugins.slots.contextEngine = headroom"
    Write-Host ""
}

function Invoke-OpenClawUnwrap {
    param([string[]]$Arguments)

    Require-Command openclaw
    $parsed = Parse-OpenClawUnwrapArgs -Arguments $Arguments
    $existingEntryJson = Get-OpenClawExistingEntryJson
    $entryJson = Invoke-OpenClawPrepareUnwrapEntryJson -ExistingEntryJson $existingEntryJson

    Write-Host ""
    Write-Host "  ╔═══════════════════════════════════════════════╗"
    Write-Host "  ║          HEADROOM UNWRAP: OPENCLAW            ║"
    Write-Host "  ╚═══════════════════════════════════════════════╝"
    Write-Host ""
    Write-Host '  Disabling Headroom plugin and removing engine mapping...'

    [void](Invoke-CapturedCommand -Action 'openclaw config set plugins.entries.headroom' -Command 'openclaw' -Arguments @('config','set','plugins.entries.headroom',$entryJson,'--strict-json'))
    [void](Invoke-CapturedCommand -Action 'openclaw config set plugins.slots.contextEngine' -Command 'openclaw' -Arguments @('config','set','plugins.slots.contextEngine','"legacy"','--strict-json'))
    [void](Invoke-CapturedCommand -Action 'openclaw config validate' -Command 'openclaw' -Arguments @('config','validate'))

    if ($parsed.NoRestart) {
        Write-Host '  Skipping gateway restart (--no-restart).'
        Write-Host '  Run `openclaw gateway restart` (or `openclaw gateway start`) to apply unwrap changes.'
    } else {
        Write-Host '  Applying unwrap changes to OpenClaw gateway...'
        $gatewayResult = Restart-OrStartOpenClawGateway
        Write-Host "  Gateway $($gatewayResult.Action)."
        if ($parsed.Verbose -and $gatewayResult.Output) {
            Write-Host $gatewayResult.Output
        }
    }

    if ($parsed.Verbose) {
        $inspectOutput = Invoke-CapturedCommand -Action 'openclaw plugins inspect headroom' -Command 'openclaw' -Arguments @('plugins','inspect','headroom')
        if ($inspectOutput) {
            Write-Host $inspectOutput
        }
    }

    Write-Host ""
    Write-Host "✓ OpenClaw Headroom wrap removed."
    Write-Host "  Plugin: headroom (installed, disabled)"
    Write-Host "  Slot:   plugins.slots.contextEngine = legacy"
    Write-Host ""
}

function Parse-WrapArgs {
    param([string[]]$Arguments)

    $known = New-Object System.Collections.Generic.List[string]
    $host = New-Object System.Collections.Generic.List[string]
    $port = 8787
    $noRtk = $false
    $noProxy = $false
    $learn = $false
    $backend = $null
    $anyllm = $null
    $region = $null

    $i = 0
    while ($i -lt $Arguments.Count) {
        $arg = $Arguments[$i]
        switch -Regex ($arg) {
            '^--$' {
                for ($j = $i + 1; $j -lt $Arguments.Count; $j++) {
                    $host.Add($Arguments[$j])
                }
                $i = $Arguments.Count
                continue
            }
            '^--port$|^-p$' {
                $port = [int]$Arguments[$i + 1]
                $known.Add($arg)
                $known.Add($Arguments[$i + 1])
                $i += 2
                continue
            }
            '^--port=' {
                $port = [int]($arg -replace '^--port=', '')
                $known.Add($arg)
                $i += 1
                continue
            }
            '^--no-rtk$' {
                $noRtk = $true
                $known.Add($arg)
                $i += 1
                continue
            }
            '^--no-proxy$' {
                $noProxy = $true
                $known.Add($arg)
                $i += 1
                continue
            }
            '^--learn$' {
                $learn = $true
                $known.Add($arg)
                $i += 1
                continue
            }
            '^--verbose$|^-v$' {
                $known.Add($arg)
                $i += 1
                continue
            }
            '^--backend$' {
                $backend = $Arguments[$i + 1]
                $known.Add($arg)
                $known.Add($Arguments[$i + 1])
                $i += 2
                continue
            }
            '^--backend=' {
                $backend = $arg -replace '^--backend=', ''
                $known.Add($arg)
                $i += 1
                continue
            }
            '^--anyllm-provider$' {
                $anyllm = $Arguments[$i + 1]
                $known.Add($arg)
                $known.Add($Arguments[$i + 1])
                $i += 2
                continue
            }
            '^--anyllm-provider=' {
                $anyllm = $arg -replace '^--anyllm-provider=', ''
                $known.Add($arg)
                $i += 1
                continue
            }
            '^--region$' {
                $region = $Arguments[$i + 1]
                $known.Add($arg)
                $known.Add($Arguments[$i + 1])
                $i += 2
                continue
            }
            '^--region=' {
                $region = $arg -replace '^--region=', ''
                $known.Add($arg)
                $i += 1
                continue
            }
            default {
                for ($j = $i; $j -lt $Arguments.Count; $j++) {
                    $host.Add($Arguments[$j])
                }
                $i = $Arguments.Count
            }
        }
    }

    [pscustomobject]@{
        KnownArgs = $known.ToArray()
        HostArgs = $host.ToArray()
        Port = $port
        NoRtk = $noRtk
        NoProxy = $noProxy
        Learn = $learn
        Backend = $backend
        Anyllm = $anyllm
        Region = $region
    }
}

function Invoke-PrepareOnly {
    param(
        [string]$Tool,
        [string[]]$KnownArgs
    )

    $dockerArgs = New-Object System.Collections.Generic.List[string]
    $dockerArgs.AddRange(@('run','--rm','-it'))
    $dockerArgs.AddRange((Get-SharedDockerArgs))
    $dockerArgs.Add('--env')
    $dockerArgs.Add("HEADROOM_RTK_TARGET=$(Get-RtkTarget)")
    $dockerArgs.Add('--entrypoint')
    $dockerArgs.Add('headroom')
    $dockerArgs.Add($HeadroomImage)
    $dockerArgs.AddRange(@('wrap',$Tool,'--prepare-only'))
    foreach ($arg in $KnownArgs) {
        $dockerArgs.Add($arg)
    }

    & docker @dockerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to prepare docker-native wrap for $Tool"
    }
}

Require-Command docker

if ($args.Count -eq 0) {
    Invoke-HeadroomDocker -Arguments @('--help')
    exit 0
}

switch ($args[0]) {
    'wrap' {
        if ($args.Count -lt 2) {
            Fail 'Usage: headroom wrap <claude|codex|aider|cursor|openclaw> [...]'
        }

        $tool = $args[1]
        $wrapArgs = if ($args.Count -gt 2) { $args[2..($args.Count - 1)] } else { @() }

        if ($tool -eq 'openclaw') {
            if (Test-HelpFlag -Arguments $wrapArgs) {
                $helpArgs = @('wrap','openclaw') + $wrapArgs
                Invoke-HeadroomDocker -Arguments $helpArgs
                exit 0
            }

            Invoke-OpenClawWrap -Arguments $wrapArgs
            exit 0
        }

        $parsed = Parse-WrapArgs -Arguments $wrapArgs
        $proxyArgs = New-Object System.Collections.Generic.List[string]
        if ($parsed.Learn) { $proxyArgs.Add('--learn') }
        if ($parsed.Backend) { $proxyArgs.AddRange(@('--backend', $parsed.Backend)) }
        if ($parsed.Anyllm) { $proxyArgs.AddRange(@('--anyllm-provider', $parsed.Anyllm)) }
        if ($parsed.Region) { $proxyArgs.AddRange(@('--region', $parsed.Region)) }

        switch ($tool) {
            'claude' { }
            'codex' { }
            'aider' { }
            'cursor' { }
            default { Fail "Unsupported wrap target: $tool" }
        }

        $containerName = $null
        try {
            if (-not $parsed.NoProxy) {
                $containerName = Start-ProxyContainer -Port $parsed.Port -ProxyArgs $proxyArgs.ToArray()
            }

            $prepareArgs = New-Object System.Collections.Generic.List[string]
            foreach ($arg in $parsed.KnownArgs) {
                $prepareArgs.Add($arg)
            }
            if (-not $parsed.NoProxy) {
                $prepareArgs.Add('--no-proxy')
            }
            Invoke-PrepareOnly -Tool $tool -KnownArgs $prepareArgs.ToArray()

            switch ($tool) {
                'claude' {
                    if (-not $parsed.NoRtk) { Invoke-ClaudeRtkInit }
                    $exitCode = Invoke-WithTemporaryEnv -Environment @{ ANTHROPIC_BASE_URL = "http://127.0.0.1:$($parsed.Port)" } -Command 'claude' -Arguments $parsed.HostArgs
                    exit $exitCode
                }
                'codex' {
                    $exitCode = Invoke-WithTemporaryEnv -Environment @{ OPENAI_BASE_URL = "http://127.0.0.1:$($parsed.Port)/v1" } -Command 'codex' -Arguments $parsed.HostArgs
                    exit $exitCode
                }
                'aider' {
                    $exitCode = Invoke-WithTemporaryEnv -Environment @{
                        OPENAI_API_BASE = "http://127.0.0.1:$($parsed.Port)/v1"
                        ANTHROPIC_BASE_URL = "http://127.0.0.1:$($parsed.Port)"
                    } -Command 'aider' -Arguments $parsed.HostArgs
                    exit $exitCode
                }
                'cursor' {
                    Write-Host "Headroom proxy is running for Cursor."
                    Write-Host ""
                    Write-Host "OpenAI base URL:     http://127.0.0.1:$($parsed.Port)/v1"
                    Write-Host "Anthropic base URL:  http://127.0.0.1:$($parsed.Port)"
                    Write-Host ""
                    Write-Host "Press Ctrl+C to stop the proxy."
                    while ($true) { Start-Sleep -Seconds 1 }
                }
            }
        } finally {
            Stop-ProxyContainer -ContainerName $containerName
        }
    }
    'unwrap' {
        if ($args.Count -ge 2 -and $args[1] -eq 'openclaw') {
            $unwrapArgs = if ($args.Count -gt 2) { $args[2..($args.Count - 1)] } else { @() }
            if (Test-HelpFlag -Arguments $unwrapArgs) {
                $helpArgs = @('unwrap','openclaw') + $unwrapArgs
                Invoke-HeadroomDocker -Arguments $helpArgs
                exit 0
            }

            Invoke-OpenClawUnwrap -Arguments $unwrapArgs
            exit 0
        }
        Invoke-HeadroomDocker -Arguments $args
    }
    'proxy' {
        $port = 8787
        $forwardArgs = New-Object System.Collections.Generic.List[string]
        foreach ($arg in $args) { $forwardArgs.Add($arg) }
        for ($i = 1; $i -lt $args.Count; $i++) {
            if ($args[$i] -eq '--port' -or $args[$i] -eq '-p') {
                $port = [int]$args[$i + 1]
                break
            }
            if ($args[$i] -match '^--port=') {
                $port = [int]($args[$i] -replace '^--port=', '')
                break
            }
        }

        $dockerArgs = New-Object System.Collections.Generic.List[string]
        $dockerArgs.AddRange(@('run','--rm','-it','-p',"$port`:$port"))
        $dockerArgs.AddRange((Get-SharedDockerArgs))
        $dockerArgs.Add('--entrypoint')
        $dockerArgs.Add('headroom')
        $dockerArgs.Add($HeadroomImage)
        foreach ($arg in $forwardArgs) {
            $dockerArgs.Add($arg)
        }

        & docker @dockerArgs
        exit $LASTEXITCODE
    }
    default {
        Invoke-HeadroomDocker -Arguments $args
    }
}
'@

    $cmdWrapper = @'
@echo off
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0headroom.ps1" %*
'@

    Set-Content -Path $wrapperPath -Value $wrapper
    Set-Content -Path $cmdPath -Value $cmdWrapper
}

Require-Command docker
docker version | Out-Null

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Write-Wrapper -TargetDir $InstallDir
Ensure-PathEntry -PathEntry $InstallDir
Ensure-ProfileBlock -PathEntry $InstallDir

Write-Info "Pulling $ImageDefault"
docker pull $ImageDefault | Out-Null

Write-Host ''
Write-Host 'Headroom Docker-native install complete.'
Write-Host ''
Write-Host "Installed wrappers:"
Write-Host "  $InstallDir\headroom.ps1"
Write-Host "  $InstallDir\headroom.cmd"
Write-Host ''
Write-Host 'Next steps:'
Write-Host "  1. Restart PowerShell"
Write-Host "  2. Try: headroom proxy"
Write-Host "  3. Docs: https://github.com/chopratejas/headroom/blob/main/docs/docker-install.md"
