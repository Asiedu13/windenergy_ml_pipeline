# Supervisor for the v6 regional ERA5 archive download.
#
# Registered as a scheduled task that REPEATS every 30 minutes. It MUST return in
# seconds. The task is MultipleInstances=IgnoreNew, so an instance that blocks for
# the duration of the download causes every later firing to be refused — which
# silently disables the stall check, the single mode that produced the four-day
# outage. The previous version of this script blocked on `& cmd.exe` and had
# exactly that defect: its STALLED branch was unreachable while the download
# lived, i.e. precisely when it was needed.
#
# So: the worker is launched DETACHED and the supervisor exits immediately.
#
# Detach is via WMI Win32_Process.Create rather than Start-Process. A
# Start-Process child sits inside the task's process tree, which Task Scheduler
# may tear down when the task completes; a WMI-created process is parented to
# WmiPrvSE and outlives its launcher by construction.
#
# Two failure modes, and "start if not running" only covers the first:
#   DEAD    no worker -> start one
#   STALLED worker alive but making no progress. Happens when cdsapi sits in its
#           own retry loop (up to 500 attempts x 120 s = 16.7 h on a DNS outage),
#           during which the process looks perfectly healthy and our own
#           _retrieve_with_retry never fires because cdsapi never raises
#           -> kill it, then start one
#
# $StallMinutes is 120, comfortably above the slowest observed rate (~37
# min/request), so a merely slow request is never mistaken for a stalled one.
#
# Five deaths so far, each a different mechanism — worth keeping, because every
# one of them looked like the last one had been fixed:
#   1  Start-Process from the agent's shell -> killed with the agent's process tree
#   2  WMI Create issued from a SANDBOXED agent call -> instant "^C" in the log
#   3  machine sleep/shutdown
#   4  machine sleep again -> went unnoticed for FOUR DAYS
#   5  Start-ScheduledTask issued from a SANDBOXED agent call -> the task exited
#      in under a second with 0xC000013A (STATUS_CONTROL_C_EXIT), before python
#      had even imported. Measured 2026-10-06: the same call with the sandbox
#      disabled left the worker running. The sandbox kills processes CREATED
#      during a sandboxed call; it does NOT touch a process already running.
#      => NEVER start this task, or any worker, from a sandboxed tool call.
#         Deaths 2 and 5 were the same root cause wearing different clothes.
#
# Pass order is deliberate (docs/13 §4c): the forecast window 2024-03..2025-12
# first, because at the measured rate the full archive overruns the pre-registered
# time budget, and the 24 h ladder is what tests thesis §4.9.3. The pass is chosen
# by COUNTING FILES, not by sequencing two blocking calls, so a restart at any
# point resumes in the right place without the supervisor having to stay alive to
# remember where it was.
#
# NEVER run any other CDS request alongside this. CDS limits concurrent jobs per
# account; a 3-month probe run in parallel starved this to 3 requests in 13.4 h.

param([switch]$DryRun)

$repo     = 'c:\Users\Dell\Documents\creations\personal\finalYearProject\windenergy_ml_pipeline'
$python   = 'C:\Users\Dell\AppData\Local\Programs\Python\Python313\python.exe'
$log      = Join-Path $repo 'outputs\logs\era5gh_archive.log'
$pattern  = Join-Path $repo 'data\raw\era5gh_*.nc'

$StallMinutes = 120
$Families     = @('core', 'extra', 'pressure')
$Pass1First   = [datetime]'2024-03-01'
$Pass1Last    = [datetime]'2025-12-01'

Set-Location $repo

function Write-Log($msg) {
    $line = "=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg ==="
    if ($DryRun) { Write-Output "[dry-run] $line"; return }
    Add-Content -Path $log -Encoding utf8 -Value $line
}

function Get-Worker {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like '*multisite*' }
}

# Which months of the forecast window are still missing. Returned rather than a
# bare bool so the log records how much is left, which is the number anyone
# actually wants when they check on this.
function Get-Pass1Missing {
    $missing = 0
    $m = $Pass1First
    while ($m -le $Pass1Last) {
        foreach ($f in $Families) {
            $p = Join-Path $repo ('data\raw\era5gh_{0}_{1}.nc' -f $f, $m.ToString('yyyy-MM'))
            if (-not (Test-Path $p)) { $missing++ }
        }
        $m = $m.AddMonths(1)
    }
    return $missing
}

$worker = Get-Worker

if ($worker) {
    # Alive — but is it making progress?
    #
    # Staleness cannot be measured from file mtime alone. A process that has just
    # started has not written a file yet, so the newest mtime can be days old
    # while the process is perfectly healthy — which would make this supervisor
    # kill the very thing it exists to protect, every cycle, forever. So the clock
    # runs from whichever is LATER: the newest file, or the worker's own start
    # time. A fresh worker therefore gets a full grace period.
    #
    # The glob is built with Join-Path into $pattern rather than inline, because an
    # escaping slip that silently breaks it makes $newest always null and degrades
    # this to "kill any process older than $StallMinutes" — which looks like it
    # works and is the opposite of what is wanted. That happened once already.
    $newest = Get-ChildItem $pattern -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $newest) {
        Write-Log "SUPERVISOR: WARNING no files match $pattern - check the path"
    }
    $marks = @($worker | ForEach-Object { $_.CreationDate })
    if ($newest) { $marks += $newest.LastWriteTime }
    $since = ($marks | Measure-Object -Maximum).Maximum
    $idle  = [int]((Get-Date) - $since).TotalMinutes

    if ($idle -lt $StallMinutes) {
        if ($DryRun) {
            Write-Output "[dry-run] HEALTHY pid=$($worker.ProcessId) idle=$idle min (limit $StallMinutes) -> exit 0"
        }
        exit 0                      # healthy, nothing to say
    }

    Write-Log "SUPERVISOR: stalled - no progress for $idle min (limit $StallMinutes), killing PID $($worker.ProcessId)"
    if (-not $DryRun) {
        $worker | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Start-Sleep -Seconds 5
    }
}

$missing = Get-Pass1Missing
if ($missing -gt 0) {
    $fetchArgs = '--start 2024-03-01 --end 2025-12-31'
    $label     = "pass 1 (forecast window 2024-03..2025-12, $missing files missing)"
} else {
    $fetchArgs = ''
    $label     = 'pass 2 (full archive 2010-01..2025-12)'
}

# Redirection goes through cmd.exe: PowerShell's `*>>` transcodes to UTF-16 and
# corrupts a log python writes as UTF-8. cmd appends raw bytes. `-u` replaces the
# PYTHONUNBUFFERED env var, which a WMI-created process would not inherit.
$cmdline = 'cmd.exe /c ""{0}" -u -m src.multisite fetch {1} >> "{2}" 2>&1"' -f $python, $fetchArgs, $log

if ($DryRun) {
    Write-Output "[dry-run] would launch $label"
    Write-Output "[dry-run] $cmdline"
    exit 0
}

$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine      = $cmdline
    CurrentDirectory = $repo
}

if ($r.ReturnValue -eq 0) {
    Write-Log "$label launched detached, pid $($r.ProcessId)"
} else {
    Write-Log "SUPERVISOR: FAILED to launch $label - Win32_Process.Create returned $($r.ReturnValue)"
}
