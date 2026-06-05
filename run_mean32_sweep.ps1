param(
    [string]$Seeds = "7,13,21,42,77",
    [string]$GroupSizes = "4,8,16",
    [int]$Steps = 200,
    [int]$PpoEpochs = 2,
    [double]$PriorWeight = 0.05,
    [double]$LearningRate = 0.005,
    [string]$Device = "cuda",
    [string]$PythonExe = "D:\anaconda\envs\dw_env\python.exe",
    [string]$SweepRoot = "runs\mean32_seed_group_sweep",
    [switch]$Force,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$TrainScript = Join-Path $ProjectRoot "train_rl_fwi.py"
$SweepRootAbs = Join-Path $ProjectRoot $SweepRoot
$SummaryPath = Join-Path $SweepRootAbs "summary.csv"

function Convert-ToIntList {
    param(
        [string]$Value,
        [string]$Name
    )
    $items = @(
        $Value -split "," |
            ForEach-Object { $_.Trim() } |
            Where-Object { $_ -ne "" } |
            ForEach-Object { [int]$_ }
    )
    if ($items.Count -eq 0) {
        throw "$Name must contain at least one integer"
    }
    return $items
}

function Ensure-Directory {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Get-MetricSummary {
    param(
        [string]$MetricsPath,
        [int]$Seed,
        [int]$GroupSize,
        [string]$RunDir,
        [int]$ExitCode,
        [double]$WallSeconds,
        [string]$Status
    )

    $result = [ordered]@{
        seed = $Seed
        group_size = $GroupSize
        status = $Status
        exit_code = $ExitCode
        wall_seconds = [math]::Round($WallSeconds, 1)
        steps_recorded = 0
        best_mae = ""
        best_mae_step = ""
        final_mae = ""
        final_reward_l1 = ""
        final_reward_l2 = ""
        ratio_mean_min = ""
        ratio_mean_max = ""
        ratio_std_max = ""
        clip_frac_max = ""
        entropy_first = ""
        entropy_last = ""
        run_dir = $RunDir
    }

    if (-not (Test-Path -LiteralPath $MetricsPath)) {
        return [pscustomobject]$result
    }

    $rows = @(Import-Csv -LiteralPath $MetricsPath)
    if ($rows.Count -eq 0) {
        return [pscustomobject]$result
    }

    $final = $rows[-1]
    $hasNewMaeColumns = (
        ($rows[0].PSObject.Properties.Name -contains "best_mae_global") -and
        ($rows[0].PSObject.Properties.Name -contains "mae_oracle_best") -and
        ($rows[0].PSObject.Properties.Name -contains "mae_reward_best")
    )
    if ($hasNewMaeColumns) {
        $best = $rows | Sort-Object { [double]$_.best_mae_global } | Select-Object -First 1
        $bestMae = [double]$best.best_mae_global
        $bestStep = [int]$best.step
        $finalMae = [double]$final.mae_reward_best
    }
    else {
        $best = $rows | Sort-Object { [double]$_.mae_best } | Select-Object -First 1
        $bestMae = [double]$best.mae_best
        $bestStep = [int]$best.step
        $finalMae = [double]$final.mae_best
    }
    $ratioMeans = @($rows | ForEach-Object { [double]$_.ratio_mean })
    $ratioStds = @($rows | ForEach-Object { [double]$_.ratio_std })
    $clips = @($rows | ForEach-Object { [double]$_.clip_frac })

    $result.steps_recorded = $rows.Count
    $result.best_mae = [math]::Round($bestMae, 4)
    $result.best_mae_step = $bestStep
    $result.final_mae = [math]::Round($finalMae, 4)
    $result.final_reward_l1 = [math]::Round([double]$final.reward_l1_mean, 4)
    $result.final_reward_l2 = [math]::Round([double]$final.reward_l2_mean, 4)
    $result.ratio_mean_min = [math]::Round(($ratioMeans | Measure-Object -Minimum).Minimum, 6)
    $result.ratio_mean_max = [math]::Round(($ratioMeans | Measure-Object -Maximum).Maximum, 6)
    $result.ratio_std_max = [math]::Round(($ratioStds | Measure-Object -Maximum).Maximum, 6)
    $result.clip_frac_max = [math]::Round(($clips | Measure-Object -Maximum).Maximum, 6)
    $result.entropy_first = [math]::Round([double]$rows[0].entropy, 6)
    $result.entropy_last = [math]::Round([double]$final.entropy, 6)

    return [pscustomobject]$result
}

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $TrainScript)) {
    throw "Training script not found: $TrainScript"
}

Ensure-Directory -Path $SweepRootAbs

$SeedList = Convert-ToIntList -Value $Seeds -Name "Seeds"
$GroupSizeList = Convert-ToIntList -Value $GroupSizes -Name "GroupSizes"

$summaries = @()
$totalRuns = $SeedList.Count * $GroupSizeList.Count
$runIndex = 0

foreach ($groupSize in $GroupSizeList) {
    foreach ($seed in $SeedList) {
        $runIndex += 1
        $runName = "g{0}_seed{1}" -f $groupSize, $seed
        $runDir = Join-Path $SweepRootAbs $runName
        $metricsPath = Join-Path $runDir "metrics.csv"
        $stdoutPath = Join-Path $runDir "stdout.log"
        $stderrPath = Join-Path $runDir "stderr.log"
        $commandPath = Join-Path $runDir "command.txt"

        Ensure-Directory -Path $runDir

        $argList = @(
            $TrainScript,
            "--policy_type", "mean",
            "--steps", "$Steps",
            "--group_size", "$groupSize",
            "--ppo_epochs", "$PpoEpochs",
            "--reward_prior_weight", "$PriorWeight",
            "--lr", "$LearningRate",
            "--seed", "$seed",
            "--device", "$Device",
            "--out_dir", $runDir
        )

        $displayCommand = '"{0}" {1}' -f $PythonExe, ($argList -join " ")
        Set-Content -LiteralPath $commandPath -Value $displayCommand -Encoding UTF8

        if ($DryRun) {
            Write-Host "[DRY RUN $runIndex/$totalRuns] $displayCommand"
            $summaries += Get-MetricSummary -MetricsPath $metricsPath -Seed $seed -GroupSize $groupSize -RunDir $runDir -ExitCode -1 -WallSeconds 0 -Status "dry_run"
            continue
        }

        if ((Test-Path -LiteralPath $metricsPath) -and (-not $Force)) {
            Write-Host "[SKIP $runIndex/$totalRuns] Existing metrics: $metricsPath"
            $summaries += Get-MetricSummary -MetricsPath $metricsPath -Seed $seed -GroupSize $groupSize -RunDir $runDir -ExitCode 0 -WallSeconds 0 -Status "skipped_existing"
            continue
        }

        Write-Host "[RUN $runIndex/$totalRuns] group_size=$groupSize seed=$seed"
        Write-Host "  out_dir: $runDir"

        $startTime = Get-Date
        Push-Location $ProjectRoot
        try {
            & $PythonExe @argList > $stdoutPath 2> $stderrPath
            $exitCode = $LASTEXITCODE
        }
        finally {
            Pop-Location
        }
        $wallSeconds = ((Get-Date) - $startTime).TotalSeconds

        $status = if ($exitCode -eq 0) { "ok" } else { "failed" }
        Write-Host "  status=$status exit_code=$exitCode wall_seconds=$([math]::Round($wallSeconds, 1))"

        $summaries += Get-MetricSummary -MetricsPath $metricsPath -Seed $seed -GroupSize $groupSize -RunDir $runDir -ExitCode $exitCode -WallSeconds $wallSeconds -Status $status
    }
}

$summaries | Export-Csv -LiteralPath $SummaryPath -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "Sweep complete."
Write-Host "Summary: $SummaryPath"
Write-Host ""
$summaries |
    Sort-Object group_size, seed |
    Format-Table seed, group_size, status, best_mae, best_mae_step, ratio_mean_min, ratio_mean_max, clip_frac_max, entropy_first, entropy_last -AutoSize
