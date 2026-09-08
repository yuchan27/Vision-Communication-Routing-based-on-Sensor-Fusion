# AppYOLO usage note

The canonical, maintained documentation is the repository root [English README](../../readme.md) | [Traditional Chinese README](../../README.zh-TW.md). It contains the installation path, PowerShell commands, dashboard workflow, API fields, dataset layout, VCN package, temperature-source contract, and the recommended radiometric thermal-sensor roadmap.

Start from `C:\Code\FIRE\AppYOLO` so the existing inference modules resolve their model and output paths consistently. The project baseline is Python 3.13 and the `start.ps1` script keeps all backend modules on the same interpreter:

```powershell
Set-Location C:\Code\FIRE\AppYOLO
.\start.ps1 -Install
```

The dashboard is available at `http://127.0.0.1:8000`. Optional image/video form field:

```text
sensor_temperature_celsius=<calibrated scene reading in °C>
```

When omitted, the backend uses the RGB estimate or ambient fallback and reports the source explicitly. Host CPU temperature is diagnostic only.
