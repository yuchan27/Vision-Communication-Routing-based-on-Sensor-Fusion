const dom = {
  btnHealth: document.getElementById("btnHealth"),
  healthBadge: document.getElementById("healthBadge"),
  modelMeta: document.getElementById("modelMeta"),
  uploadForm: document.getElementById("uploadForm"),
  imageInput: document.getElementById("imageInput"),
  imageSensorTempInput: document.getElementById("imageSensorTempInput"),
  uploadBtn: document.getElementById("uploadBtn"),
  uploadStatus: document.getElementById("uploadStatus"),
  videoForm: document.getElementById("videoForm"),
  videoInput: document.getElementById("videoInput"),
  videoSensorTempInput: document.getElementById("videoSensorTempInput"),
  videoBtn: document.getElementById("videoBtn"),
  videoStatus: document.getElementById("videoStatus"),
  btnRunVCN: document.getElementById("btnRunVCN"),
  pipelineStatus: document.getElementById("pipelineStatus"),
  sourceInput: document.getElementById("sourceInput"),
  confInput: document.getElementById("confInput"),
  frameSkipInput: document.getElementById("frameSkipInput"),
  maxWidthInput: document.getElementById("maxWidthInput"),
  btnLiveStart: document.getElementById("btnLiveStart"),
  btnLiveStop: document.getElementById("btnLiveStop"),
  streamBadge: document.getElementById("streamBadge"),
  streamStatus: document.getElementById("streamStatus"),
  streamLogPath: document.getElementById("streamLogPath"),
  mediaLiveLayout: document.getElementById("mediaLiveLayout"),
  liveSidePanel: document.getElementById("liveSidePanel"),
  liveFrame: document.getElementById("liveFrame"),
  frameFallback: document.getElementById("frameFallback"),
  btnPreviewZoom: document.getElementById("btnPreviewZoom"),
  riskValue: document.getElementById("riskValue"),
  riskState: document.getElementById("riskState"),
  tempValue: document.getElementById("tempValue"),
  tempSourceValue: document.getElementById("tempSourceValue"),
  fpsValue: document.getElementById("fpsValue"),
  detCountValue: document.getElementById("detCountValue"),
  decisionAction: document.getElementById("decisionAction"),
  topConfValue: document.getElementById("topConfValue"),
  topClassValue: document.getElementById("topClassValue"),
  detectionBody: document.getElementById("detectionBody"),
  sideTempValue: document.getElementById("sideTempValue"),
  sideTempSourceValue: document.getElementById("sideTempSourceValue"),
  sideRiskValue: document.getElementById("sideRiskValue"),
  sideTopConfValue: document.getElementById("sideTopConfValue"),
  sideDetCountValue: document.getElementById("sideDetCountValue"),
  sideTopClassValue: document.getElementById("sideTopClassValue"),
  sideActionValue: document.getElementById("sideActionValue"),
  sideDetectionList: document.getElementById("sideDetectionList"),
  chart: document.getElementById("riskChart"),
  btnRefreshArtifacts: document.getElementById("btnRefreshArtifacts"),
  btnCleanupArtifacts: document.getElementById("btnCleanupArtifacts"),
  artifactLimit: document.getElementById("artifactLimit"),
  artifactSummary: document.getElementById("artifactSummary"),
  artifactList: document.getElementById("artifactList"),
  mediaModal: document.getElementById("mediaModal"),
  mediaModalTitle: document.getElementById("mediaModalTitle"),
  mediaModalImage: document.getElementById("mediaModalImage"),
  mediaModalDownload: document.getElementById("mediaModalDownload"),
  btnMediaClose: document.getElementById("btnMediaClose"),
};

let liveEventSource = null;
let frameTimer = null;
let riskChart = null;
let chartLabels = [];
let chartRisk = [];
let chartTemp = [];
let previousFocusedElement = null;
const MAX_CHART_POINTS = 120;

function setBadge(node, text, type) {
  if (!node) {
    return;
  }
  node.textContent = text;
  node.classList.remove("neutral", "good", "alert");
  node.classList.add(type);
}

function setLiveSnapshotVisible(visible) {
  if (!dom.liveSidePanel || !dom.mediaLiveLayout) {
    return;
  }

  dom.liveSidePanel.classList.toggle("hidden", !visible);
  dom.mediaLiveLayout.classList.toggle("snapshot-hidden", !visible);
}

function bindSafe(node, eventName, handler) {
  if (!node) {
    return;
  }
  node.addEventListener(eventName, handler);
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "0";
  }
  return Number(value).toFixed(digits);
}

function formatFileSize(bytes) {
  const value = Number(bytes || 0);
  if (value <= 0) {
    return "0 B";
  }
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }
  return `${(value / (1024 * 1024)).toFixed(2)} MB`;
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatTimeLabel(isoTimestamp) {
  if (!isoTimestamp) {
    return "--:--:--";
  }
  const date = new Date(isoTimestamp);
  if (Number.isNaN(date.getTime())) {
    return String(isoTimestamp);
  }
  return date.toLocaleTimeString("en-US", { hour12: false });
}

function drawFallbackChart(chart) {
  const canvas = chart.canvas;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(Math.floor(rect.width), 320);
  const height = Math.max(Math.floor(rect.height), 220);
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);

  const context = canvas.getContext("2d");
  if (!context) {
    return;
  }
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);

  const colors = {
    grid: "rgba(154, 171, 180, 0.16)",
    text: "#9aabb4",
    risk: "#ff745f",
    temperature: "#63e2d1",
  };
  const padding = { top: 16, right: 48, bottom: 28, left: 42 };
  const plotWidth = Math.max(width - padding.left - padding.right, 1);
  const plotHeight = Math.max(height - padding.top - padding.bottom, 1);
  const labels = chart.data.labels || [];
  const riskData = chart.data.datasets[0].data || [];
  const temperatureData = chart.data.datasets[1].data || [];
  const tempMin = Number(chart.options.scales.yTemp.min || 0);
  const tempMax = Math.max(Number(chart.options.scales.yTemp.max || 80), tempMin + 1);

  context.font = '11px "IBM Plex Mono", Consolas, monospace';
  context.lineWidth = 1;
  context.strokeStyle = colors.grid;
  context.fillStyle = colors.text;
  context.textBaseline = "middle";

  for (let step = 0; step <= 4; step += 1) {
    const ratio = step / 4;
    const y = padding.top + plotHeight * ratio;
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(width - padding.right, y);
    context.stroke();
    context.fillText((1 - ratio).toFixed(1), 8, y);
    const temperatureLabel = tempMax - ((tempMax - tempMin) * ratio);
    context.fillText(`${temperatureLabel.toFixed(0)}°`, width - padding.right + 8, y);
  }

  const drawLine = (values, color, mapValue) => {
    const finiteValues = values.map((value) => Number(value));
    if (finiteValues.length === 0) {
      return;
    }
    context.beginPath();
    finiteValues.forEach((value, index) => {
      const x = padding.left + (labels.length <= 1 ? plotWidth / 2 : (index / (labels.length - 1)) * plotWidth);
      const y = padding.top + mapValue(value) * plotHeight;
      if (index === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    });
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.stroke();
  };

  drawLine(riskData, colors.risk, (value) => 1 - Math.max(0, Math.min(1, value)));
  drawLine(temperatureData, colors.temperature, (value) => 1 - Math.max(0, Math.min(1, (value - tempMin) / (tempMax - tempMin))));

  if (labels.length === 0) {
    context.fillStyle = colors.text;
    context.fillText("Waiting for telemetry...", padding.left, padding.top + plotHeight / 2);
  } else {
    context.fillStyle = colors.text;
    context.textBaseline = "alphabetic";
    context.fillText(labels[0], padding.left, height - 8);
    if (labels.length > 1) {
      const lastLabel = labels[labels.length - 1];
      const labelWidth = context.measureText(lastLabel).width;
      context.fillText(lastLabel, width - padding.right - labelWidth, height - 8);
    }
  }
}

function createFallbackChart(canvas, maxPoints) {
  const chart = {
    canvas,
    __maxPoints: maxPoints,
    data: {
      labels: [],
      datasets: [{ data: [] }, { data: [] }],
    },
    options: { scales: { yTemp: { min: 0, max: 80 } } },
    update() {
      drawFallbackChart(this);
    },
  };
  chart.update();
  return chart;
}

function createDualAxisChart(canvas, maxPoints = MAX_CHART_POINTS) {
  if (!canvas) {
    return null;
  }
  if (typeof Chart === "undefined") {
    return createFallbackChart(canvas, maxPoints);
  }

  const chart = new Chart(canvas, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Risk score",
          data: [],
          borderColor: "#d44e1a",
          backgroundColor: "rgba(212, 78, 26, 0.2)",
          borderWidth: 2,
          tension: 0.35,
          yAxisID: "yRisk",
          pointRadius: 0,
        },
        {
          label: "Scene temperature °C",
          data: [],
          borderColor: "#006f6b",
          backgroundColor: "rgba(0, 111, 107, 0.15)",
          borderWidth: 2,
          tension: 0.3,
          yAxisID: "yTemp",
          pointRadius: 0,
        },
      ],
    },
    options: {
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false,
      },
      plugins: {
        legend: {
          labels: {
            usePointStyle: true,
          },
        },
      },
      scales: {
        x: {
          ticks: {
            maxTicksLimit: 8,
          },
        },
        yRisk: {
          type: "linear",
          position: "left",
          min: 0,
          max: 1,
          ticks: {
            stepSize: 0.1,
          },
        },
        yTemp: {
          type: "linear",
          position: "right",
          min: 0,
          max: 1200,
          grid: {
            drawOnChartArea: false,
          },
        },
      },
    },
  });

  chart.__maxPoints = maxPoints;
  return chart;
}

function initChart() {
  riskChart = createDualAxisChart(dom.chart, MAX_CHART_POINTS);
}

function updateCharts() {
  if (riskChart) {
    const numericTemps = chartTemp
      .map((value) => Number(value))
      .filter((value) => Number.isFinite(value));

    if (numericTemps.length > 0) {
      const tMin = Math.min(...numericTemps);
      const tMax = Math.max(...numericTemps);
      const span = Math.max(tMax - tMin, 1.0);

      const dynamicMin = Math.max(0, tMin - (span * 0.45));
      const dynamicMax = tMax + (span * 0.55);

      riskChart.options.scales.yTemp.min = Math.floor(dynamicMin / 5) * 5;
      riskChart.options.scales.yTemp.max = Math.ceil(dynamicMax / 5) * 5;
    } else {
      riskChart.options.scales.yTemp.min = 0;
      riskChart.options.scales.yTemp.max = 80;
    }

    riskChart.data.labels = chartLabels;
    riskChart.data.datasets[0].data = chartRisk;
    riskChart.data.datasets[1].data = chartTemp;
    riskChart.update("none");
  }
}

function resetChart() {
  chartLabels = [];
  chartRisk = [];
  chartTemp = [];

  updateCharts();
}

function pushChartPoint(label, risk, temp) {
  if (!riskChart) {
    return;
  }

  const maxPoints = riskChart?.__maxPoints || MAX_CHART_POINTS;
  chartLabels.push(label);
  chartRisk.push(risk);
  chartTemp.push(temp);

  while (chartLabels.length > maxPoints) {
    chartLabels.shift();
    chartRisk.shift();
    chartTemp.shift();
  }

  updateCharts();
}

function setFrameSource(src) {
  if (!dom.liveFrame || !dom.frameFallback) {
    return;
  }
  if (!src) {
    dom.liveFrame.style.display = "none";
    dom.frameFallback.style.display = "grid";
    return;
  }
  dom.liveFrame.src = src;
  dom.liveFrame.style.display = "block";
  dom.frameFallback.style.display = "none";
}

function paintDecision(decision) {
  const risk = Number(decision?.risk_score || 0);
  const alarm = Boolean(decision?.trigger_alarm);
  const action = decision?.suggested_action || "CONTINUE_MONITORING";

  if (dom.riskValue) {
    dom.riskValue.textContent = formatNumber(risk, 2);
  }
  if (dom.riskState) {
    dom.riskState.textContent = alarm ? "ALARM" : "MONITOR";
  }
  if (dom.decisionAction) {
    dom.decisionAction.textContent = action;
  }
  if (dom.sideActionValue) {
    dom.sideActionValue.textContent = action;
  }
  if (dom.sideRiskValue) {
    dom.sideRiskValue.textContent = formatNumber(risk, 2);
  }

  document.body.classList.toggle("alarm-mode", alarm);
}

function buildCurrentTelemetryRow(metrics) {
  if (!metrics) {
    return "";
  }

  const frameId = metrics.frame_id ?? "-";
  const risk = Number(metrics?.decision?.risk_score ?? metrics?.risk_score ?? 0);
  const fps = Number(metrics?.fps ?? 0);
  const timestamp = formatTimeLabel(metrics?.timestamp || "");

  const adjustedTemp = getSceneTemperature(metrics);
  const rawSystemTemp = metrics?.system_temperature_celsius;
  const tempSource = getTemperatureSource(metrics);

  const sceneText = adjustedTemp === null || !Number.isFinite(adjustedTemp)
    ? `-- °C (${tempSource})`
    : `${formatNumber(adjustedTemp, 2)} °C (${tempSource})`;
  const hostText = rawSystemTemp === null || rawSystemTemp === undefined
    ? "host temp unavailable"
    : `host diagnostic ${formatNumber(rawSystemTemp, 2)} °C`;

  const detail = `frame=${frameId}, risk=${formatNumber(risk, 2)}, fps=${formatNumber(fps, 1)}, ${hostText}, time=${timestamp}`;
  return `<tr><td>SCENE</td><td>${escapeHtml(sceneText)}</td><td>${escapeHtml(detail)}</td></tr>`;
}

function renderDetectionDetails(detections, metrics = null) {
  const rows = Array.isArray(detections) ? detections : [];
  const currentRowHtml = buildCurrentTelemetryRow(metrics);

  if (dom.topConfValue) {
    dom.topConfValue.textContent = "0%";
  }
  if (dom.topClassValue) {
    dom.topClassValue.textContent = "No detection (current frame)";
  }
  if (dom.sideTopConfValue) {
    dom.sideTopConfValue.textContent = "0%";
  }
  if (dom.sideTopClassValue) {
    dom.sideTopClassValue.textContent = "No detection (current frame)";
  }

  if (!dom.detectionBody) {
    // Continue to keep side panel functional even if table body is absent.
  }

  if (rows.length === 0) {
    if (dom.detectionBody) {
      const emptyRow = "<tr><td colspan='3' class='empty-cell'>No detections in current frame</td></tr>";
      dom.detectionBody.innerHTML = `${currentRowHtml}${emptyRow}`;
    }
    if (dom.sideDetectionList) {
      dom.sideDetectionList.innerHTML = "<li class='side-empty'>No detections in current frame</li>";
    }
    return;
  }

  const sorted = [...rows].sort((a, b) => Number(b?.confidence || 0) - Number(a?.confidence || 0));
  const top = sorted[0];
  if (dom.topConfValue) {
    dom.topConfValue.textContent = `${formatNumber(Number(top?.confidence || 0) * 100, 1)}%`;
  }
  if (dom.topClassValue) {
    dom.topClassValue.textContent = String(top?.class_name || "unknown");
  }
  if (dom.sideTopConfValue) {
    dom.sideTopConfValue.textContent = `${formatNumber(Number(top?.confidence || 0) * 100, 1)}%`;
  }
  if (dom.sideTopClassValue) {
    dom.sideTopClassValue.textContent = String(top?.class_name || "unknown");
  }

  const limited = sorted.slice(0, 8);
  const html = limited
    .map((item) => {
      const cls = escapeHtml(item?.class_name || "unknown");
      const conf = `${formatNumber(Number(item?.confidence || 0) * 100, 1)}%`;
      const bbox = Array.isArray(item?.bbox) ? item.bbox.map((v) => formatNumber(v, 2)).join(", ") : "-";
      return `<tr><td>${cls}</td><td>${conf}</td><td>${bbox}</td></tr>`;
    })
    .join("");

  if (dom.detectionBody) {
    dom.detectionBody.innerHTML = `${currentRowHtml}${html}`;
  }

  if (dom.sideDetectionList) {
    const sideHtml = limited
      .slice(0, 5)
      .map((item) => {
        const cls = escapeHtml(item?.class_name || "unknown");
        const conf = `${formatNumber(Number(item?.confidence || 0) * 100, 1)}%`;
        return `<li><span class='side-det-class'>${cls}</span><span class='side-det-conf'>${conf}</span></li>`;
      })
      .join("");
    dom.sideDetectionList.innerHTML = sideHtml;
  }
}

function paintMetrics(metrics) {
  if (!metrics) {
    return;
  }

  const detectionRows = Array.isArray(metrics.detections) ? metrics.detections : [];
  const currentDetectionCount = detectionRows.length;

  if (dom.fpsValue) {
    dom.fpsValue.textContent = formatNumber(metrics.fps, 1);
  }
  if (dom.detCountValue) {
    dom.detCountValue.textContent = String(currentDetectionCount);
  }
  if (dom.sideDetCountValue) {
    dom.sideDetCountValue.textContent = String(currentDetectionCount);
  }

  renderDetectionDetails(detectionRows, metrics);

  const temp = getSceneTemperature(metrics);
  const temperatureSource = getTemperatureSource(metrics);
  if (dom.tempValue) {
    dom.tempValue.textContent = temp === null || !Number.isFinite(temp) ? "--" : formatNumber(temp, 2);
  }
  if (dom.sideTempValue) {
    dom.sideTempValue.textContent = temp === null || !Number.isFinite(temp)
      ? "-- °C"
      : `${formatNumber(temp, 2)} °C`;
  }
  if (dom.tempSourceValue) {
    dom.tempSourceValue.textContent = temperatureSource;
  }
  if (dom.sideTempSourceValue) {
    dom.sideTempSourceValue.textContent = temperatureSource;
  }

  paintDecision(metrics.decision || {});

  const risk = Number(metrics.decision?.risk_score || 0);
  const chartTempValue = temp === null || !Number.isFinite(temp) ? 0 : Number(temp);
  const label = formatTimeLabel(metrics.timestamp);
  pushChartPoint(label, risk, chartTempValue);
}

function toPointTimeLabel(point, index = 0) {
  if (point?.timestamp) {
    return formatTimeLabel(point.timestamp);
  }

  if (point?.time_sec !== null && point?.time_sec !== undefined) {
    const sec = Number(point.time_sec);
    if (Number.isFinite(sec)) {
      return `T+${sec.toFixed(1)}s`;
    }
  }

  const frameId = Number(point?.frame_id);
  if (Number.isFinite(frameId)) {
    return `T+${(frameId / 30).toFixed(1)}s`;
  }

  return `T+${(index / 5).toFixed(1)}s`;
}

function paintVideoTelemetry(video) {
  if (!video) {
    return;
  }

  const points = Array.isArray(video.telemetry?.points) ? video.telemetry.points : [];
  if (points.length > 0) {
    resetChart();
    for (const [index, point] of points.entries()) {
      const frameLabel = toPointTimeLabel(point, index);
      const risk = Number(point.risk_score || 0);
      const temp = Number(
        point.scene_temperature_celsius ?? point.vision_temperature_celsius ?? 0,
      );
      pushChartPoint(frameLabel, risk, temp);
    }

    const last = points[points.length - 1];
    paintDecision({
      risk_score: Number(last.risk_score || 0),
      trigger_alarm: Boolean(last.trigger_alarm),
      suggested_action: last.suggested_action || "CONTINUE_MONITORING",
    });

    if (dom.detCountValue) {
      dom.detCountValue.textContent = String(last.detection_count || 0);
    }
    if (dom.tempValue) {
      const lastTemp = last.scene_temperature_celsius ?? last.vision_temperature_celsius;
      dom.tempValue.textContent = formatNumber(lastTemp, 2);
    }
    if (dom.tempSourceValue) {
      dom.tempSourceValue.textContent = getTemperatureSource(last);
    }
    if (dom.sideTempValue) {
      const lastTemp = last.scene_temperature_celsius ?? last.vision_temperature_celsius;
      dom.sideTempValue.textContent = `${formatNumber(lastTemp, 2)} °C`;
    }
    if (dom.sideTempSourceValue) {
      dom.sideTempSourceValue.textContent = getTemperatureSource(last);
    }
  }

  if (dom.fpsValue) {
    dom.fpsValue.textContent = "0";
  }

  // Video telemetry points do not contain per-box confidence details.
  const lastPoint = points.length > 0 ? points[points.length - 1] : null;
  renderDetectionDetails([], {
    frame_id: lastPoint?.frame_id,
    timestamp: lastPoint?.timestamp,
    vision_temperature_celsius: lastPoint?.vision_temperature_celsius,
    scene_temperature_celsius: lastPoint?.scene_temperature_celsius,
    scene_temperature_source: lastPoint?.scene_temperature_source || lastPoint?.temperature_source,
    system_temperature_celsius: lastPoint?.system_temperature_celsius,
    system_temperature_source: lastPoint?.system_temperature_source || "video-telemetry",
    fps: 0,
    risk_score: lastPoint?.risk_score,
    decision: { risk_score: Number(lastPoint?.risk_score || 0) },
  });

  const preview = video.preview_image_base64;
  if (preview) {
    setFrameSource(`data:image/jpeg;base64,${preview}`);
  }
}

function renderLiveHistoryFromState(state) {
  const history = state?.history || {};
  const risks = Array.isArray(history.risk) ? history.risk : [];
  const temps = Array.isArray(history.temperature) ? history.temperature : [];
  const timestamps = Array.isArray(history.timestamps) ? history.timestamps : [];

  if (risks.length === 0 && temps.length === 0) {
    return;
  }

  resetChart();
  const length = Math.max(risks.length, temps.length, timestamps.length);
  for (let i = 0; i < length; i += 1) {
    const label = timestamps[i] ? formatTimeLabel(timestamps[i]) : `T+${(i / 5).toFixed(1)}s`;
    const risk = Number(risks[i] || 0);
    const temp = Number(temps[i] || 0);
    pushChartPoint(label, risk, temp);
  }
}

function openMediaModal({ src, title, downloadHref = "", isMjpeg = false }) {
  if (!dom.mediaModal || !dom.mediaModalImage) {
    return;
  }
  if (!src) {
    return;
  }

  previousFocusedElement = document.activeElement;
  dom.mediaModal.classList.remove("hidden");
  dom.mediaModal.setAttribute("aria-hidden", "false");

  if (dom.mediaModalTitle) {
    dom.mediaModalTitle.textContent = title || "Preview";
  }

  if (dom.mediaModalDownload) {
    if (downloadHref) {
      dom.mediaModalDownload.href = downloadHref;
      dom.mediaModalDownload.classList.remove("hidden");
    } else {
      dom.mediaModalDownload.href = "#";
      dom.mediaModalDownload.classList.add("hidden");
    }
  }

  const sourceWithBust = isMjpeg ? `${src}${src.includes("?") ? "&" : "?"}t=${Date.now()}` : src;
  dom.mediaModalImage.src = sourceWithBust;
  dom.btnMediaClose?.focus();
}

function setStatus(node, text, tone = "neutral") {
  if (!node) {
    return;
  }
  node.textContent = text;
  node.classList.remove("status-neutral", "status-good", "status-alert");
  node.classList.add(`status-${tone}`);
}

function setBusy(button, busy, busyLabel = "Working...") {
  if (!button) {
    return;
  }
  if (busy) {
    if (!button.dataset.idleLabel) {
      button.dataset.idleLabel = button.textContent;
    }
    button.disabled = true;
    button.setAttribute("aria-busy", "true");
    button.textContent = busyLabel;
    return;
  }
  button.disabled = false;
  button.removeAttribute("aria-busy");
  if (button.dataset.idleLabel) {
    button.textContent = button.dataset.idleLabel;
  }
}

async function getResponseError(response) {
  try {
    const payload = await response.json();
    const detail = payload.detail || payload.error_message;
    if (Array.isArray(detail)) {
      return detail.map((item) => item?.msg || JSON.stringify(item)).join("; ");
    }
    return String(detail || `HTTP ${response.status}`);
  } catch (error) {
    return `HTTP ${response.status}`;
  }
}

function readOptionalNumber(node) {
  if (!node || !node.value.trim()) {
    return null;
  }
  const value = Number(node.value);
  return Number.isFinite(value) ? value : null;
}

function getSceneTemperature(metrics) {
  const value = metrics?.scene_temperature_celsius ?? metrics?.vision_temperature_celsius;
  return value === null || value === undefined ? null : Number(value);
}

function getTemperatureSource(metrics) {
  return String(
    metrics?.scene_temperature_source
      || metrics?.temperature_source
      || metrics?.temperature?.scene_temperature_source
      || "unavailable",
  ).replaceAll("_", " ");
}

function closeMediaModal() {
  if (!dom.mediaModal || !dom.mediaModalImage) {
    return;
  }

  dom.mediaModal.classList.add("hidden");
  dom.mediaModal.setAttribute("aria-hidden", "true");
  dom.mediaModalImage.src = "";
  if (previousFocusedElement && typeof previousFocusedElement.focus === "function") {
    previousFocusedElement.focus();
  }
  previousFocusedElement = null;
}

function renderArtifacts(files) {
  if (!dom.artifactList) {
    return;
  }

  const visualFiles = (Array.isArray(files) ? files : []).filter((file) => file?.kind === "image" || file?.kind === "video");

  if (visualFiles.length === 0) {
    dom.artifactList.innerHTML = "<p class='helper'>No previewable image/video files found.</p>";
    if (dom.artifactSummary) {
      dom.artifactSummary.textContent = "No previewable image/video files found.";
    }
    return;
  }

  const rows = visualFiles.map((file) => {
    const encodedPath = encodeURIComponent(file.relative_path || "");
    const safeUrl = escapeHtml(file.url || "");
    const safeName = escapeHtml(file.name || "artifact");

    let preview = "<div class='artifact-fallback'>No preview</div>";
    if (file.kind === "image") {
      preview = `<img src='${safeUrl}' alt='${safeName}' loading='lazy' />`;
    } else if (file.kind === "video") {
      preview = `<img src='/api/video/thumbnail?path=${encodedPath}' alt='${safeName}' loading='lazy' />`;
    }

    const openPreviewButton = `<button
      class='artifact-link artifact-open-btn'
      type='button'
      data-kind='${file.kind}'
      data-name='${safeName}'
      data-url='${safeUrl}'
      data-path='${encodedPath}'
    >Open Preview</button>`;
    const downloadLink = `<a class='artifact-link' href='${safeUrl}' target='_blank' rel='noopener noreferrer'>Download</a>`;

    return `
      <article class='artifact-item'>
        <div class='artifact-preview'>${preview}</div>
        <div class='artifact-meta'>
          <div class='artifact-name' title='${safeName}'>${safeName}</div>
          <div class='artifact-detail'>${file.kind.toUpperCase()} | ${formatFileSize(file.size_bytes)}</div>
          ${openPreviewButton}
          ${downloadLink}
        </div>
      </article>
    `;
  });

  dom.artifactList.innerHTML = rows.join("");
  if (dom.artifactSummary) {
    dom.artifactSummary.textContent = `Showing ${visualFiles.length} previewable files.`;
  }
}

async function refreshArtifacts() {
  const limit = Number(dom.artifactLimit?.value || 48);

  try {
    const response = await fetch(`/api/generated/files?limit=${limit}`);
    if (!response.ok) {
      throw new Error(await getResponseError(response));
    }
    const payload = await response.json();
    renderArtifacts(payload.files || []);
  } catch (error) {
    if (dom.artifactList) {
      dom.artifactList.innerHTML = "<p class='helper'>Failed to load generated artifacts.</p>";
    }
    if (dom.artifactSummary) {
      dom.artifactSummary.textContent = "Failed to load generated files.";
    }
  }
}

async function cleanupArtifacts() {
  if (!window.confirm("Remove generated files older than the selected keep count?")) {
    return;
  }

  if (dom.artifactSummary) {
    dom.artifactSummary.textContent = "Cleaning old generated files...";
  }

  const keepLatest = Number(dom.artifactLimit?.value || 48);
  try {
    const response = await fetch(`/api/generated/files/cleanup?keep_latest=${keepLatest}`, {
      method: "POST",
    });
    if (!response.ok) {
      throw new Error(await getResponseError(response));
    }

    const payload = await response.json();
    renderArtifacts(payload.files || []);
    if (dom.artifactSummary) {
      dom.artifactSummary.textContent = `Cleanup done. Removed ${payload.deleted_count || 0} files.`;
    }
  } catch (error) {
    if (dom.artifactSummary) {
      dom.artifactSummary.textContent = `Cleanup failed: ${error.message}`;
    }
  }
}

async function refreshHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const payload = await response.json();
    if (payload.model_ready) {
      setBadge(dom.healthBadge, "READY", "good");
    } else {
      setBadge(dom.healthBadge, "DEGRADED", "alert");
    }
  } catch (error) {
    setBadge(dom.healthBadge, "OFFLINE", "alert");
  }
}

async function refreshModelMeta() {
  try {
    const response = await fetch("/api/model/info");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const payload = await response.json();
    const classes = payload.classes || {};
    const classCount = Object.keys(classes).length;
    const modelPath = payload.model_path || "unknown";
    if (dom.modelMeta) {
      dom.modelMeta.textContent = `Model: ${modelPath} | Classes: ${classCount}`;
    }
  } catch (error) {
    if (dom.modelMeta) {
      dom.modelMeta.textContent = "Model metadata unavailable";
    }
  }
}

function stopFrameLoop() {
  if (frameTimer) {
    clearInterval(frameTimer);
    frameTimer = null;
  }
}

function startFrameLoop() {
  stopFrameLoop();
  frameTimer = setInterval(() => {
    if (dom.liveFrame) {
      dom.liveFrame.src = `/api/live/frame?t=${Date.now()}`;
    }
  }, 240);
}

function disconnectEvents() {
  if (liveEventSource) {
    liveEventSource.close();
    liveEventSource = null;
  }
}

function connectEvents() {
  disconnectEvents();
  liveEventSource = new EventSource("/api/live/events");

  liveEventSource.onmessage = (event) => {
    let state = null;
    try {
      state = JSON.parse(event.data);
    } catch (error) {
      return;
    }

    if (!state) {
      return;
    }

    if (dom.streamLogPath) {
      const logPath = state.current_log_path || "not started";
      dom.streamLogPath.textContent = `Live log: ${logPath}`;
    }

    if (state.last_error) {
      setStatus(dom.streamStatus, state.last_error, "alert");
      setBadge(dom.streamBadge, "WARN", "alert");
    }

    if (state.running) {
      setBadge(dom.streamBadge, "LIVE", "good");
      setStatus(dom.streamStatus, "Receiving live telemetry.", "good");
    } else if (!state.last_error && state.latest_metrics && Object.keys(state.latest_metrics).length > 0) {
      setBadge(dom.streamBadge, "IDLE", "neutral");
      setStatus(dom.streamStatus, "Live source stopped after the latest frame.", "neutral");
    }

    setLiveSnapshotVisible(Boolean(state.running));

    if (state.latest_metrics && Object.keys(state.latest_metrics).length > 0) {
      paintMetrics(state.latest_metrics);
    }
  };

  liveEventSource.onerror = () => {
    setBadge(dom.streamBadge, "RETRY", "alert");
    setStatus(dom.streamStatus, "Live events disconnected; the browser will retry.", "alert");
  };
}

async function startLive() {
  if (!dom.sourceInput || !dom.confInput || !dom.frameSkipInput || !dom.maxWidthInput) {
    return;
  }
  const source = (dom.sourceInput.value || "0").trim() || "0";
  const conf = Number(dom.confInput.value || 0.25);
  const frameSkip = Math.max(1, Math.min(8, Number(dom.frameSkipInput.value || 1)));
  const maxFrameWidth = Math.max(640, Math.min(3840, Number(dom.maxWidthInput.value || 1280)));
  setBusy(dom.btnLiveStart, true, "Starting...");
  setStatus(dom.streamStatus, "Opening the live source...", "neutral");

  try {
    const response = await fetch("/api/live/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source,
        conf,
        frame_skip: frameSkip,
        max_frame_width: maxFrameWidth,
      }),
    });

    if (!response.ok) {
      throw new Error(await getResponseError(response));
    }

    setBadge(dom.streamBadge, "STARTING", "neutral");
    setStatus(dom.streamStatus, `Live source opened: ${source}. Waiting for the first frame...`, "neutral");

    setLiveSnapshotVisible(true);
    resetChart();
    connectEvents();
    startFrameLoop();
  } catch (error) {
    setBadge(dom.streamBadge, "ERROR", "alert");
    setStatus(dom.streamStatus, `Start failed: ${error.message}`, "alert");
  } finally {
    setBusy(dom.btnLiveStart, false);
  }
}

async function stopLive() {
  setBusy(dom.btnLiveStop, true, "Stopping...");
  try {
    const response = await fetch("/api/live/stop", { method: "POST" });
    if (!response.ok) {
      throw new Error(await getResponseError(response));
    }
  } catch (error) {
    setStatus(dom.streamStatus, `Stop failed: ${error.message}`, "alert");
  }

  disconnectEvents();
  stopFrameLoop();
  setLiveSnapshotVisible(false);
  setBadge(dom.streamBadge, "IDLE", "neutral");
  if (!dom.streamStatus?.classList.contains("status-alert")) {
    setStatus(dom.streamStatus, "Live stream stopped.", "neutral");
  }
  setBusy(dom.btnLiveStop, false);
}

async function runImageInference(event) {
  event.preventDefault();

  if (!dom.imageInput || !dom.uploadBtn || !dom.uploadStatus) {
    return;
  }

  setLiveSnapshotVisible(false);

  const file = dom.imageInput.files[0];
  if (!file) {
    setStatus(dom.uploadStatus, "Choose an image file first.", "alert");
    return;
  }

  const sensorValue = readOptionalNumber(dom.imageSensorTempInput);
  if (dom.imageSensorTempInput?.value.trim() && sensorValue === null) {
    setStatus(dom.uploadStatus, "Thermal sensor temperature must be a number.", "alert");
    return;
  }
  if (sensorValue !== null && (sensorValue < -50 || sensorValue > 200)) {
    setStatus(dom.uploadStatus, "Thermal sensor temperature must be between -50 and 200 °C.", "alert");
    return;
  }

  setBusy(dom.uploadBtn, true, "Processing...");
  setStatus(dom.uploadStatus, "Running image inference...", "neutral");

  const formData = new FormData();
  formData.append("file", file);
  formData.append("save_annotated", "true");
  if (sensorValue !== null) {
    formData.append("sensor_temperature_celsius", String(sensorValue));
  }

  try {
    const response = await fetch("/api/inference/image", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(await getResponseError(response));
    }

    const payload = await response.json();
    const inference = payload.inference;

    paintMetrics({
      fps: 0,
      detection_count: inference.detection_count,
      vision_temperature_celsius: inference.vision_temperature_celsius,
      scene_temperature_celsius: inference.scene_temperature_celsius,
      scene_temperature_source: inference.scene_temperature_source,
      temperature: inference.temperature,
      decision: inference.decision,
      detections: inference.detections,
      frame_id: inference.frame_id,
    });

    if (inference.annotated_image_base64) {
      setFrameSource(`data:image/jpeg;base64,${inference.annotated_image_base64}`);
    }

    const action = inference.decision?.suggested_action || "CONTINUE_MONITORING";
    setStatus(dom.uploadStatus, `Image inference complete. Action: ${action}.`, "good");
    await refreshArtifacts();
  } catch (error) {
    setStatus(dom.uploadStatus, `Image inference failed: ${error.message}`, "alert");
  } finally {
    setBusy(dom.uploadBtn, false);
  }
}

async function runVideoInference(event) {
  event.preventDefault();

  if (!dom.videoInput || !dom.videoBtn || !dom.videoStatus) {
    return;
  }

  setLiveSnapshotVisible(false);

  const file = dom.videoInput.files[0];
  if (!file) {
    setStatus(dom.videoStatus, "Choose a video file first.", "alert");
    return;
  }

  const sensorValue = readOptionalNumber(dom.videoSensorTempInput);
  if (dom.videoSensorTempInput?.value.trim() && sensorValue === null) {
    setStatus(dom.videoStatus, "Thermal sensor temperature must be a number.", "alert");
    return;
  }
  if (sensorValue !== null && (sensorValue < -50 || sensorValue > 200)) {
    setStatus(dom.videoStatus, "Thermal sensor temperature must be between -50 and 200 °C.", "alert");
    return;
  }

  setBusy(dom.videoBtn, true, "Processing...");
  setStatus(dom.videoStatus, "Running video inference. This may take a while...", "neutral");

  const formData = new FormData();
  formData.append("file", file);
  formData.append("with_decision", "true");
  if (sensorValue !== null) {
    formData.append("sensor_temperature_celsius", String(sensorValue));
  }

  try {
    const response = await fetch("/api/inference/video", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(await getResponseError(response));
    }

    const payload = await response.json();
    const video = payload.video || {};
    paintVideoTelemetry(video);

    const playerUrl = video.player_url || "";
    setStatus(dom.videoStatus, `Video inference complete. Frames: ${video.frame_count || 0}; alarm frames: ${video.alarm_frame_count || 0}${playerUrl ? `; player: ${playerUrl}` : ""}`, "good");
    await refreshArtifacts();
  } catch (error) {
    setStatus(dom.videoStatus, `Video inference failed: ${error.message}`, "alert");
  } finally {
    setBusy(dom.videoBtn, false);
  }
}

async function runVCNPipeline() {
  if (!dom.pipelineStatus || !dom.btnRunVCN) {
    return;
  }

  setLiveSnapshotVisible(false);

  setBusy(dom.btnRunVCN, true, "Running...");
  setStatus(dom.pipelineStatus, "Running the four-camera VCN pipeline...", "neutral");

  try {
    const response = await fetch("/api/pipeline/vcn/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });

    if (!response.ok) {
      throw new Error(await getResponseError(response));
    }

    const payload = await response.json();
    const count = (payload.pipeline?.processed_files || []).length;
    const outputUrl = payload.pipeline?.output_url;
    if (outputUrl) {
      setFrameSource(`${outputUrl}?t=${Date.now()}`);
    }
    setStatus(dom.pipelineStatus, `VCN pipeline complete. Processed cameras: ${count}.`, "good");
    await refreshArtifacts();
  } catch (error) {
    setStatus(dom.pipelineStatus, `VCN pipeline failed: ${error.message}`, "alert");
  } finally {
    setBusy(dom.btnRunVCN, false);
  }
}

async function restoreLiveState() {
  try {
    const response = await fetch("/api/live/state");
    if (!response.ok) {
      return;
    }

    const state = await response.json();
    renderLiveHistoryFromState(state);
    setLiveSnapshotVisible(Boolean(state.running));

    if (state.running) {
      setBadge(dom.streamBadge, "LIVE", "good");
      setStatus(dom.streamStatus, "Recovered the existing live stream state.", "good");
      connectEvents();
      startFrameLoop();
    } else if (state.last_error) {
      setBadge(dom.streamBadge, "WARN", "alert");
      setStatus(dom.streamStatus, state.last_error, "alert");
    }

    if (dom.streamLogPath) {
      const logPath = state.current_log_path || "not started";
      dom.streamLogPath.textContent = `Live log: ${logPath}`;
    }

    if (state.latest_metrics && Object.keys(state.latest_metrics).length > 0) {
      paintMetrics(state.latest_metrics);
    }
  } catch (error) {
    // Keep quiet during startup.
  }
}

function openLivePreviewZoom() {
  if (!dom.liveFrame || !dom.liveFrame.src) {
    return;
  }

  openMediaModal({
    src: dom.liveFrame.src,
    title: "Annotated Preview",
    downloadHref: dom.liveFrame.src,
  });
}

function onArtifactListClick(event) {
  const button = event.target.closest(".artifact-open-btn");
  if (!button) {
    return;
  }

  const kind = button.dataset.kind || "other";
  const name = button.dataset.name || "Artifact";
  const fileUrl = button.dataset.url || "";
  const encodedPath = button.dataset.path || "";
  const relativePath = decodeURIComponent(encodedPath || "");

  if (kind === "video" && relativePath) {
    openMediaModal({
      src: `/api/video/mjpeg?path=${encodeURIComponent(relativePath)}&fps=15&loop=true`,
      title: `${name} (Embedded Player)`,
      downloadHref: fileUrl,
      isMjpeg: true,
    });
    return;
  }

  if (fileUrl) {
    openMediaModal({
      src: fileUrl,
      title: name,
      downloadHref: fileUrl,
    });
  }
}

function bindEvents() {
  bindSafe(dom.btnHealth, "click", refreshHealth);
  bindSafe(dom.uploadForm, "submit", runImageInference);
  bindSafe(dom.videoForm, "submit", runVideoInference);
  bindSafe(dom.btnRunVCN, "click", runVCNPipeline);
  bindSafe(dom.btnRefreshArtifacts, "click", refreshArtifacts);
  bindSafe(dom.btnCleanupArtifacts, "click", cleanupArtifacts);
  bindSafe(dom.artifactLimit, "change", refreshArtifacts);
  bindSafe(dom.btnLiveStart, "click", startLive);
  bindSafe(dom.btnLiveStop, "click", stopLive);
  bindSafe(dom.btnPreviewZoom, "click", openLivePreviewZoom);
  bindSafe(dom.artifactList, "click", onArtifactListClick);
  bindSafe(dom.btnMediaClose, "click", closeMediaModal);
  bindSafe(dom.mediaModal, "click", (event) => {
    if (event.target === dom.mediaModal) {
      closeMediaModal();
    }
  });

  bindSafe(dom.liveFrame, "error", () => {
    setFrameSource("");
  });

  bindSafe(dom.liveFrame, "load", () => {
    if (dom.liveFrame && dom.liveFrame.src) {
      dom.liveFrame.style.display = "block";
      if (dom.frameFallback) {
        dom.frameFallback.style.display = "none";
      }
    }
  });
  bindSafe(dom.liveFrame, "click", openLivePreviewZoom);
}

function init() {
  try {
    setLiveSnapshotVisible(false);
    initChart();
    bindEvents();
    refreshHealth();
    refreshModelMeta();
    restoreLiveState();
    refreshArtifacts();
    setInterval(refreshHealth, 8000);
  } catch (error) {
    if (dom.pipelineStatus) {
      dom.pipelineStatus.textContent = `UI init error: ${error.message}`;
    }
  }
}

init();

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeMediaModal();
  }
});

window.addEventListener("error", (event) => {
  if (dom.pipelineStatus) {
    dom.pipelineStatus.textContent = `Frontend error: ${event.message}`;
  }
});
