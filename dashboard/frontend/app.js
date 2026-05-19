/**
 * AI Predictive Monitor — Dashboard client
 *
 * Responsibilities:
 *   - WebSocket connection with auto-reconnect
 *   - Render prediction cards sorted by severity
 *   - Live countdown timers that tick every second
 *   - Metrics overview tiles
 *   - Header status indicators
 */

'use strict';

// ── State ──────────────────────────────────────────────────────────
const state = {
  predictions: [],
  metrics: {},
  lastUpdated: null,
  wsConnected: false,
};

// Map each prediction ID → { createdAt (ms), horizonHours }
// Used to compute live countdowns without re-rendering cards.
const countdownMap = new Map();

// ── Severity ordering ──────────────────────────────────────────────
const SEV_ORDER = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 };
const SEV_ICONS = { CRITICAL: '●', HIGH: '●', MEDIUM: '◆', LOW: '◆' };

// ── DOM refs ───────────────────────────────────────────────────────
const els = {
  initOverlay:    () => document.getElementById('init-overlay'),
  mainContent:    () => document.getElementById('main-content'),
  predictGrid:    () => document.getElementById('predictions-grid'),
  metricsGrid:    () => document.getElementById('metrics-grid'),
  predCount:      () => document.getElementById('pred-count'),
  metricsCount:   () => document.getElementById('metrics-count'),
  alertBanner:    () => document.getElementById('alert-banner'),
  alertText:      () => document.getElementById('alert-text'),
  wsDot:          () => document.getElementById('ws-dot'),
  wsLabel:        () => document.getElementById('ws-label'),
  lastUpdated:    () => document.getElementById('last-updated'),
  footerClock:    () => document.getElementById('footer-clock'),
  footerClients:  () => document.getElementById('footer-clients'),
  initStatus:     () => document.getElementById('init-status'),
};

// ── WebSocket ──────────────────────────────────────────────────────
let ws = null;
let reconnectDelay = 1500;

function connectWS() {
  setWsStatus('connecting');
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    setWsStatus('connected');
    reconnectDelay = 1500;
  };

  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      handleMessage(msg);
    } catch (e) {
      console.error('WS parse error', e);
    }
  };

  ws.onclose = () => {
    setWsStatus('disconnected');
    setTimeout(connectWS, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 1.5, 15000);
  };

  ws.onerror = () => ws.close();
}

function handleMessage(msg) {
  if (msg.type === 'status') {
    els.initStatus().textContent = msg.message;
    return;
  }

  if (msg.type === 'predictions_update') {
    state.predictions = msg.predictions || [];
    state.metrics = msg.metrics_summary || {};
    state.lastUpdated = new Date();
    showDashboard();
    renderPredictions(state.predictions);
    renderMetrics(state.metrics);
    updateHeaderMeta();
  }
}

// ── Layout control ─────────────────────────────────────────────────
function showDashboard() {
  els.initOverlay().style.display = 'none';
  els.mainContent().style.display = 'block';
}

// ── Predictions ────────────────────────────────────────────────────
function renderPredictions(predictions) {
  const grid = els.predictGrid();

  // Sort: CRITICAL → HIGH → MEDIUM → LOW, then by confidence desc
  const sorted = [...predictions].sort((a, b) => {
    const so = (SEV_ORDER[a.severity] ?? 9) - (SEV_ORDER[b.severity] ?? 9);
    return so !== 0 ? so : b.confidence - a.confidence;
  });

  // Register countdown entries
  countdownMap.clear();
  sorted.forEach(p => {
    const createdMs = new Date(p.created_at).getTime();
    countdownMap.set(p.id, { createdMs, horizonHours: p.horizon_hours });
  });

  // Update badge
  const n = sorted.length;
  els.predCount().textContent = `${n} prediction${n !== 1 ? 's' : ''}`;

  // Alert banner: show if any CRITICAL
  const criticals = sorted.filter(p => p.severity === 'CRITICAL');
  const banner = els.alertBanner();
  if (criticals.length > 0) {
    els.alertText().textContent =
      `${criticals.length} CRITICAL prediction${criticals.length > 1 ? 's' : ''} active — immediate action required`;
    banner.classList.add('visible');
  } else {
    banner.classList.remove('visible');
  }

  if (sorted.length === 0) {
    grid.innerHTML = '<div class="no-predictions">No active predictions — system nominal.</div>';
    return;
  }

  grid.innerHTML = sorted.map(p => cardHTML(p)).join('');
}

function cardHTML(p) {
  const confPct = Math.round(p.confidence * 100);
  const horizonLabel = formatHorizon(p.horizon_hours);
  const countdownId = `cd-${CSS.escape(p.id)}`;
  const scenario = (p.scenario || '').replace(/_/g, ' ');
  const timeClass = p.horizon_hours < 1 ? 'urgent' : p.horizon_hours < 4 ? 'warning' : '';

  return `
<div class="pred-card sev-${p.severity}">
  <div class="pred-card-header">
    <div class="pred-card-left">
      <span class="sev-badge ${p.severity}">${SEV_ICONS[p.severity] || '●'} ${p.severity}</span>
      <span class="pred-metric-name">${escHtml(p.metric_name)}</span>
    </div>
    <div class="pred-countdown">
      <span class="countdown-label">Time to event</span>
      <span class="countdown-time ${timeClass}" id="${countdownId}">${horizonLabel}</span>
    </div>
  </div>

  <div class="pred-message">${escHtml(p.message)}</div>

  <div class="pred-card-footer">
    <div class="confidence-section">
      <div class="confidence-label-row">
        <span>Confidence</span>
        <span class="confidence-pct">${confPct}%</span>
      </div>
      <div class="confidence-bar-bg">
        <div class="confidence-bar-fill" style="width:${confPct}%"></div>
      </div>
    </div>
    <span class="pred-scenario">${escHtml(scenario)}</span>
  </div>
</div>`;
}

// ── Metrics tiles ──────────────────────────────────────────────────
function renderMetrics(metrics) {
  const grid = els.metricsGrid();
  const names = Object.keys(metrics);

  els.metricsCount().textContent = `${names.length} metrics`;

  if (names.length === 0) {
    grid.innerHTML = '';
    return;
  }

  grid.innerHTML = names.sort().map(name => {
    const m = metrics[name];
    const { display, unit } = formatMetricValue(name, m.current_value);
    return `
<div class="metric-tile">
  <div class="metric-tile-name" title="${escHtml(name)}">${escHtml(name)}</div>
  <div class="metric-tile-value">${display}<span class="metric-tile-unit"> ${unit}</span></div>
</div>`;
  }).join('');
}

function formatMetricValue(name, value) {
  if (name.includes('_pct') || name.includes('_percent')) {
    return { display: value.toFixed(1), unit: '%' };
  }
  if (name.includes('_mb') || name.includes('memory_mb')) {
    return { display: value.toFixed(0), unit: 'MB' };
  }
  if (name.includes('_gb')) {
    return { display: value.toFixed(1), unit: 'GB' };
  }
  if (name.includes('_ms') || name.includes('latency')) {
    return { display: value.toFixed(0), unit: 'ms' };
  }
  if (name.includes('rate') || name.includes('rps')) {
    return { display: value.toFixed(0), unit: 'rps' };
  }
  if (name.includes('connections') && !name.includes('pct')) {
    return { display: value.toFixed(0), unit: 'conn' };
  }
  return { display: value.toFixed(1), unit: '' };
}

// ── Live countdown ticks ───────────────────────────────────────────
function tickCountdowns() {
  const now = Date.now();
  countdownMap.forEach(({ createdMs, horizonHours }, id) => {
    const el = document.getElementById(`cd-${CSS.escape(id)}`);
    if (!el) return;

    const horizonMs = horizonHours * 3600 * 1000;
    const elapsedMs = now - createdMs;
    const remainMs  = Math.max(0, horizonMs - elapsedMs);

    el.textContent = formatCountdownMs(remainMs);

    // update urgency class
    const remainH = remainMs / 3600000;
    el.classList.toggle('urgent',  remainH < 1);
    el.classList.toggle('warning', remainH >= 1 && remainH < 4);
  });
}

function formatCountdownMs(ms) {
  if (ms <= 0) return 'NOW';
  const totalSec = Math.floor(ms / 1000);
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m`;
  if (m > 0) return `${m}m ${String(s).padStart(2, '0')}s`;
  return `${s}s`;
}

function formatHorizon(hours) {
  if (!hours || hours <= 0) return 'IMMINENT';
  if (hours < 1) return `~${Math.round(hours * 60)}m`;
  if (hours < 2) return `~${hours.toFixed(1)}h`;
  return `~${Math.round(hours)}h`;
}

// ── Header / footer updates ────────────────────────────────────────
function updateHeaderMeta() {
  if (state.lastUpdated) {
    els.lastUpdated().textContent = `Updated ${state.lastUpdated.toLocaleTimeString()}`;
  }
}

function setWsStatus(status) {
  const dot = els.wsDot();
  const label = els.wsLabel();
  dot.className = `ws-dot ${status}`;
  const labels = { connected: 'Live', connecting: 'Connecting...', disconnected: 'Reconnecting...' };
  label.textContent = labels[status] || status;
  state.wsConnected = status === 'connected';
}

function tickClock() {
  const now = new Date();
  els.footerClock().textContent = now.toLocaleTimeString();
}

// ── Utility ────────────────────────────────────────────────────────
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Boot ───────────────────────────────────────────────────────────
connectWS();
setInterval(tickCountdowns, 1000);
setInterval(tickClock, 1000);
tickClock();

// Poll REST /api/predictions every 10s as WebSocket fallback
setInterval(async () => {
  if (state.wsConnected) return;
  try {
    const res = await fetch('/api/predictions');
    const data = await res.json();
    if (data.predictions?.length) {
      state.predictions = data.predictions;
      showDashboard();
      renderPredictions(state.predictions);
    }
  } catch (_) { /* ignore */ }
}, 10000);
