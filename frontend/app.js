/**
 * CoopTech — Frontend Application
 * Connects to the FastAPI backend at localhost:8000
 * and renders real-time analytics dashboards.
 */

const API_BASE = 'http://localhost:8000/api/v1';

// ─── State ───────────────────────────────────────────────
let dashboardData = null;
let segmentosData = null;
let clustersData = null;
let charts = {};

// ─── Chart.js Global Config ─────────────────────────────
Chart.defaults.color = '#A5D6A7';
Chart.defaults.borderColor = 'rgba(46, 125, 50, 0.08)';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.padding = 16;
Chart.defaults.plugins.tooltip.backgroundColor = '#172118';
Chart.defaults.plugins.tooltip.titleColor = '#E8F5E9';
Chart.defaults.plugins.tooltip.bodyColor = '#A5D6A7';
Chart.defaults.plugins.tooltip.borderColor = 'rgba(46, 125, 50, 0.3)';
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.cornerRadius = 10;
Chart.defaults.plugins.tooltip.padding = 12;

// ─── Color Palette ──────────────────────────────────────
const COLORS = {
    green: '#4CAF50',
    greenDark: '#2E7D32',
    greenLight: '#81C784',
    orange: '#F5A623',
    orangeLight: '#FFB74D',
    emerald: '#66BB6A',
    amber: '#FFA726',
    rose: '#EF5350',
    teal: '#26A69A',
    sky: '#42A5F5',
    pink: '#EC407A',
    cyan: '#26C6DA',
};

const CHART_COLORS = [COLORS.green, COLORS.orange, COLORS.emerald, COLORS.rose, COLORS.teal, COLORS.orangeLight, COLORS.sky, COLORS.pink, COLORS.greenLight, COLORS.cyan];
const CHART_COLORS_ALPHA = CHART_COLORS.map(c => c + '33');

// ─── Utility ────────────────────────────────────────────
function formatNumber(n) {
    if (n === null || n === undefined) return '—';
    return new Intl.NumberFormat('es-EC').format(n);
}

function animateValue(el, end, duration = 800) {
    const start = 0;
    const startTime = performance.now();
    const isFloat = String(end).includes('.');

    function update(now) {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        const current = start + (end - start) * eased;
        el.textContent = isFloat ? current.toFixed(2) + '%' : formatNumber(Math.round(current));
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

// ─── API Calls ──────────────────────────────────────────
async function fetchJSON(path) {
    const res = await fetch(API_BASE + path);
    if (!res.ok) throw new Error(`API Error: ${res.status} ${res.statusText}`);
    return res.json();
}

async function loadHealth() {
    try {
        const data = await fetchJSON('/health');
        document.getElementById('models-count').textContent = `${data.models_ready || 0} modelos`;
        document.getElementById('version-label').textContent = `v${data.version || '1.0.0'}`;

        const statusEl = document.getElementById('system-status');
        const dot = statusEl.querySelector('.status-dot');
        if (data.status === 'ok') {
            dot.style.background = 'var(--success)';
            dot.style.boxShadow = '0 0 8px var(--success)';
            statusEl.querySelector('span').textContent = 'Sistema activo';
        } else {
            dot.style.background = 'var(--danger)';
            dot.style.boxShadow = '0 0 8px var(--danger)';
            statusEl.querySelector('span').textContent = 'Error en sistema';
        }
    } catch (e) {
        console.error('Health check failed:', e);
        document.getElementById('models-count').textContent = 'Offline';
    }
}

async function loadDashboard() {
    try {
        dashboardData = await fetchJSON('/dashboard/kpis');
        renderKPIs(dashboardData);
        renderRiesgoChart(dashboardData.distribucion_riesgo);
        renderAlertasChart(dashboardData.distribucion_alerta);
    } catch (e) {
        console.error('Dashboard load failed:', e);
    }
}

async function loadSegmentos() {
    try {
        segmentosData = await fetchJSON('/dashboard/segmentos');
        renderProductosChart(segmentosData.por_producto);
        renderCanalesChart(segmentosData.por_canal_cobranza);
        renderSegmentsTable(segmentosData.por_producto);
        renderSegProductosChart(segmentosData.por_producto);
    } catch (e) {
        console.error('Segmentos load failed:', e);
    }
}

async function loadClusters() {
    try {
        clustersData = await fetchJSON('/dashboard/clusters');
        renderClustersChart(clustersData.clusters);
    } catch (e) {
        console.error('Clusters load failed:', e);
    }
}

// ─── Render KPIs ────────────────────────────────────────
function renderKPIs(data) {
    animateValue(document.getElementById('kpi-total-socios'), data.total_socios);
    animateValue(document.getElementById('kpi-tasa-mora'), data.tasa_mora_pct);
    animateValue(document.getElementById('kpi-alertas-desvio'), data.alertas_desvio);
    animateValue(document.getElementById('kpi-menores-detectados'), data.menores_detectados);

    // Alerts view
    const dist = data.distribucion_riesgo || {};
    const total = Object.values(dist).reduce((a, b) => a + b, 0) || 1;

    document.getElementById('alert-sin-riesgo').textContent = formatNumber(dist.sin_mora || 0);
    document.getElementById('alert-30d').textContent = formatNumber(dist.mora_30d || 0);
    document.getElementById('alert-60d').textContent = formatNumber(dist.mora_60d || 0);
    document.getElementById('alert-90d').textContent = formatNumber(dist.mora_90d || 0);

    // Animate bars
    setTimeout(() => {
        document.getElementById('bar-sin-riesgo').style.width = ((dist.sin_mora || 0) / total * 100) + '%';
        document.getElementById('bar-30d').style.width = ((dist.mora_30d || 0) / total * 100) + '%';
        document.getElementById('bar-60d').style.width = ((dist.mora_60d || 0) / total * 100) + '%';
        document.getElementById('bar-90d').style.width = ((dist.mora_90d || 0) / total * 100) + '%';
    }, 300);
}

// ─── Chart Renderers ────────────────────────────────────
function destroyChart(id) {
    if (charts[id]) {
        charts[id].destroy();
        delete charts[id];
    }
}

function renderRiesgoChart(dist) {
    if (!dist) return;
    destroyChart('riesgo');
    const ctx = document.getElementById('canvas-riesgo').getContext('2d');
    const labels = Object.keys(dist).map(k => k.replace('_', ' ').replace('mora ', 'Mora ').replace('sin mora', 'Sin mora'));
    const values = Object.values(dist);
    const colors = [COLORS.emerald, COLORS.orange, COLORS.orangeLight, COLORS.rose];

    charts.riesgo = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data: values,
                backgroundColor: colors,
                borderWidth: 0,
                hoverOffset: 8,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: { position: 'bottom', labels: { padding: 14, font: { size: 11 } } },
            },
            animation: { animateRotate: true, duration: 1000, easing: 'easeOutQuart' }
        }
    });
}

function renderAlertasChart(dist) {
    if (!dist) return;
    destroyChart('alertas');
    const ctx = document.getElementById('canvas-alertas').getContext('2d');
    const labels = Object.keys(dist).map(k => k.replace('_', ' '));
    const values = Object.values(dist);
    const colors = [COLORS.emerald, COLORS.rose, COLORS.orangeLight, COLORS.orange, COLORS.teal];

    charts.alertas = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data: values,
                backgroundColor: colors,
                borderWidth: 0,
                hoverOffset: 8,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: { position: 'bottom', labels: { padding: 14, font: { size: 11 } } },
            },
            animation: { animateRotate: true, duration: 1000, easing: 'easeOutQuart' }
        }
    });
}

function renderProductosChart(productos) {
    if (!productos) return;
    destroyChart('productos');
    const ctx = document.getElementById('canvas-productos').getContext('2d');
    const sorted = [...productos].sort((a, b) => b.score_riesgo - a.score_riesgo);
    const labels = sorted.map(p => `Prod. ${p.prod_bancario}`);
    const scores = sorted.map(p => (p.score_riesgo * 100).toFixed(1));
    const tasas = sorted.map(p => p.tasa_mora_pct);

    charts.productos = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                {
                    label: 'Score Riesgo (%)',
                    data: scores,
                    backgroundColor: scores.map(s => s > 70 ? COLORS.rose + '99' : s > 40 ? COLORS.orange + '99' : COLORS.emerald + '99'),
                    borderColor: scores.map(s => s > 70 ? COLORS.rose : s > 40 ? COLORS.orange : COLORS.emerald),
                    borderWidth: 1,
                    borderRadius: 6,
                    barPercentage: 0.6,
                },
                {
                    label: 'Tasa Mora (%)',
                    data: tasas,
                    backgroundColor: COLORS.violet + '55',
                    borderColor: COLORS.violet,
                    borderWidth: 1,
                    borderRadius: 6,
                    barPercentage: 0.6,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: { position: 'top', labels: { font: { size: 11 } } },
            },
            scales: {
                x: {
                    max: 100,
                    grid: { color: 'rgba(46, 125, 50, 0.06)' },
                    ticks: { callback: v => v + '%' }
                },
                y: {
                    grid: { display: false },
                }
            },
            animation: { duration: 1200, easing: 'easeOutQuart' }
        }
    });
}

function renderCanalesChart(canales) {
    if (!canales) return;
    destroyChart('canales');
    const ctx = document.getElementById('canvas-canales').getContext('2d');
    const labels = Object.keys(canales).map(k => k.charAt(0).toUpperCase() + k.slice(1));
    const values = Object.values(canales);
    const colors = [COLORS.green, COLORS.teal, COLORS.rose];

    charts.canales = new Chart(ctx, {
        type: 'pie',
        data: {
            labels,
            datasets: [{
                data: values,
                backgroundColor: colors,
                borderWidth: 0,
                hoverOffset: 10,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { padding: 14, font: { size: 11 } } },
            },
            animation: { animateRotate: true, duration: 1000, easing: 'easeOutQuart' }
        }
    });
}

function renderSegProductosChart(productos) {
    if (!productos) return;
    destroyChart('seg-productos');
    const ctx = document.getElementById('canvas-seg-productos').getContext('2d');
    const sorted = [...productos].sort((a, b) => b.tasa_mora_pct - a.tasa_mora_pct);
    const labels = sorted.map(p => `Producto ${p.prod_bancario}`);
    const socios = sorted.map(p => p.n_socios);
    const tasas = sorted.map(p => p.tasa_mora_pct);

    charts['seg-productos'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                {
                    label: 'Socios',
                    data: socios,
                    backgroundColor: COLORS.green + '88',
                    borderColor: COLORS.green,
                    borderWidth: 1,
                    borderRadius: 6,
                    yAxisID: 'y',
                },
                {
                    label: 'Tasa Mora (%)',
                    data: tasas,
                    type: 'line',
                    borderColor: COLORS.rose,
                    backgroundColor: COLORS.rose + '22',
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: COLORS.rose,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    yAxisID: 'y1',
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top', labels: { font: { size: 11 } } },
            },
            scales: {
                y: {
                    position: 'left',
                    grid: { color: 'rgba(46, 125, 50, 0.06)' },
                    title: { display: true, text: 'N° Socios', color: '#64748b' }
                },
                y1: {
                    position: 'right',
                    grid: { display: false },
                    title: { display: true, text: 'Tasa Mora %', color: '#64748b' },
                    max: 100,
                    ticks: { callback: v => v + '%' }
                },
                x: {
                    grid: { display: false },
                }
            },
            animation: { duration: 1200, easing: 'easeOutQuart' }
        }
    });
}

function renderSegmentsTable(productos) {
    if (!productos) return;
    const tbody = document.getElementById('segments-tbody');
    tbody.innerHTML = '';
    const sorted = [...productos].sort((a, b) => b.score_riesgo - a.score_riesgo);

    sorted.forEach(p => {
        const score = p.score_riesgo;
        let level, levelClass;
        if (score >= 0.8) { level = 'Crítico'; levelClass = 'critical'; }
        else if (score >= 0.5) { level = 'Alto'; levelClass = 'high'; }
        else if (score >= 0.2) { level = 'Medio'; levelClass = 'medium'; }
        else { level = 'Bajo'; levelClass = 'low'; }

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong>Producto ${p.prod_bancario}</strong></td>
            <td>${formatNumber(p.n_socios)}</td>
            <td>${p.tasa_mora_pct.toFixed(2)}%</td>
            <td>${(score * 100).toFixed(1)}%</td>
            <td><span class="risk-tag ${levelClass}">${level}</span></td>
        `;
        tbody.appendChild(tr);
    });
}

function renderClustersChart(clusters) {
    if (!clusters) return;
    destroyChart('clusters');
    const ctx = document.getElementById('canvas-clusters').getContext('2d');
    const labels = Object.keys(clusters).map(k => k.replace('cluster_', 'Cluster '));
    const values = Object.values(clusters);
    const colors = [COLORS.green, COLORS.teal, COLORS.orangeLight, COLORS.emerald, COLORS.orange];

    charts.clusters = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data: values,
                backgroundColor: colors.slice(0, labels.length),
                borderWidth: 0,
                hoverOffset: 8,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '60%',
            plugins: {
                legend: { position: 'bottom', labels: { padding: 14, font: { size: 11 } } },
            },
            animation: { animateRotate: true, duration: 1000, easing: 'easeOutQuart' }
        }
    });
}

function renderAlertasDetailChart(dist) {
    if (!dist) return;
    destroyChart('alertas-detail');
    const ctx = document.getElementById('canvas-alertas-detail').getContext('2d');
    const labels = Object.keys(dist).map(k => k.replace('_', ' '));
    const values = Object.values(dist);
    const colors = [COLORS.emerald, COLORS.orange, COLORS.orangeLight, COLORS.rose];

    charts['alertas-detail'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Socios',
                data: values,
                backgroundColor: colors.map(c => c + '88'),
                borderColor: colors,
                borderWidth: 1,
                borderRadius: 8,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
            },
            scales: {
                y: { grid: { color: 'rgba(46, 125, 50, 0.06)' } },
                x: { grid: { display: false } }
            },
            animation: { duration: 1200, easing: 'easeOutQuart' }
        }
    });
}

function renderAlertasTipoChart(dist) {
    if (!dist) return;
    destroyChart('alertas-tipo');
    const ctx = document.getElementById('canvas-alertas-tipo').getContext('2d');
    const labels = Object.keys(dist).map(k => k.replace('_', ' '));
    const values = Object.values(dist);
    const colors = [COLORS.emerald, COLORS.rose, COLORS.orangeLight, COLORS.orange, COLORS.teal];

    charts['alertas-tipo'] = new Chart(ctx, {
        type: 'polarArea',
        data: {
            labels,
            datasets: [{
                data: values,
                backgroundColor: colors.map(c => c + '66'),
                borderColor: colors,
                borderWidth: 1,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { padding: 12, font: { size: 10 } } },
            },
            scales: {
                r: {
                    grid: { color: 'rgba(46, 125, 50, 0.08)' },
                    ticks: { display: false }
                }
            },
            animation: { duration: 1000, easing: 'easeOutQuart' }
        }
    });
}

// ─── Scoring ────────────────────────────────────────────
async function scoreCliente(clienteId) {
    const loading = document.getElementById('scoring-loading');
    const result = document.getElementById('score-result');
    loading.classList.remove('hidden');
    result.classList.add('hidden');

    try {
        // Try GET by ID first
        let data;
        try {
            data = await fetchJSON(`/score/cliente/${clienteId}`);
        } catch {
            // Fallback to POST
            const res = await fetch(API_BASE + '/score/cliente', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ cliente_id: parseInt(clienteId), v_ah_cliente: parseInt(clienteId) })
            });
            if (!res.ok) throw new Error(`Score failed: ${res.status}`);
            data = await res.json();
        }

        loading.classList.add('hidden');
        result.classList.remove('hidden');
        renderScoreResult(data);
    } catch (e) {
        loading.classList.add('hidden');
        console.error('Score failed:', e);
        alert('Error al evaluar el socio. Verifique que el ID sea correcto.');
    }
}

function renderScoreResult(data) {
    // Summary
    document.getElementById('res-cliente-id').textContent = data.cliente_id || '—';
    // riesgo_global comes as a percentage (0-100), e.g. 8.35 means 8.35%
    const riesgoPct = data.riesgo_global !== undefined ? data.riesgo_global : 0;
    const riesgoNorm = riesgoPct / 100; // normalize to 0-1 for gauge
    const riesgoEl = document.getElementById('res-riesgo');
    riesgoEl.textContent = riesgoPct.toFixed(1) + '%';

    if (riesgoPct >= 70) {
        riesgoEl.style.background = 'var(--danger-bg)';
        riesgoEl.style.color = 'var(--danger)';
    } else if (riesgoPct >= 40) {
        riesgoEl.style.background = 'var(--warning-bg)';
        riesgoEl.style.color = 'var(--warning)';
    } else {
        riesgoEl.style.background = 'var(--success-bg)';
        riesgoEl.style.color = 'var(--success)';
    }

    document.getElementById('res-elegibilidad').textContent = data.elegibilidad_credito ? '✅ Elegible' : '❌ No elegible';
    document.getElementById('res-canal').textContent = data.canal_cobranza ? data.canal_cobranza.charAt(0).toUpperCase() + data.canal_cobranza.slice(1) : 'N/A';
    document.getElementById('res-dia-pago').textContent = data.dia_pago_sugerido ? `Día ${data.dia_pago_sugerido}` : 'N/A';

    // Gauge
    drawGauge(riesgoNorm);
    document.getElementById('gauge-value').textContent = riesgoPct.toFixed(0);

    // Lists
    renderList('res-alertas-list', data.alertas_activas, 'alert-item', 'Sin alertas activas');
    renderList('res-acciones-list', data.acciones_priorizadas, 'action-item', 'Sin acciones pendientes');
    renderList('res-bloqueos-list', data.bloqueos, 'block-item', 'Sin bloqueos');

    // Agent details
    renderAgentBreakdown(data.agentes);
}

function renderList(containerId, items, itemClass, emptyMsg) {
    const ul = document.getElementById(containerId);
    ul.innerHTML = '';
    if (!items || items.length === 0) {
        const li = document.createElement('li');
        li.className = 'empty-msg';
        li.textContent = emptyMsg;
        ul.appendChild(li);
        return;
    }
    items.forEach(item => {
        const li = document.createElement('li');
        li.className = itemClass;
        li.textContent = item;
        ul.appendChild(li);
    });
}

function drawGauge(value) {
    const canvas = document.getElementById('canvas-gauge');
    const ctx = canvas.getContext('2d');
    const size = 180;
    canvas.width = size * 2;
    canvas.height = size * 2;
    canvas.style.width = size + 'px';
    canvas.style.height = size + 'px';
    ctx.scale(2, 2);

    const cx = size / 2;
    const cy = size / 2;
    const r = 70;
    const lineWidth = 12;
    const startAngle = 0.75 * Math.PI;
    const endAngle = 2.25 * Math.PI;
    const range = endAngle - startAngle;

    // Background arc
    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, endAngle);
    ctx.strokeStyle = 'rgba(46, 125, 50, 0.12)';
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Value arc
    const valueAngle = startAngle + range * Math.min(value, 1);
    let color;
    if (value >= 0.7) color = '#EF5350';
    else if (value >= 0.4) color = '#F5A623';
    else color = '#4CAF50';

    const gradient = ctx.createLinearGradient(0, 0, size, size);
    gradient.addColorStop(0, color);
    gradient.addColorStop(1, color + 'aa');

    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, valueAngle);
    ctx.strokeStyle = gradient;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.stroke();
}

function renderAgentBreakdown(agentes) {
    const grid = document.getElementById('agents-grid');
    grid.innerHTML = '';

    if (!agentes || typeof agentes !== 'object') return;

    const agentColors = {
        credit_scoring: COLORS.green,
        early_warning: COLORS.orange,
        roll_rate: COLORS.orangeLight,
        overindebtedness: COLORS.rose,
        clustering: COLORS.teal,
        product_risk: COLORS.greenLight,
        age_validation: COLORS.emerald,
        date_optimization: COLORS.sky,
        family_impact: COLORS.pink,
    };

    for (const [key, val] of Object.entries(agentes)) {
        const card = document.createElement('div');
        card.className = 'agent-card';

        const color = agentColors[key] || COLORS.indigo;
        let details = '';

        if (typeof val === 'object' && val !== null) {
            for (const [k, v] of Object.entries(val)) {
                if (k === 'nombre' || k === 'agent_name') continue;
                const displayVal = typeof v === 'number' ? (v < 1 && v > 0 ? (v * 100).toFixed(1) + '%' : v.toFixed ? v.toFixed(2) : v) : String(v);
                details += `<strong>${k.replace(/_/g, ' ')}:</strong> ${displayVal}<br>`;
            }
        } else {
            details = String(val);
        }

        card.innerHTML = `
            <div class="agent-card-header">
                <div class="agent-dot" style="background: ${color}; box-shadow: 0 0 6px ${color}"></div>
                <span class="agent-name">${key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</span>
            </div>
            <div class="agent-detail">${details || 'Sin datos'}</div>
        `;
        grid.appendChild(card);
    }
}

// ─── Navigation ─────────────────────────────────────────
const viewTitles = {
    dashboard: { title: 'Dashboard', subtitle: 'Vista general del sistema de predicción de pago' },
    scoring: { title: 'Scoring Individual', subtitle: 'Evalúa el perfil de riesgo de un socio' },
    segments: { title: 'Segmentos & Clusters', subtitle: 'Análisis por producto bancario y agrupaciones' },
    alerts: { title: 'Alertas & Riesgo', subtitle: 'Distribución de alertas tempranas y niveles de mora' },
};

function switchView(viewName) {
    // Update nav
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    document.querySelector(`[data-view="${viewName}"]`).classList.add('active');

    // Update views
    document.querySelectorAll('.view').forEach(el => el.classList.remove('active'));
    document.getElementById(`view-${viewName}`).classList.add('active');

    // Update header
    const info = viewTitles[viewName] || viewTitles.dashboard;
    document.getElementById('page-title').textContent = info.title;
    document.getElementById('page-subtitle').textContent = info.subtitle;

    // Close mobile sidebar
    document.getElementById('sidebar').classList.remove('open');
    document.getElementById('sidebar-overlay').classList.remove('active');

    // Load view-specific data
    if (viewName === 'alerts' && dashboardData) {
        renderAlertasDetailChart(dashboardData.distribucion_riesgo);
        renderAlertasTipoChart(dashboardData.distribucion_alerta);
    }
}

// ─── Event Listeners ────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    // Navigation
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            switchView(item.dataset.view);
        });
    });

    // Mobile menu
    document.getElementById('menu-toggle').addEventListener('click', () => {
        document.getElementById('sidebar').classList.toggle('open');
        document.getElementById('sidebar-overlay').classList.toggle('active');
    });

    document.getElementById('sidebar-overlay').addEventListener('click', () => {
        document.getElementById('sidebar').classList.remove('open');
        document.getElementById('sidebar-overlay').classList.remove('active');
    });

    // Refresh button
    document.getElementById('btn-refresh').addEventListener('click', async function () {
        this.classList.add('spinning');
        await Promise.all([loadHealth(), loadDashboard(), loadSegmentos(), loadClusters()]);
        this.classList.remove('spinning');
    });

    // Scoring
    document.getElementById('btn-score').addEventListener('click', () => {
        const id = document.getElementById('input-cliente-id').value.trim();
        if (!id) return;
        scoreCliente(id);
    });

    document.getElementById('input-cliente-id').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const id = e.target.value.trim();
            if (id) scoreCliente(id);
        }
    });

    // ── Initial Load ──
    await loadHealth();
    await Promise.all([loadDashboard(), loadSegmentos(), loadClusters()]);
});
