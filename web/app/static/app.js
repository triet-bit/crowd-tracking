/* =======================================
   CrowdTrack – app.js (Full Features)
   ======================================= */

// ---- State ----
let ws = null;
let alertCount = 0;
let densityChart = null;
let currentCameraId = 'cam_001';
let currentView = 'dashboard';

// Zone drawing state
let isDrawingZone = false;
let zonePoints = [];
let zoneCanvasCtx = null;

const chartData = {
    labels: [],
    datasets: [{
        label: 'Số người',
        data: [],
        borderColor: '#1a73e8',
        backgroundColor: 'rgba(26,115,232,0.08)',
        tension: 0.4,
        fill: true,
        pointBackgroundColor: '#1a73e8',
        pointRadius: 2,
        pointHoverRadius: 5,
    }]
};

// ---- DOM refs ----
const $ = id => document.getElementById(id);

const UI = {
    wsStatusText:    $('ws-status-text'),
    wsDot:           $('ws-dot'),
    btnStart:        $('btn-start'),
    btnStop:         $('btn-stop'),
    cameraInput:     $('camera-id-input'),
    metricCount:     $('metric-count'),
    metricTracks:    $('metric-tracks'),
    metricEnter:     $('metric-enter'),
    metricExit:      $('metric-exit'),
    alertsContainer: $('alerts-container'),
    alertBadge:      $('alert-badge'),
    navAlertCount:   $('nav-alert-count'),
    currentTime:     $('current-time'),
    pageTitle:       $('page-title'),
    liveFeed:        $('live-feed'),
    videoPlaceholder:$('video-placeholder'),
    videoContainer:  $('video-container'),
    zoneCanvas:      $('zone-canvas'),
    streamBadge:     $('stream-badge'),
    btnDrawZone:     $('btn-draw-zone'),
    btnClearZone:    $('btn-clear-zone'),
    btnSaveZone:     $('btn-save-zone'),
};

// ---- Clock ----
function updateClock() {
    const now = new Date();
    UI.currentTime.textContent = now.toLocaleString('vi-VN', {
        hour: '2-digit', minute: '2-digit', second: '2-digit',
        day: '2-digit', month: '2-digit', year: 'numeric'
    });
}
updateClock();
setInterval(updateClock, 1000);

// ---- Chart ----
function initChart() {
    const ctx = $('densityChart').getContext('2d');
    densityChart = new Chart(ctx, {
        type: 'line',
        data: chartData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#1e3a5f',
                    titleColor: '#fff',
                    bodyColor: '#bfdbfe',
                    borderColor: '#1a73e8',
                    borderWidth: 1,
                    padding: 10,
                }
            },
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(30,58,95,0.06)' }, ticks: { color: '#8faabf', font: { size: 11 } } },
                x: { grid: { display: false }, ticks: { color: '#8faabf', font: { size: 11 }, maxTicksLimit: 10 } }
            }
        }
    });
}

function updateChart(count) {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    chartData.labels.push(timeStr);
    chartData.datasets[0].data.push(count);
    if (chartData.labels.length > 30) {
        chartData.labels.shift();
        chartData.datasets[0].data.shift();
    }
    densityChart.update('none');
}

// ---- Live Video Stream ----
function startVideoStream(cameraId) {
    UI.liveFeed.src = `/api/cameras/${cameraId}/stream?t=${Date.now()}`;
    UI.liveFeed.style.display = 'block';
    UI.videoPlaceholder.style.display = 'none';
    UI.streamBadge.style.display = 'inline';

    // Resize canvas to match video
    UI.liveFeed.onload = () => resizeZoneCanvas();
}

function stopVideoStream() {
    UI.liveFeed.src = '';
    UI.liveFeed.style.display = 'none';
    UI.videoPlaceholder.style.display = 'flex';
    UI.streamBadge.style.display = 'none';
}

function resizeZoneCanvas() {
    const container = UI.videoContainer;
    const img = UI.liveFeed;
    UI.zoneCanvas.width = img.clientWidth || container.clientWidth;
    UI.zoneCanvas.height = img.clientHeight || container.clientHeight;
    zoneCanvasCtx = UI.zoneCanvas.getContext('2d');
    redrawZone();
}

// ---- Zone Drawing on Canvas ----
function enableZoneDrawing() {
    isDrawingZone = true;
    zonePoints = [];
    UI.zoneCanvas.classList.add('drawing');
    UI.btnDrawZone.style.display = 'none';
    UI.btnClearZone.style.display = 'inline-flex';
    UI.btnSaveZone.style.display = 'inline-flex';
    resizeZoneCanvas();
}

function disableZoneDrawing() {
    isDrawingZone = false;
    UI.zoneCanvas.classList.remove('drawing');
    UI.btnDrawZone.style.display = 'inline-flex';
    UI.btnClearZone.style.display = 'none';
    UI.btnSaveZone.style.display = 'none';
}

function clearZone() {
    zonePoints = [];
    redrawZone();
}

function redrawZone() {
    if (!zoneCanvasCtx) return;
    const ctx = zoneCanvasCtx;
    ctx.clearRect(0, 0, UI.zoneCanvas.width, UI.zoneCanvas.height);

    if (zonePoints.length === 0) return;

    // Draw filled polygon
    ctx.fillStyle = 'rgba(26, 115, 232, 0.15)';
    ctx.strokeStyle = '#1a73e8';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 3]);

    ctx.beginPath();
    ctx.moveTo(zonePoints[0].x, zonePoints[0].y);
    for (let i = 1; i < zonePoints.length; i++) {
        ctx.lineTo(zonePoints[i].x, zonePoints[i].y);
    }
    if (zonePoints.length > 2) {
        ctx.closePath();
        ctx.fill();
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw points
    zonePoints.forEach((p, i) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = i === 0 ? '#ef4444' : '#1a73e8';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
    });

    // Instruction text
    if (isDrawingZone) {
        ctx.fillStyle = 'rgba(0,0,0,0.6)';
        ctx.fillRect(10, 10, 280, 28);
        ctx.fillStyle = '#fff';
        ctx.font = '12px Inter, sans-serif';
        ctx.fillText(`Click để thêm điểm (${zonePoints.length} điểm) • Cần ≥ 3`, 18, 28);
    }
}

// Canvas click handler
UI.zoneCanvas.addEventListener('click', (e) => {
    if (!isDrawingZone) return;
    const rect = UI.zoneCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    zonePoints.push({ x, y });
    redrawZone();
});

// Convert canvas coords to original image coords
function zonePointsToPolygon() {
    const img = UI.liveFeed;
    const canvasW = UI.zoneCanvas.width;
    const canvasH = UI.zoneCanvas.height;

    // If no video loaded, use canvas dimensions as-is (mock scenario)
    const imgW = img.naturalWidth || canvasW;
    const imgH = img.naturalHeight || canvasH;

    const scaleX = imgW / canvasW;
    const scaleY = imgH / canvasH;

    return zonePoints.map(p => [Math.round(p.x * scaleX), Math.round(p.y * scaleY)]);
}

async function saveZone() {
    if (zonePoints.length < 3) {
        alert('Cần ít nhất 3 điểm để tạo zone.');
        return;
    }

    const camId = UI.cameraInput.value.trim();
    if (!camId) return;

    const polygon = zonePointsToPolygon();
    try {
        const result = await api('POST', `/api/cameras/${camId}/zones`, {
            name: `Zone ${Date.now().toString(36)}`,
            polygon: polygon,
            max_people_threshold: 10,
            loitering_time_threshold: 30,
        });
        console.log('Zone saved:', result);
        alert(`Zone đã lưu: ${result.id}`);
        disableZoneDrawing();
    } catch (e) {
        console.error('Save zone failed:', e);
        alert('Lỗi lưu zone');
    }
}

// ---- WebSocket ----
function connectWebSocket(cameraId) {
    if (ws) { ws.close(); ws = null; }

    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${location.host}/ws/cameras/${cameraId}`);

    ws.onopen = () => setWsStatus(true);
    ws.onclose = () => {
        setWsStatus(false);
        setTimeout(() => connectWebSocket(currentCameraId), 5000);
    };
    ws.onerror = () => setWsStatus(false);
    ws.onmessage = (e) => {
        try {
            const payload = JSON.parse(e.data);
            if (payload.event === 'frame.metrics') updateMetrics(payload.data);
            else if (payload.event === 'alert.created') addAlert(payload.data);
        } catch (_) {}
    };

    // Keep-alive ping every 20s
    setInterval(() => { if (ws && ws.readyState === 1) ws.send('ping'); }, 20000);
}

function setWsStatus(online) {
    UI.wsDot.className = 'ws-dot' + (online ? ' connected' : '');
    UI.wsStatusText.textContent = online ? 'Live' : 'Offline';
}

// ---- Metrics ----
function animateValue(el, val) {
    el.classList.add('updating');
    el.textContent = val;
    setTimeout(() => el.classList.remove('updating'), 400);
}

function updateMetrics(data) {
    animateValue(UI.metricCount,  data.people_count  ?? 0);
    animateValue(UI.metricTracks, data.total_tracks   ?? 0);
    animateValue(UI.metricEnter,  data.enter_count    ?? 0);
    animateValue(UI.metricExit,   data.exit_count     ?? 0);
    updateChart(data.people_count ?? 0);
}

// ---- Alerts ----
function formatAlertType(type) {
    return (type || 'alert').replace(/_/g, ' ');
}

function addAlert(alert) {
    const empty = UI.alertsContainer.querySelector('.empty-state');
    if (empty) empty.remove();

    alertCount++;
    UI.alertBadge.textContent = alertCount;
    UI.navAlertCount.textContent = alertCount;

    const severity = alert.severity || 'medium';
    const timeStr = new Date().toLocaleTimeString('vi-VN');

    const div = document.createElement('div');
    div.className = `alert-item ${severity}`;
    div.innerHTML = `
        <div class="alert-header">
            <span class="alert-type">${formatAlertType(alert.alert_type)}</span>
            <span class="alert-time">${timeStr}</span>
        </div>
        <div class="alert-msg">${alert.message || '–'}</div>
        <div class="alert-meta">Camera: ${alert.camera_id || '–'} • ${alert.people_count ?? '?'} người</div>
    `;
    UI.alertsContainer.prepend(div);
    appendAlertHistory(div.cloneNode(true));
}

function appendAlertHistory(node) {
    const hist = $('alert-history-list');
    const empty = hist?.querySelector('.empty-state');
    if (empty) empty.remove();
    hist?.prepend(node);
}

// ---- API helpers ----
async function api(method, path, body) {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body) opts.body = JSON.stringify(body);
    const res = await fetch(path, opts);
    return res.json();
}

// ---- Load cameras ----
async function loadCameras() {
    const list = $('camera-list');
    try {
        const cameras = await api('GET', '/api/cameras');
        if (!cameras.length) {
            list.innerHTML = '<div class="empty-state"><p>Chưa có camera nào. Nhấn "+ Thêm camera" để thêm mới.</p></div>';
            return;
        }
        list.innerHTML = '';
        cameras.forEach(cam => {
            const zonesInfo = cam.zones && cam.zones.length
                ? `${cam.zones.length} zone(s)`
                : 'Chưa có zone';
            const item = document.createElement('div');
            item.className = 'camera-item';
            item.innerHTML = `
                <div class="camera-info">
                    <div class="camera-name">${cam.name || cam.id}</div>
                    <div class="camera-meta">${cam.id} • ${cam.source_type} • ${cam.mode} • ${zonesInfo}</div>
                </div>
                <span class="camera-status ${cam.is_active ? 'running' : 'stopped'}">
                    ${cam.is_active ? 'Running' : 'Stopped'}
                </span>
                <div class="camera-actions">
                    <button class="btn btn-primary btn-sm" onclick="selectAndStartCamera('${cam.id}')">Start</button>
                    <button class="btn btn-ghost btn-sm" onclick="stopCamera('${cam.id}')">Stop</button>
                </div>
            `;
            list.appendChild(item);
        });
    } catch (e) {
        list.innerHTML = '<div class="empty-state"><p>Lỗi khi tải danh sách camera.</p></div>';
    }
}

// ---- Load alerts history ----
async function loadAlertHistory() {
    const list = $('alert-history-list');
    try {
        const camId = UI.cameraInput.value || null;
        const alerts = await api('GET', `/api/alerts?limit=50${camId ? '&camera_id=' + camId : ''}`);
        if (!alerts.length) {
            list.innerHTML = '<div class="empty-state"><p>Chưa có lịch sử cảnh báo.</p></div>';
            return;
        }
        list.innerHTML = '';
        alerts.forEach(a => {
            const div = document.createElement('div');
            div.className = `alert-item ${a.severity || 'medium'}`;
            div.innerHTML = `
                <div class="alert-header">
                    <span class="alert-type">${formatAlertType(a.alert_type)}</span>
                    <span class="alert-time">${a.created_at ? new Date(a.created_at).toLocaleString('vi-VN') : '–'}</span>
                </div>
                <div class="alert-msg">${a.message || '–'}</div>
                <div class="alert-meta">Camera: ${a.camera_id || '–'} • Zone: ${a.zone_id || '–'} • ${a.people_count ?? '?'} người</div>
            `;
            list.appendChild(div);
        });
    } catch (e) {
        list.innerHTML = '<div class="empty-state"><p>Lỗi khi tải lịch sử.</p></div>';
    }
}

// ---- Camera actions ----
async function selectAndStartCamera(camId) {
    UI.cameraInput.value = camId;
    currentCameraId = camId;
    switchView('dashboard');
    await startCameraStream(camId);
}

async function startCameraStream(camId) {
    try {
        await api('POST', `/api/cameras/${camId}/start`);
        connectWebSocket(camId);
        startVideoStream(camId);
    } catch (e) { console.error('Start failed:', e); }
}

async function stopCamera(camId) {
    try {
        await api('POST', `/api/cameras/${camId}/stop`);
        if (camId === currentCameraId) stopVideoStream();
        setTimeout(loadCameras, 800);
    } catch (e) { console.error(e); }
}

// ---- Navigation ----
function switchView(name) {
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

    const viewEl = $(`view-${name}`);
    const navEl  = $(`nav-${name}`);
    if (viewEl) viewEl.classList.add('active');
    if (navEl) navEl.classList.add('active');

    const titles = { dashboard: 'Dashboard', cameras: 'Cameras', alerts: 'Cảnh báo' };
    UI.pageTitle.textContent = titles[name] || name;
    currentView = name;

    if (name === 'cameras') loadCameras();
    if (name === 'alerts') loadAlertHistory();
}

// ---- Event Listeners ----
$('nav-dashboard').addEventListener('click', e => { e.preventDefault(); switchView('dashboard'); });
$('nav-cameras').addEventListener('click',   e => { e.preventDefault(); switchView('cameras'); });
$('nav-alerts').addEventListener('click',    e => { e.preventDefault(); switchView('alerts'); });

UI.btnStart.addEventListener('click', async () => {
    const camId = UI.cameraInput.value.trim();
    if (!camId) return;
    currentCameraId = camId;
    await startCameraStream(camId);
});

UI.btnStop.addEventListener('click', async () => {
    const camId = UI.cameraInput.value.trim();
    if (!camId) return;
    await stopCamera(camId);
});

// Zone drawing buttons
UI.btnDrawZone.addEventListener('click', () => enableZoneDrawing());
UI.btnClearZone.addEventListener('click', () => clearZone());
UI.btnSaveZone.addEventListener('click', () => saveZone());

// Camera management
$('btn-add-camera').addEventListener('click', () => {
    const form = $('add-camera-form');
    form.style.display = form.style.display === 'none' ? 'block' : 'none';
});

$('btn-cancel-camera').addEventListener('click', () => {
    $('add-camera-form').style.display = 'none';
});

$('btn-create-camera').addEventListener('click', async () => {
    const payload = {
        name: $('new-cam-name').value || 'Camera mới',
        source_type: $('new-cam-type').value,
        source_url: $('new-cam-url').value,
        mode: $('new-cam-mode').value,
    };
    try {
        await api('POST', '/api/cameras', payload);
        $('add-camera-form').style.display = 'none';
        loadCameras();
    } catch (e) { console.error('Create camera failed:', e); }
});

$('btn-test-telegram').addEventListener('click', async () => {
    try {
        const d = await api('POST', '/api/telegram/test');
        alert('Telegram Test: ' + d.status);
    } catch (e) { alert('Lỗi test Telegram'); }
});

$('btn-test-hardware').addEventListener('click', async () => {
    try {
        const camId = UI.cameraInput.value || 'cam_test';
        const d = await api('POST', `/api/hardware/test-alert?camera_id=${camId}`);
        alert('Hardware Test: ' + d.status);
    } catch (e) { alert('Lỗi test Hardware'); }
});

// Window resize → resize canvas
window.addEventListener('resize', () => {
    if (UI.liveFeed.style.display !== 'none') resizeZoneCanvas();
});

// ---- Init ----
initChart();
connectWebSocket(currentCameraId);
